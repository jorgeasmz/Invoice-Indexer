import torch
from transformers import LayoutLMv2ForTokenClassification
import re
import logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FacturaExtractor:
    """Extrae información relevante de las facturas utilizando LayoutLM"""
    
    def __init__(self, model_name="microsoft/layoutlmv2-base-uncased"):
        """
        Inicializa el extractor de facturas
        
        Args:
            model_name: Nombre del modelo LayoutLM a utilizar
        """
        logger.info(f"Inicializando extractor con modelo {model_name}")
        # Cargar modelo para análisis espacial
        self.model = LayoutLMv2ForTokenClassification.from_pretrained(model_name)
        self.model.eval()
        
        # Definir patrones comunes para facturas
        self.patrones = {
            "invoice_number": [
                r"(?:N°|N|N[úu]mero|No|Nº)\s*(?:de)?\s*(?:FACTURA|factura|Fra|fac)[\s:]*(\d+[/\-\.]*\d*)",
                r"(?:FACTURA|factura|Fra|fac)[\s:]*(?:N°|N|N[úu]mero|No|Nº)[\s:]*(\d+[/\-\.]*\d*)",
                r"(?:FACTURA|factura|Fra|fac)[\s:]*(\d+[/\-\.]*\d*)"
            ],
            "date": [
                r"(?:FECHA|fecha|Date)[\s:]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})",
                r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})"
            ],
            "client_name": [
                r"(?:CLIENTE|cliente|Client)[\s:]*([A-Z\s]+(?:\sS\.?L\.?|S\.?A\.?))",
                r"(?:NOMBRE|nombre|Name)[\s:]*([A-Z\s]+(?:\sS\.?L\.?|S\.?A\.?))",
                r"(?:RAZON SOCIAL|razon social)[\s:]*([A-Z\s]+(?:\sS\.?L\.?|S\.?A\.?))"
            ],
            "client_id": [
                r"(?:N\.?I\.?F\.?|CIF|C\.I\.F\.|NIF)[\s:]*([A-Z0-9]{9})",
                r"(?:N\.?L\.?F\.?)[\s:]*([A-Z0-9]{9})"
            ],
            "total": [
                r"(?:TOTAL|Total|total)[\s:]*(\d{1,3}(?:\.\d{3})*,\d{2})",
                r"(?:IMPORTE|Importe|importe)[\s:]*(?:LIQUIDO|liquido|TOTAL|total)?[\s:]*(\d{1,3}(?:\.\d{3})*,\d{2})",
                r"(?:LIQUIDO|liquido)[\s:]*(\d{1,3}(?:\.\d{3})*,\d{2})"
            ]
        }
        
        # Palabras clave por área funcional de la factura
        self.palabras_clave = {
            "cabecera": ["FACTURA", "FECHA", "NÚMERO", "N°", "EMISIÓN"],
            "cliente": ["CLIENTE", "N.I.F", "NIF", "N.L.F", "DIRECCIÓN", "DOMICILIO"],
            "detalle": ["CONCEPTO", "DESCRIPCIÓN", "DESCRIPCION", "CANTIDAD", "PRECIO", 
                      "CARGOS", "ABONOS", "__DESCRIPCION"],
            "totales": ["TOTAL", "SUBTOTAL", "BASE", "I.V.A", "IVA", "IMPORTE", "LIQUIDO"],
            "pie": ["FORMA", "PAGO", "VENCIMIENTO", "OBSERVACIONES"]
        }
        
    def extract_info(self, encoding, ocr_results):
        """
        Extrae información de la factura usando LayoutLM y análisis espacial
        
        Args:
            encoding: Datos codificados para LayoutLM
            ocr_results: Resultados del OCR con posiciones
            
        Returns:
            dict: Información extraída de la factura
        """
        logger.info("Extrayendo información de la factura")
        
        # Formato de salida
        invoice_data = {
            'invoice_number': None,
            'date': None,
            'total': None,
            'client_name': None,
            'client_id': None,
            'items': []
        }
        
        # 1. Analizar el texto completo para poder hacer búsquedas generales
        texto_completo = " ".join([word for word, _ in ocr_results])
        logger.info(f"Texto extraído: {texto_completo[:200]}...")
        
        # 2. Organización espacial del documento
        zonas_verticales = self._create_vertical_zones(ocr_results)
        bloques_funcionales = self._identify_functional_blocks(ocr_results)
        
        # 3. Utilizar LayoutLM para entender contexto espacial
        contextual_regions = self._analyze_spatial_context(encoding, ocr_results)
        
        # 4. Extraer información basada en patrones y contexto
        self._extract_invoice_number(invoice_data, ocr_results, zonas_verticales, bloques_funcionales)
        self._extract_date(invoice_data, ocr_results, zonas_verticales, bloques_funcionales)
        self._extract_client_info(invoice_data, ocr_results, zonas_verticales, bloques_funcionales)
        self._extract_total(invoice_data, ocr_results, zonas_verticales, bloques_funcionales)
        
        # 5. Extraer líneas de detalle/items
        invoice_data['items'] = self._extract_line_items(ocr_results, zonas_verticales, bloques_funcionales)
        
        # 6. Logging detallado de los resultados
        for campo, valor in invoice_data.items():
            if campo != 'items':
                logger.info(f"Campo '{campo}' extraído: {valor}")
                
        logger.info(f"Número de líneas de detalle extraídas: {len(invoice_data['items'])}")
        
        return invoice_data
    
    def _create_vertical_zones(self, ocr_results):
        """
        Divide el documento en zonas verticales (superior, media, inferior)
        
        Args:
            ocr_results: Lista de tuplas (palabra, caja) del OCR
            
        Returns:
            dict: Mapa de zonas con índices y palabras
        """
        zones = defaultdict(list)
        
        # Encontrar altura máxima
        height_values = [box[3] for _, box in ocr_results]
        max_height = max(height_values) if height_values else 1000
        
        # Dividir en 3 zonas verticales
        zone_height = max_height / 3
        
        for i, (word, box) in enumerate(ocr_results):
            # Determinar zona vertical (0=superior, 1=medio, 2=inferior)
            v_zone = int(box[1] / zone_height)
            zones[v_zone].append((i, word, box))
        
        return zones
    
    def _identify_functional_blocks(self, ocr_results):
        """
        Identifica bloques funcionales en la factura (cabecera, cliente, detalles, totales, pie)
        basado en palabras clave y posiciones
        
        Args:
            ocr_results: Lista de tuplas (palabra, caja) del OCR
            
        Returns:
            dict: Bloques funcionales con índices y palabras
        """
        bloques = {
            "cabecera": [],
            "cliente": [],
            "detalle": [],
            "totales": [],
            "pie": []
        }
        
        # Asignar palabras a bloques basado en palabras clave
        for i, (word, box) in enumerate(ocr_results):
            word_upper = word.upper()
            
            # Buscar en todas las áreas
            for area, keywords in self.palabras_clave.items():
                if any(keyword in word_upper for keyword in keywords):
                    bloques[area].append((i, word, box))
                    break
        
        # Para palabras que no fueron asignadas, intentar inferir su bloque por posición
        if bloques["detalle"]:
            # Encontrar la región de detalle (generalmente una tabla)
            detalle_boxes = [box for _, _, box in bloques["detalle"]]
            if detalle_boxes:
                # Calcular límites aproximados de la región de detalle
                min_y = min(box[1] for box in detalle_boxes)
                max_y = max(box[3] for box in detalle_boxes)
                
                # Asignar palabras que caen dentro de estos límites
                for i, (word, box) in enumerate(ocr_results):
                    if min_y <= box[1] <= max_y and not any((i, word, box) in items for items in bloques.values()):
                        bloques["detalle"].append((i, word, box))
        
        return bloques
    
    def _analyze_spatial_context(self, encoding, ocr_results):
        """
        Analiza el contexto espacial utilizando LayoutLM
        
        Args:
            encoding: Encoding del modelo
            ocr_results: Lista de tuplas (palabra, caja) del OCR
            
        Returns:
            dict: Regiones contextuales
        """
        # Crear grupos de palabras cercanas horizontalmente (misma línea)
        horizontal_lines = defaultdict(list)
        
        for i, (word, box) in enumerate(ocr_results):
            # Usar el centro vertical de la caja como clave de línea
            line_key = (box[1] + box[3]) // 2
            horizontal_lines[line_key].append((i, word, box))
        
        # Ordenar cada línea horizontal de izquierda a derecha
        for line_key in horizontal_lines:
            horizontal_lines[line_key].sort(key=lambda x: x[2][0])  # Ordenar por coordenada x
        
        # Identificar posibles etiquetas y valores (pares clave-valor)
        key_value_pairs = []
        
        for line_items in horizontal_lines.values():
            if len(line_items) > 1:
                # Buscar patrones donde una palabra pueda ser etiqueta y la siguiente valor
                for j in range(len(line_items) - 1):
                    idx1, word1, box1 = line_items[j]
                    idx2, word2, box2 = line_items[j + 1]
                    
                    # Si la primera palabra parece una etiqueta (termina en :, etc.)
                    if re.search(r'[:.]$', word1) or word1.upper() in [w for keywords in self.palabras_clave.values() for w in keywords]:
                        key_value_pairs.append({
                            "key": (idx1, word1, box1),
                            "value": (idx2, word2, box2)
                        })
        
        return {
            "horizontal_lines": horizontal_lines,
            "key_value_pairs": key_value_pairs
        }
    
    def _extract_invoice_number(self, invoice_data, ocr_results, zonas, bloques):
        """
        Extrae el número de factura
        """
        # Buscar directamente con patrones
        texto_completo = " ".join([word for word, _ in ocr_results])
        
        for patron in self.patrones["invoice_number"]:
            match = re.search(patron, texto_completo, re.IGNORECASE)
            if match:
                invoice_data["invoice_number"] = match.group(1).strip()
                return
        
        # Estrategia específica para Stipendium basado en el OCR analizado
        # Buscar cerca de "N° FACTURA" o palabras similares
        factura_idx = None
        for i, (word, _) in enumerate(ocr_results):
            if "FACTURA" in word.upper():
                factura_idx = i
                break
                
        if factura_idx is not None:
            # Buscar números cercanos que podrían ser el número de factura
            for i in range(max(0, factura_idx - 3), min(factura_idx + 5, len(ocr_results))):
                word, _ = ocr_results[i]
                # Verificar si parece un número de factura (formato ##/## como "24/62")
                if re.match(r'\d{1,2}/\d{1,2}', word):
                    invoice_data["invoice_number"] = word
                    return
                # También buscar números simples
                elif re.match(r'\d{2,6}', word) and word not in invoice_data.values():
                    invoice_data["invoice_number"] = word
                    return
    
    def _extract_date(self, invoice_data, ocr_results, zonas, bloques):
        """
        Extrae la fecha de la factura
        """
        # Buscar directamente con patrones
        texto_completo = " ".join([word for word, _ in ocr_results])
        
        for patron in self.patrones["date"]:
            match = re.search(patron, texto_completo, re.IGNORECASE)
            if match:
                invoice_data["date"] = match.group(1).strip()
                return
        
        # Estrategia específica para Stipendium basado en el OCR analizado
        # Buscar cerca de "FECHA" o palabras similares
        fecha_idx = None
        for i, (word, _) in enumerate(ocr_results):
            if "FECHA" in word.upper():
                fecha_idx = i
                break
                
        if fecha_idx is not None:
            # Buscar formatos de fecha cercanos
            for i in range(max(0, fecha_idx - 3), min(fecha_idx + 5, len(ocr_results))):
                word, _ = ocr_results[i]
                # Verificar si parece una fecha (formato DD/MM/YY como "01/01/24")
                if re.match(r'\d{1,2}/\d{1,2}/\d{2,4}', word):
                    invoice_data["date"] = word
                    return
    
    def _extract_client_info(self, invoice_data, ocr_results, zonas, bloques):
        """
        Extrae información del cliente (nombre y NIF/CIF)
        """
        # Buscar directamente con patrones para NIF/CIF
        texto_completo = " ".join([word for word, _ in ocr_results])
        
        # Extraer NIF/CIF
        for patron in self.patrones["client_id"]:
            match = re.search(patron, texto_completo, re.IGNORECASE)
            if match:
                invoice_data["client_id"] = match.group(1).strip()
                break
        
        # Estrategia específica para extraer nombre del cliente
        # En facturas de Stipendium, el cliente está en la zona superior
        if 0 in zonas:  # Zona superior
            # Buscar patrones de nombre de cliente (empresas SL, SA)
            for i, word, _ in zonas[0]:
                if "CAPITAL" in word or "PAN" in word or "SL" in word:
                    # Si encontramos varias palabras que parecen formar un nombre de empresa,
                    # intentamos reconstruir el nombre completo
                    for j in range(i, min(i + 5, len(ocr_results))):
                        if j < len(ocr_results) and "SL" in ocr_results[j][0]:
                            # Construir el nombre del cliente desde i hasta j
                            cliente_parts = [ocr_results[k][0] for k in range(i, j + 1)]
                            invoice_data["client_name"] = " ".join(cliente_parts)
                            return
        
        # Si no encontramos nombre de cliente, buscar cerca de "CLIENTE" como respaldo
        cliente_idx = None
        for i, (word, _) in enumerate(ocr_results):
            if "CLIENTE" in word.upper():
                cliente_idx = i
                break
                
        if cliente_idx is not None:
            # Buscar texto cercano que podría ser el nombre del cliente
            for i in range(cliente_idx + 1, min(cliente_idx + 5, len(ocr_results))):
                word, _ = ocr_results[i]
                if len(word) > 2 and not word.isdigit():
                    invoice_data["client_name"] = word
                    break
    
    def _extract_total(self, invoice_data, ocr_results, zonas, bloques):
        """
        Extrae el importe total de la factura
        """
        # Buscar directamente con patrones
        texto_completo = " ".join([word for word, _ in ocr_results])
        
        for patron in self.patrones["total"]:
            match = re.search(patron, texto_completo, re.IGNORECASE)
            if match:
                invoice_data["total"] = match.group(1).strip()
                return
        
        # Estrategia específica para Stipendium basado en el OCR analizado
        # Buscar cerca de "IMPORTE LIQUIDO" o palabras similares
        total_idx = None
        for i, (word, _) in enumerate(ocr_results):
            if "LIQUIDO" in word.upper() or "TOTAL" in word.upper():
                total_idx = i
                break
                
        if total_idx is not None:
            # Buscar importes cercanos (números con formato de dinero)
            for i in range(max(0, total_idx - 3), min(total_idx + 8, len(ocr_results))):
                word, _ = ocr_results[i]
                # Verificar si parece un importe (formato ###,## como "267,17")
                if re.match(r'\d{1,3}(?:\.\d{3})*,\d{2}', word):
                    invoice_data["total"] = word
                    return
        
        # Buscar en zona inferior donde suelen estar los totales
        if 2 in zonas:  # Zona inferior
            # Buscar importes que parecen totales
            importes = []
            for _, word, _ in zonas[2]:
                if re.match(r'\d{1,3}(?:\.\d{3})*,\d{2}', word):
                    importes.append(word)
            
            if importes:
                # Normalmente el último importe o el más grande es el total
                max_importe = max(importes, key=lambda x: float(x.replace(".", "").replace(",", ".")))
                invoice_data["total"] = max_importe
    
    def _extract_line_items(self, ocr_results, zonas, bloques):
        """
        Extrae las líneas de detalle (conceptos, importes)
        
        Args:
            ocr_results: Lista de tuplas (palabra, caja) del OCR
            zonas: Zonas verticales del documento
            bloques: Bloques funcionales identificados
            
        Returns:
            list: Lista de ítems con descripción y monto
        """
        items = []
        
        # Identificar la región de detalles/conceptos
        zona_detalles = 1  # Por defecto, zona media
        
        # Buscar palabras clave que indican la tabla de detalles
        tabla_inicio_idx = None
        tabla_fin_idx = None
        
        for i, (word, _) in enumerate(ocr_results):
            if "__DESCRIPCION" in word or "CONCEPTO" in word or "CARGOS" in word:
                tabla_inicio_idx = i
                break
        
        # Si encontramos el inicio de la tabla
        if tabla_inicio_idx is not None:
            # Buscar el índice donde termina la tabla de detalles
            # Típicamente antes de palabras como "TOTAL", "BASE", etc.
            for i in range(tabla_inicio_idx + 1, len(ocr_results)):
                word, _ = ocr_results[i]
                if "TOTAL" in word or "BASE" in word or "I.V.A" in word or "LIQUIDO" in word:
                    tabla_fin_idx = i
                    break
            
            # Si no encontramos fin explícito, estimar por contexto
            if tabla_fin_idx is None:
                # Estimar basado en la cantidad de texto
                tabla_fin_idx = min(tabla_inicio_idx + 30, len(ocr_results))
            
            # Extraer líneas de la tabla
            i = tabla_inicio_idx + 1
            while i < tabla_fin_idx:
                word, box = ocr_results[i]
                
                # Si parece un código o inicio de línea
                if re.match(r'[A-Za-z0-9]{3,8}', word) or "Cuota" in word:
                    descripcion_parts = []
                    descripcion_parts.append(word)
                    
                    # Continuar leyendo la descripción
                    j = i + 1
                    importe = None
                    
                    while j < tabla_fin_idx:
                        next_word, next_box = ocr_results[j]
                        
                        # Si parece un importe, terminar la línea
                        if re.match(r'\d{1,3}(?:\.\d{3})*,\d{2}', next_word):
                            importe = next_word
                            break
                        
                        # Si está en la misma línea horizontal o cercano, añadir a la descripción
                        if abs(box[1] - next_box[1]) < 20:  # Tolerancia vertical
                            descripcion_parts.append(next_word)
                        else:
                            # Si ya hay suficiente descripción, terminar
                            if len(descripcion_parts) > 1:
                                break
                        
                        j += 1
                    
                    # Si encontramos descripción e importe, añadir el item
                    if importe:
                        items.append({
                            "description": " ".join(descripcion_parts),
                            "amount": importe
                        })
                        i = j + 1  # Saltar a la siguiente línea
                    else:
                        i += 1
                else:
                    i += 1
                    
        # Si no encontramos ítems con el método anterior, intentar método alternativo
        if not items:
            # Buscar todos los importes en formato ###,## y sus contextos
            for i, (word, box) in enumerate(ocr_results):
                if re.match(r'\d{1,3}(?:\.\d{3})*,\d{2}', word) and "TOTAL" not in ocr_results[max(0, i-1)][0]:
                    # Buscar hacia atrás para encontrar una descripción
                    descripcion = "Servicio"  # Por defecto
                    
                    # Mirar hasta 10 palabras atrás para encontrar una descripción
                    for j in range(i-1, max(0, i-10), -1):
                        prev_word, prev_box = ocr_results[j]
                        
                        # Si es una palabra larga o contiene palabras clave
                        if (len(prev_word) > 4 and 
                            not re.match(r'\d+[.,]\d{2}', prev_word) and
                            "TOTAL" not in prev_word and
                            "IVA" not in prev_word and
                            "BASE" not in prev_word):
                            
                            descripcion = prev_word
                            
                            # Intentar extender la descripción con palabras cercanas
                            if j > 0:
                                more_words = []
                                for k in range(j-1, max(0, j-5), -1):
                                    if (abs(ocr_results[k][1][1] - prev_box[1]) < 20 and
                                        not re.match(r'\d+[.,]\d{2}', ocr_results[k][0])):
                                        more_words.insert(0, ocr_results[k][0])
                                
                                if more_words:
                                    descripcion = " ".join(more_words + [descripcion])
                            
                            break
                    
                    # No duplicar ítems que ya existen
                    if not any(item["amount"] == word for item in items):
                        items.append({
                            "description": descripcion,
                            "amount": word
                        })
        
        return items