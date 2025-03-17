import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import LayoutLMv2Processor
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Procesa documentos PDF/imágenes para preparar entrada al modelo LayoutLM"""
    
    def __init__(self, processor_name="microsoft/layoutlmv2-base-uncased"):
        """
        Inicializa el procesador de documentos
        
        Args:
            processor_name: Nombre del procesador LayoutLM a utilizar
        """
        logger.info(f"Inicializando procesador con {processor_name}")
        # Inicializar el procesador con apply_ocr=False para evitar conflictos
        self.processor = LayoutLMv2Processor.from_pretrained(processor_name, apply_ocr=False)
        
    def process_document(self, file_path, save_ocr=False):
        """
        Procesa un documento (PDF o imagen) y lo prepara para LayoutLM
        
        Args:
            file_path: Ruta al archivo a procesar
            save_ocr: Si es True, guarda los resultados del OCR en un archivo .txt
            
        Returns:
            dict: Entradas procesadas para el modelo
            list: Lista de palabras detectadas con sus posiciones
            PIL.Image: Imagen procesada
        """
        logger.info(f"Procesando documento: {file_path}")
        
        # Convertir PDF a imagen si es necesario
        if file_path.lower().endswith('.pdf'):
            logger.info("Convirtiendo PDF a imagen")
            images = convert_from_path(file_path)
            image = images[0]  # Procesamos solo la primera página por simplicidad
        else:
            image = Image.open(file_path)
        
        # Aplicar OCR para obtener palabras y sus posiciones
        ocr_results = self._perform_ocr(image)
        
        # Guardar resultados del OCR en archivo de texto solo si se solicita
        if save_ocr:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            ocr_output_path = f"{base_name}_ocr_output.txt"
            self._save_ocr_results(ocr_results, ocr_output_path)
            logger.info(f"Resultados del OCR guardados en {ocr_output_path}")
        
        # Procesar para LayoutLM
        encoding = self._prepare_for_layoutlm(image, ocr_results)
        
        return encoding, ocr_results, image
        
    def _perform_ocr(self, image):
        """
        Realiza OCR en la imagen para detectar palabras y sus posiciones
        
        Args:
            image: Imagen PIL a procesar
            
        Returns:
            list: Resultados de OCR con palabra y posición
        """
        logger.info("Aplicando OCR a la imagen")
        # Obtener resultados completos del OCR
        ocr_df = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
        
        # Filtrar solo filas con texto
        ocr_df = ocr_df[ocr_df.conf > 0]
        
        # Crear lista de resultados OCR con formato esperado por LayoutLM
        ocr_results = []
        for _, row in ocr_df.iterrows():
            if str(row['text']).strip():
                # Formato: (palabra, box)
                box = [row['left'], row['top'], row['left'] + row['width'], row['top'] + row['height']]
                ocr_results.append((row['text'], box))
        
        logger.info(f"OCR completado: {len(ocr_results)} palabras detectadas")
        return ocr_results
    
    def _save_ocr_results(self, ocr_results, output_path):
        """
        Guarda los resultados del OCR en un archivo de texto para análisis
        
        Args:
            ocr_results: Resultados del OCR
            output_path: Ruta donde guardar el archivo
        """
        logger.info(f"Guardando resultados del OCR en {output_path}")
        
        # Crear texto plano con todas las palabras
        full_text = " ".join([word for word, _ in ocr_results])
        
        # Crear diccionario con información detallada
        ocr_data = {
            "full_text": full_text,
            "words": [{"text": word, "box": box} for word, box in ocr_results]
        }
        
        # Guardar en formato JSON para análisis posterior
        with open(output_path, "w") as f:
            f.write("=== TEXTO COMPLETO ===\n")
            f.write(full_text)
            f.write("\n\n=== PALABRAS Y POSICIONES ===\n")
            json.dump(ocr_data["words"], f, indent=2)
        
        logger.info(f"Resultados del OCR guardados en {output_path}")
        
    def _prepare_for_layoutlm(self, image, ocr_results):
        """
        Prepara los datos para el modelo LayoutLM
        
        Args:
            image: Imagen PIL
            ocr_results: Resultados del OCR
            
        Returns:
            dict: Entradas procesadas para el modelo
        """
        # Extraer solo las palabras y cajas
        words = [result[0] for result in ocr_results]
        boxes = [result[1] for result in ocr_results]
        
        # Normalizar las cajas al formato que espera LayoutLMv2 
        # (convertir a coordenadas normalizadas entre 0 y 1000)
        normalized_boxes = []
        width, height = image.size
        for box in boxes:
            normalized_box = [
                int(1000 * box[0] / width),
                int(1000 * box[1] / height),
                int(1000 * box[2] / width),
                int(1000 * box[3] / height)
            ]
            normalized_boxes.append(normalized_box)
        
        # Usar el procesador de LayoutLM para crear entradas del modelo
        encoding = self.processor(
            image,
            words,
            boxes=normalized_boxes,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        
        return encoding