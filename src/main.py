import os
import argparse
import logging
from processor import DocumentProcessor
from extractor import FacturaExtractor
from excel_writer import ExcelWriter
import time
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_invoice(file_path, output_dir=None, save_ocr=False):
    """
    Procesa una factura y extrae su información
    
    Args:
        file_path: Ruta al archivo de factura (PDF o imagen)
        output_dir: Directorio para guardar la salida (opcional)
        save_ocr: Si es True, guarda los resultados del OCR en un archivo .txt
        
    Returns:
        dict: Datos extraídos de la factura
    """
    # Inicializar componentes
    start_time = time.time()
    logger.info(f"Iniciando procesamiento de {file_path}")
    
    processor = DocumentProcessor()
    extractor = FacturaExtractor()
    
    # Procesar documento
    encoding, ocr_results, image = processor.process_document(file_path, save_ocr)
    
    # Extraer información
    invoice_data = extractor.extract_info(encoding, ocr_results)
    
    # Añadir metadatos sobre el procesamiento
    processing_time = time.time() - start_time
    invoice_data['metadata'] = {
        'processing_time': f"{processing_time:.2f} segundos",
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'file': os.path.basename(file_path)
    }
    
    # Calcular y mostrar estadísticas del procesamiento
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Procesamiento completo en {total_time:.2f} segundos")
    
    # Mostrar resumen de los resultados
    logger.info("Resumen de extracción:")
    for campo, valor in invoice_data.items():
        if campo != 'items' and campo != 'metadata':
            logger.info(f"  - {campo}: {valor}")
    logger.info(f"  - Líneas de detalle: {len(invoice_data.get('items', []))} items")
    
    return invoice_data

def generate_single_invoice_excel(invoice_data, file_path, output_dir=None):
    """
    Genera un archivo Excel para una sola factura
    
    Args:
        invoice_data: Diccionario con datos extraídos de la factura
        file_path: Ruta al archivo de factura original
        output_dir: Directorio para guardar la salida (opcional)
        
    Returns:
        str: Ruta al archivo Excel generado
    """
    writer = ExcelWriter()
    
    # Generar nombre de archivo Excel
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    if output_dir:
        excel_path = os.path.join(output_dir, f"{base_name}_procesado.xlsx")
    else:
        excel_path = f"{base_name}_procesado.xlsx"
    
    # Escribir a Excel
    output_file = writer.write_invoice_to_excel(invoice_data, excel_path)
    logger.info(f"Excel generado: {output_file}")
    
    return output_file

def process_multiple_invoices(directory_path, output_dir=None, save_ocr=False):
    """
    Procesa múltiples facturas y genera un archivo Excel consolidado
    
    Args:
        directory_path: Directorio con los archivos de factura
        output_dir: Directorio para guardar la salida (opcional)
        save_ocr: Si es True, guarda los resultados del OCR en un archivo .txt
        
    Returns:
        str: Ruta al archivo Excel generado
    """
    # Buscar archivos de factura en el directorio (PDF e imágenes)
    pdf_files = []
    pdf_files.extend(glob.glob(os.path.join(directory_path, "*.pdf")))
    pdf_files.extend(glob.glob(os.path.join(directory_path, "*.PDF")))
    
    image_files = []
    for ext in ["jpg", "jpeg", "png", "tiff", "JPG", "JPEG", "PNG", "TIFF"]:
        image_files.extend(glob.glob(os.path.join(directory_path, f"*.{ext}")))
    
    invoice_files = pdf_files + image_files
    
    if not invoice_files:
        logger.error(f"No se encontraron facturas en el directorio {directory_path}")
        return None
    
    logger.info(f"Se encontraron {len(invoice_files)} facturas para procesar")
    
    # Procesar cada factura
    all_invoice_data = []
    for file_path in invoice_files:
        try:
            logger.info(f"Procesando {os.path.basename(file_path)}")
            invoice_data = process_invoice(file_path, output_dir, save_ocr)
            all_invoice_data.append(invoice_data)
        except Exception as e:
            logger.error(f"Error procesando {os.path.basename(file_path)}: {e}", exc_info=True)
    
    # Generar Excel consolidado
    if all_invoice_data:
        writer = ExcelWriter()
        
        # Generar nombre para el Excel consolidado
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder_name = os.path.basename(os.path.normpath(directory_path))
        
        if output_dir:
            excel_path = os.path.join(output_dir, f"facturas_{folder_name}_{timestamp}.xlsx")
        else:
            excel_path = f"facturas_{folder_name}_{timestamp}.xlsx"
        
        # Escribir todas las facturas en un único Excel
        output_file = writer.write_multiple_invoices(all_invoice_data, excel_path)
        logger.info(f"Excel consolidado generado: {output_file}")
        
        return output_file
    else:
        logger.warning("No se pudo generar Excel consolidado: no hay datos de facturas")
        return None

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Extrae información de facturas y genera Excel')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', '-f', help='Ruta al archivo de factura individual (PDF o imagen)')
    group.add_argument('--directory', '-d', help='Directorio con múltiples facturas para procesar')
    
    parser.add_argument('--output', '-o', help='Directorio de salida para el Excel')
    parser.add_argument('--debug', action='store_true', help='Habilita modo debug con más información')
    parser.add_argument('--save-ocr', action='store_true', help='Guarda los resultados del OCR en archivos de texto')
    
    args = parser.parse_args()
    
    # Crear directorio de salida si no existe
    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Configurar nivel de log según modo debug
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Modo debug habilitado")
    
    try:
        if args.file:
            # Procesar una sola factura
            invoice_data = process_invoice(args.file, args.output, args.save_ocr)
            excel_file = generate_single_invoice_excel(invoice_data, args.file, args.output)
            print(f"✅ Factura procesada exitosamente. Excel guardado en: {excel_file}")
        elif args.directory:
            # Procesar múltiples facturas
            excel_file = process_multiple_invoices(args.directory, args.output, args.save_ocr)
            if excel_file:
                print(f"✅ Facturas procesadas exitosamente. Excel consolidado guardado en: {excel_file}")
            else:
                print("❌ No se pudo generar el Excel consolidado")
    except Exception as e:
        logger.error(f"❌ Error en el procesamiento: {e}", exc_info=True)
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()