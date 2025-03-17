# Extractor de Información de Facturas

Este proyecto implementa un sistema para extraer información relevante de facturas en formato PDF o imagen utilizando técnicas de OCR y procesamiento de documentos basado en LayoutLMv2.

## Descripción

El sistema analiza facturas y extrae datos como:
- Número de factura
- Fecha
- Información del cliente y proveedor
- Conceptos facturados
- Importes y totales

Todos los datos extraídos se estructuran en un archivo Excel organizado.

## Requisitos del Sistema

> **⚠️ Importante:** Este sistema actualmente **solo es compatible con Linux** debido a la dependencia de detectron2, que no está disponible oficialmente para Windows o macOS. Aunque es posible instalar detectron2 en Windows, esta resulta más complicada que en Linux. 

### Software necesario

- Sistema operativo Linux (Ubuntu/Debian recomendado)
- Python 3.8 o superior
- Tesseract OCR 
- Poppler (para convertir PDF a imágenes)
- CUDA compatible con Detectron2 (recomendado para mejor rendimiento)

### Instalación de componentes del sistema

- Tesseract OCR: `sudo apt-get install tesseract-ocr`
- Poppler: `sudo apt-get install poppler-utils`

### Instalación de dependencias Python

1. Instalar dependencias principales:

```bash
pip install -r requirements.txt
```

2. Instalar Detectron2 (no disponible oficialmente en PyPI):

```python
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Estructura del Proyecto

```
Invoice-Indexer/
├── src/
│   ├── main.py          # Punto de entrada del programa
│   ├── processor.py     # Procesamiento de documentos y OCR
│   ├── extractor.py     # Extracción de datos específicos de facturas
│   └── excel_writer.py  # Generación de archivos Excel con resultados
├── requirements.txt     # Dependencias del proyecto
└── README.md            
```

### Uso

El programa se puede ejecutar en dos modos:

1. **Procesar una factura individual**

```python
python src/main.py --file "<ruta-factura>"
```

2. **Procesar múltiples facturas en un directorio**

```python
python src/main.py --directory "<ruta-carpeta-facturas>" 
```

### Opciones adicionales

- `--debug`: Muestra información detallada durante el procesamiento

- `--save-ocr`: Guarda los resultados del OCR en archivos de texto para análisis

- `--output <ruta-salida>`: Indica la ruta donde se guardarán los resultados

## Limitaciones

- El sistema está inicialmente diseñado para facturas con un único formato.
- Actualmente procesa solo la primera página de documentos PDF multipágina.

# Resolución de problemas comunes

- Error en la instalación de Detectron2: Asegúrate de tener un compilador de C++ instalado y las dependencias de CUDA correctas.
- Error de Tesseract no encontrado: Asegúrate de que Tesseract OCR está correctamente instalado y en el PATH del sistema.
- Error al procesar PDFs: Verifica que Poppler está instalado correctamente.