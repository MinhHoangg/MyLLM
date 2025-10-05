"""
File handlers package for processing different document types.
"""

from .pdf_handler import PDFHandler
from .image_handler import ImageHandler
from .doc_handler import DocHandler
from .xlsx_handler import XLSXHandler
from .txt_handler import TXTHandler

__all__ = [
    'PDFHandler',
    'ImageHandler',
    'DocHandler',
    'XLSXHandler',
    'TXTHandler'
]
