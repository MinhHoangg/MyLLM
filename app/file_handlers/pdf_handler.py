"""
PDF Handler: Extract text and images from PDF files using PyMuPDF (fitz).
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
import io
from PIL import Image


class PDFHandler:
    """Handler for processing PDF files - extracts both text and images."""
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text and images from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing text content and images with metadata
        """
        pdf_path = Path(file_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        results = {
            'text_content': [],
            'images': [],
            'metadata': {
                'file_name': pdf_path.name,
                'file_path': str(pdf_path),
                'file_type': 'pdf'
            }
        }
        
        try:
            # Open PDF document
            doc = fitz.open(file_path)
            
            # Update metadata
            results['metadata']['num_pages'] = len(doc)
            results['metadata']['pdf_metadata'] = doc.metadata
            
            # Extract content from each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                if text.strip():
                    results['text_content'].append({
                        'page': page_num + 1,
                        'text': text.strip()
                    })
                
                # Extract images
                images = self._extract_images_from_page(page, page_num + 1, pdf_path.stem)
                results['images'].extend(images)
            
            doc.close()
            
        except Exception as e:
            raise Exception(f"Error processing PDF {file_path}: {str(e)}")
        
        return results
    
    def _extract_images_from_page(self, page: fitz.Page, page_num: int, pdf_name: str) -> List[Dict[str, Any]]:
        """
        Extract all images from a PDF page.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            pdf_name: Name of the PDF file (without extension)
            
        Returns:
            List of dictionaries containing image data and metadata
        """
        images = []
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = page.parent.extract_image(xref)
                
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes))
                
                # Create unique identifier for the image
                image_id = f"{pdf_name}_page{page_num}_img{img_index + 1}"
                
                images.append({
                    'image_id': image_id,
                    'image': pil_image,
                    'format': image_ext,
                    'page': page_num,
                    'width': pil_image.width,
                    'height': pil_image.height,
                    'source': 'pdf'
                })
                
            except Exception as e:
                print(f"Warning: Could not extract image {img_index} from page {page_num}: {str(e)}")
                continue
        
        return images
    
    def is_supported(self, file_path: str) -> bool:
        """Check if the file extension is supported."""
        return Path(file_path).suffix.lower() in self.supported_extensions
