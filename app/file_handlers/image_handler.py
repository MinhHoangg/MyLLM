"""
Image Handler: Process image files (PNG, JPG, JPEG) with OCR capabilities.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import pytesseract


class ImageHandler:
    """Handler for processing image files with OCR support."""
    
    def __init__(self, use_ocr: bool = True):
        """
        Initialize ImageHandler.
        
        Args:
            use_ocr: Whether to use OCR to extract text from images
        """
        self.supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
        self.use_ocr = use_ocr
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract image and optionally perform OCR.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary containing image data and extracted text (if OCR is enabled)
        """
        img_path = Path(file_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        results = {
            'images': [],
            'text_content': [],
            'metadata': {
                'file_name': img_path.name,
                'file_path': str(img_path),
                'file_type': 'image'
            }
        }
        
        try:
            # Open image
            img = Image.open(file_path)
            
            # Convert RGBA to RGB if necessary
            if img.mode == 'RGBA':
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])
                img = rgb_img
            
            # Store image information
            image_data = {
                'image_id': img_path.stem,
                'image': img,
                'format': img.format or img_path.suffix[1:],
                'width': img.width,
                'height': img.height,
                'mode': img.mode,
                'source': 'file'
            }
            results['images'].append(image_data)
            results['metadata']['dimensions'] = f"{img.width}x{img.height}"
            
            # Perform OCR if enabled
            if self.use_ocr:
                ocr_text = self._perform_ocr(img)
                if ocr_text.strip():
                    results['text_content'].append({
                        'source': 'ocr',
                        'image_id': img_path.stem,
                        'text': ocr_text.strip()
                    })
            
        except Exception as e:
            raise Exception(f"Error processing image {file_path}: {str(e)}")
        
        return results
    
    def _perform_ocr(self, image: Image.Image) -> str:
        """
        Perform OCR on an image using pytesseract.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text from the image
        """
        try:
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"Warning: OCR failed: {str(e)}")
            return ""
    
    def is_supported(self, file_path: str) -> bool:
        """Check if the file extension is supported."""
        return Path(file_path).suffix.lower() in self.supported_extensions
