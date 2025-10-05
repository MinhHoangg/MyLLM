"""
TXT/LOG Handler: Extract text from plain text and log files.
"""

from pathlib import Path
from typing import Dict, Any
import chardet


class TXTHandler:
    """Handler for processing TXT and LOG files."""
    
    def __init__(self):
        self.supported_extensions = ['.txt', '.log', '.md', '.csv', '.json', '.xml', '.yaml', '.yml']
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dictionary containing text content and metadata
        """
        txt_path = Path(file_path)
        if not txt_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        results = {
            'text_content': [],
            'images': [],
            'metadata': {
                'file_name': txt_path.name,
                'file_path': str(txt_path),
                'file_type': 'text'
            }
        }
        
        try:
            # Detect encoding
            encoding = self._detect_encoding(file_path)
            results['metadata']['encoding'] = encoding
            
            # Read file content
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
            
            if content.strip():
                results['text_content'].append({
                    'text': content.strip()
                })
            
            # Add file size
            file_size = txt_path.stat().st_size
            results['metadata']['file_size_bytes'] = file_size
            results['metadata']['num_lines'] = len(content.splitlines())
            
        except Exception as e:
            raise Exception(f"Error processing text file {file_path}: {str(e)}")
        
        return results
    
    def _detect_encoding(self, file_path: str) -> str:
        """
        Detect the encoding of a text file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected encoding (defaults to 'utf-8')
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                
                # Fallback to utf-8 if detection fails
                if encoding is None:
                    return 'utf-8'
                
                return encoding
        except Exception:
            return 'utf-8'
    
    def is_supported(self, file_path: str) -> bool:
        """Check if the file extension is supported."""
        return Path(file_path).suffix.lower() in self.supported_extensions
