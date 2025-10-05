"""
Document Ingestion Module: Orchestrates file processing, chunking, and preparation for embedding.
Uses LangChain for document chunking.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from PIL import Image
import io
import base64

from app.file_handlers import PDFHandler, ImageHandler, DocHandler, XLSXHandler, TXTHandler


class DocumentIngestion:
    """Handles document ingestion, processing, and chunking."""
    
    def __init__(
        self,
        upload_dir: str = "data/uploads",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize DocumentIngestion.
        
        Args:
            upload_dir: Directory to store uploaded files
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between consecutive chunks
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize handlers
        self.handlers = {
            'pdf': PDFHandler(),
            'image': ImageHandler(use_ocr=True),
            'doc': DocHandler(),
            'xlsx': XLSXHandler(),
            'txt': TXTHandler()
        }
        
        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def process_file(self, file_path: str, save_copy: bool = True) -> Dict[str, Any]:
        """
        Process a single file through the appropriate handler.
        
        Args:
            file_path: Path to the file to process
            save_copy: Whether to save a copy to the upload directory
            
        Returns:
            Dictionary containing processed content and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and select handler
        handler = self._get_handler(str(file_path))
        if handler is None:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Extract content using appropriate handler
        extracted_data = handler.extract(str(file_path))
        
        # Save copy if requested
        if save_copy:
            dest_path = self.upload_dir / file_path.name
            if not dest_path.exists():
                shutil.copy2(file_path, dest_path)
            extracted_data['metadata']['saved_path'] = str(dest_path)
        
        return extracted_data
    
    def chunk_documents(self, extracted_data: Dict[str, Any]) -> List[Document]:
        """
        Chunk text content using LangChain's text splitter.
        
        Args:
            extracted_data: Data extracted from file handlers
            
        Returns:
            List of LangChain Document objects with chunked text
        """
        documents = []
        
        # Process text content
        text_contents = extracted_data.get('text_content', [])
        metadata_base = extracted_data.get('metadata', {})
        
        for idx, content_item in enumerate(text_contents):
            text = content_item.get('text', '')
            if not text.strip():
                continue
            
            # Create metadata for this text section
            metadata = metadata_base.copy()
            metadata.update({
                'content_type': 'text',
                'section_index': idx
            })
            
            # Add any additional metadata from content_item
            for key in ['page', 'sheet', 'source', 'image_id']:
                if key in content_item:
                    metadata[key] = content_item[key]
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create Document objects for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = chunk_idx
                chunk_metadata['total_chunks'] = len(chunks)
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
        
        return documents
    
    def process_images(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process extracted images for embedding.
        
        Args:
            extracted_data: Data extracted from file handlers
            
        Returns:
            List of dictionaries containing image data and metadata
        """
        processed_images = []
        images = extracted_data.get('images', [])
        metadata_base = extracted_data.get('metadata', {})
        
        for idx, image_item in enumerate(images):
            image = image_item.get('image')
            if image is None:
                continue
            
            # Create metadata for this image
            metadata = metadata_base.copy()
            metadata.update({
                'content_type': 'image',
                'image_index': idx,
                'image_id': image_item.get('image_id', f"img_{idx}"),
                'format': image_item.get('format', 'unknown'),
                'dimensions': f"{image_item.get('width', 0)}x{image_item.get('height', 0)}"
            })
            
            # Add any additional metadata
            for key in ['page', 'source']:
                if key in image_item:
                    metadata[key] = image_item[key]
            
            # Convert PIL Image to bytes for storage
            img_byte_arr = io.BytesIO()
            if isinstance(image, Image.Image):
                image.save(img_byte_arr, format=image_item.get('format', 'PNG'))
                img_bytes = img_byte_arr.getvalue()
            else:
                img_bytes = image
            
            processed_images.append({
                'image_id': metadata['image_id'],
                'image': image,  # Keep PIL Image for display
                'image_bytes': img_bytes,
                'metadata': metadata
            })
        
        return processed_images
    
    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Complete ingestion pipeline for a file.
        
        Args:
            file_path: Path to the file to ingest
            
        Returns:
            Dictionary containing chunked documents and processed images
        """
        # Extract content from file
        extracted_data = self.process_file(file_path)
        
        # Chunk text documents
        text_documents = self.chunk_documents(extracted_data)
        
        # Process images
        image_documents = self.process_images(extracted_data)
        
        return {
            'text_documents': text_documents,
            'image_documents': image_documents,
            'metadata': extracted_data['metadata'],
            'num_text_chunks': len(text_documents),
            'num_images': len(image_documents)
        }
    
    def ingest_multiple_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Ingest multiple files.
        
        Args:
            file_paths: List of file paths to ingest
            
        Returns:
            List of ingestion results for each file
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.ingest_file(file_path)
                result['status'] = 'success'
                results.append(result)
            except Exception as e:
                results.append({
                    'file_path': file_path,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def _get_handler(self, file_path: str):
        """Get the appropriate handler for a file."""
        for handler in self.handlers.values():
            if handler.is_supported(file_path):
                return handler
        return None
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions."""
        extensions = []
        for handler in self.handlers.values():
            extensions.extend(handler.supported_extensions)
        return list(set(extensions))
