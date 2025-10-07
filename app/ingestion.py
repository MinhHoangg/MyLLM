"""
Document Ingestion Module: Orchestrates file processing, chunking, and preparation for embedding.
Uses LangChain for document chunking.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil
import re
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
        adaptive_chunking: bool = True,
        base_chunk_size: int = 512,  # Base size, will adapt
        model_context_window: int = 4096  # Model's context window
    ):
        """
        Initialize DocumentIngestion with adaptive chunking.
        
        Args:
            upload_dir: Directory to store uploaded files
            adaptive_chunking: Enable intelligent adaptive chunking
            base_chunk_size: Base chunk size (adapts based on content)
            model_context_window: Model's maximum context size
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Adaptive chunking parameters
        self.adaptive_chunking = adaptive_chunking
        self.base_chunk_size = base_chunk_size
        self.model_context_window = model_context_window
        
        # Initialize handlers
        self.handlers = {
            'pdf': PDFHandler(),
            'image': ImageHandler(use_ocr=True),
            'doc': DocHandler(),
            'xlsx': XLSXHandler(),
            'txt': TXTHandler()
        }
        
        # Will be initialized dynamically based on content
        self.text_splitter = None
    
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
        
        # Preserve original file name in metadata
        extracted_data['metadata']['original_file_name'] = file_path.name
        extracted_data['metadata']['original_file_path'] = str(file_path)
        
        # Save copy if requested - preserve original name
        if save_copy:
            dest_path = self.upload_dir / file_path.name
            if not dest_path.exists():
                shutil.copy2(file_path, dest_path)
            extracted_data['metadata']['saved_path'] = str(dest_path)
            # Ensure file_name reflects the original name
            extracted_data['metadata']['file_name'] = file_path.name
        
        return extracted_data
    
    def chunk_documents(self, extracted_data: Dict[str, Any]) -> List[Document]:
        """
        Intelligently chunk text content using adaptive strategies.
        
        Args:
            extracted_data: Data extracted from file handlers
            
        Returns:
            List of LangChain Document objects with optimally chunked text
        """
        documents = []
        
        # Process text content with adaptive chunking
        text_contents = extracted_data.get('text_content', [])
        
        if not text_contents:
            return documents
        
        # Analyze content to determine optimal chunking strategy
        content_analysis = self._analyze_content(text_contents)
        
        # Create adaptive text splitter based on content
        splitter = self._create_adaptive_splitter(content_analysis)
        metadata_base = extracted_data.get('metadata', {})
        
        # Log chunking strategy
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Content analysis: {content_analysis['type']}, {content_analysis['total_chars']} chars")
        
        for idx, content_item in enumerate(text_contents):
            text = content_item.get('text', '')
            if not text.strip():
                continue
            
            # Create metadata for this text section
            metadata = metadata_base.copy()
            metadata.update({
                'content_type': 'text',
                'section_index': idx,
                'chunking_strategy': content_analysis['type']
            })
            
            # Add any additional metadata from content_item
            for key in ['page', 'sheet', 'source', 'image_id']:
                if key in content_item:
                    metadata[key] = content_item[key]
            
            # Split text into chunks using adaptive splitter
            chunks = splitter.split_text(text)
            logger.info(f"Section {idx+1}: {len(text)} chars → {len(chunks)} chunks")
            
            # Create Document objects for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk),
                    'word_count': len(chunk.split())
                })
                
                # Create Document object
                doc = Document(
                    page_content=chunk.strip(),
                    metadata=chunk_metadata
                )
                documents.append(doc)
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
        
        return documents
    
    def _analyze_content(self, text_contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze content characteristics to determine optimal chunking strategy.
        
        Returns:
            Dictionary with content analysis results
        """
        total_chars = 0
        total_lines = 0
        has_structure = False
        avg_sentence_length = 0
        content_type = 'generic'
        
        all_text = ""
        for content in text_contents:
            text = content.get('text', '')
            all_text += text + "\n"
            total_chars += len(text)
            total_lines += text.count('\n')
        
        if total_chars == 0:
            return {'type': 'empty', 'chunk_size': self.base_chunk_size, 'overlap': 50}
        
        # Detect content structure
        has_headers = bool(re.search(r'^#{1,6}\s', all_text, re.MULTILINE))  # Markdown headers
        has_bullets = bool(re.search(r'^[\s]*[-*+]\s', all_text, re.MULTILINE))  # Bullet points
        has_numbers = bool(re.search(r'^[\s]*\d+\.\s', all_text, re.MULTILINE))  # Numbered lists
        has_paragraphs = all_text.count('\n\n') > 2
        
        has_structure = has_headers or has_bullets or has_numbers or has_paragraphs
        
        # Calculate average sentence length
        sentences = re.split(r'[.!?]+', all_text)
        if sentences:
            avg_sentence_length = sum(len(s.strip()) for s in sentences if s.strip()) / len([s for s in sentences if s.strip()])
        
        # Determine content type
        if has_headers and has_bullets:
            content_type = 'structured_document'
        elif total_lines / max(1, total_chars / 80) > 0.5:  # Many short lines
            content_type = 'list_data'
        elif avg_sentence_length > 100:
            content_type = 'dense_text'
        elif avg_sentence_length < 30:
            content_type = 'sparse_text'
        else:
            content_type = 'narrative_text'
        
        return {
            'type': content_type,
            'total_chars': total_chars,
            'total_lines': total_lines,
            'has_structure': has_structure,
            'avg_sentence_length': avg_sentence_length,
            'paragraph_count': all_text.count('\n\n')
        }
    
    def _create_adaptive_splitter(self, analysis: Dict[str, Any]) -> RecursiveCharacterTextSplitter:
        """
        Create an adaptive text splitter based on content analysis.
        
        Args:
            analysis: Content analysis results
            
        Returns:
            Configured RecursiveCharacterTextSplitter
        """
        content_type = analysis['type']
        total_chars = analysis['total_chars']
        has_structure = analysis['has_structure']
        avg_sentence_length = analysis['avg_sentence_length']
        
        # Adaptive chunk size calculation
        if content_type == 'structured_document':
            # Preserve structure, larger chunks
            chunk_size = min(800, max(400, int(avg_sentence_length * 6)))
            overlap = max(50, int(chunk_size * 0.15))
            separators = ["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
            
        elif content_type == 'list_data':
            # Smaller chunks, preserve list items
            chunk_size = min(400, max(200, int(total_chars / 50)))
            overlap = max(20, int(chunk_size * 0.1))
            separators = ["\n\n", "\n", "; ", ", ", " ", ""]
            
        elif content_type == 'dense_text':
            # Larger chunks for dense content
            chunk_size = min(1200, max(600, int(avg_sentence_length * 8)))
            overlap = max(100, int(chunk_size * 0.2))
            separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
            
        elif content_type == 'sparse_text':
            # Smaller chunks for sparse content
            chunk_size = min(300, max(150, int(avg_sentence_length * 4)))
            overlap = max(30, int(chunk_size * 0.15))
            separators = ["\n\n", "\n", ". ", " ", ""]
            
        else:  # narrative_text or generic
            # Balanced approach
            chunk_size = self.base_chunk_size
            overlap = max(50, int(chunk_size * 0.15))
            separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        
        # Ensure chunk size fits within model context window
        max_chunk_size = min(self.model_context_window // 4, 1500)  # Leave room for prompt
        chunk_size = min(chunk_size, max_chunk_size)
        
        # Ensure minimum viable chunk size
        chunk_size = max(chunk_size, 100)
        overlap = min(overlap, chunk_size // 3)  # Overlap shouldn't exceed 1/3 of chunk
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Adaptive chunking: type={content_type}, size={chunk_size}, overlap={overlap}")
        
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=separators,
            keep_separator=True  # Preserve separators for context
        )
    
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
