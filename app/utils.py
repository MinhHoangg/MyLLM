"""
Utility functions for the chatbot application.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime


def ensure_directory_exists(directory_path: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Path object
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Formatted file size string
    """
    size_bytes = os.path.getsize(file_path)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} TB"


def format_timestamp(timestamp: datetime = None) -> str:
    """
    Format a timestamp for display.
    
    Args:
        timestamp: Datetime object (uses current time if None)
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def save_json(data: Dict[str, Any], file_path: str) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save the file
        
    Returns:
        True if successful
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON: {str(e)}")
        return False


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary with loaded data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {str(e)}")
        return {}


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_sources(sources: List[Dict[str, Any]]) -> str:
    """
    Format source metadata for display.
    
    Args:
        sources: List of source metadata dictionaries
        
    Returns:
        Formatted string
    """
    if not sources:
        return "No sources available"
    
    formatted = []
    seen_files = set()
    
    for source in sources:
        file_name = source.get('file_name', 'Unknown')
        if file_name not in seen_files:
            seen_files.add(file_name)
            page = source.get('page', '')
            if page:
                formatted.append(f"📄 {file_name} (Page {page})")
            else:
                formatted.append(f"📄 {file_name}")
    
    return "\n".join(formatted[:5])  # Limit to 5 sources


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    return filename


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count (rough estimate: 1 token ≈ 4 characters).
    
    Args:
        text: Text to count
        
    Returns:
        Approximate token count
    """
    return len(text) // 4


def format_model_info(model_info: Dict[str, Any]) -> str:
    """
    Format model information for display.
    
    Args:
        model_info: Model information dictionary
        
    Returns:
        Formatted string
    """
    lines = [
        f"Model: {model_info.get('model_name', 'Unknown')}",
        f"Device: {model_info.get('device', 'Unknown')}",
        f"Max Length: {model_info.get('max_length', 'Unknown')} tokens",
        f"Temperature: {model_info.get('temperature', 'Unknown')}",
        f"Top-p: {model_info.get('top_p', 'Unknown')}"
    ]
    
    return "\n".join(lines)


class Timer:
    """Simple timer for performance measurement."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = datetime.now()
    
    def stop(self):
        """Stop the timer."""
        self.end_time = datetime.now()
    
    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else datetime.now()
        delta = end - self.start_time
        return delta.total_seconds()
    
    def elapsed_str(self) -> str:
        """
        Get formatted elapsed time string.
        
        Returns:
            Formatted elapsed time
        """
        seconds = self.elapsed()
        
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            seconds = seconds % 60
            return f"{minutes}m {seconds:.0f}s"


def validate_file_type(file_path: str, supported_extensions: List[str]) -> bool:
    """
    Validate if a file type is supported.
    
    Args:
        file_path: Path to the file
        supported_extensions: List of supported extensions
        
    Returns:
        True if supported
    """
    ext = Path(file_path).suffix.lower()
    return ext in [e.lower() for e in supported_extensions]
