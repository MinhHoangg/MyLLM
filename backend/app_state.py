"""
Global application state
"""
from typing import Optional

class AppState:
    """Global application state"""
    vector_db: Optional[any] = None
    chatbot: Optional[any] = None
    model: Optional[any] = None
    ingestion: Optional[any] = None
    high_parameter: bool = False

# Global state instance
state = AppState()
