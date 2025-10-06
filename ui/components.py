"""
Streamlit UI Components: Reusable widgets for the chatbot interface.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd


def render_chat_message(message: Dict[str, str], avatar: Optional[str] = None):
    """
    Render a single chat message.
    
    Args:
        message: Dictionary with 'role' and 'content'
        avatar: Optional avatar for the message
    """
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def render_chat_history(messages: List[Dict[str, str]]):
    """
    Render the entire chat history.
    
    Args:
        messages: List of message dictionaries
    """
    for message in messages:
        render_chat_message(message)


def file_uploader_component(
    supported_extensions: List[str],
    accept_multiple: bool = True,
    key: str = "file_uploader"
) -> Optional[List]:
    """
    Render a file uploader component.
    
    Args:
        supported_extensions: List of supported file extensions
        accept_multiple: Whether to accept multiple files
        key: Unique key for the component
        
    Returns:
        Uploaded files or None
    """
    # Format extensions for display
    ext_str = ", ".join(supported_extensions)
    
    uploaded_files = st.file_uploader(
        f"Upload documents ({ext_str})",
        type=[ext.replace('.', '') for ext in supported_extensions],
        accept_multiple_files=accept_multiple,
        key=key,
        help=f"Supported formats: {ext_str}"
    )
    
    return uploaded_files


def sidebar_settings_component():
    """
    Render sidebar settings component.
    
    Returns:
        Dictionary of settings
    """
    st.sidebar.header("⚙️ Settings")
    
    settings = {}
    
    # RAG settings
    st.sidebar.subheader("RAG Settings")
    
    rag_mode = st.sidebar.radio(
        "RAG Mode",
        options=["Auto-detect", "Always On", "Always Off"],
        index=0,
        help="Auto-detect: Smart detection (greetings=no RAG, questions=RAG)\nAlways On: Force RAG for all messages\nAlways Off: Never use RAG"
    )
    
    # Convert radio selection to use_rag setting
    if rag_mode == "Auto-detect":
        settings['use_rag'] = None  # None triggers auto-detection
    elif rag_mode == "Always On":
        settings['use_rag'] = True
    else:  # "Always Off"
        settings['use_rag'] = False
    
    settings['n_results'] = st.sidebar.slider(
        "Number of Retrieved Documents",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of relevant documents to retrieve"
    )
    
    # Generation settings
    st.sidebar.subheader("Generation Settings")
    settings['temperature'] = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Controls randomness in generation"
    )
    
    settings['max_tokens'] = st.sidebar.slider(
        "Max Tokens",
        min_value=128,
        max_value=2048,
        value=512,
        step=128,
        help="Maximum length of generated response"
    )
    
    # History settings
    st.sidebar.subheader("Conversation")
    settings['include_history'] = st.sidebar.checkbox(
        "Include Chat History",
        value=True,
        help="Include previous messages in context"
    )
    
    return settings


def database_stats_component(stats: Dict[str, Any]):
    """
    Render database statistics component.
    
    Args:
        stats: Dictionary with database statistics
    """
    st.subheader("📊 Vector Database Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Collection Name", stats.get('name', 'N/A'))
    
    with col2:
        st.metric("Total Documents", stats.get('count', 0))
    
    with col3:
        st.metric("Status", "Active" if stats.get('count', 0) > 0 else "Empty")
    
    with st.expander("📁 Database Details"):
        st.text(f"Persist Directory: {stats.get('persist_directory', 'N/A')}")


def document_list_component(documents: Dict[str, Any], allow_delete: bool = True):
    """
    Render a list of documents in the database.
    
    Args:
        documents: Dictionary containing document data
        allow_delete: Whether to show delete buttons
    """
    st.subheader("📄 Stored Documents")
    
    if not documents.get('documents'):
        st.info("No documents in the database yet. Upload some files to get started!")
        return
    
    # Create DataFrame for display
    data = []
    for idx, (doc_id, doc_text, metadata) in enumerate(zip(
        documents.get('ids', []),
        documents.get('documents', []),
        documents.get('metadatas', [])
    )):
        data.append({
            'ID': doc_id[:8] + '...',
            'File': metadata.get('file_name', 'Unknown'),
            'Type': metadata.get('content_type', 'Unknown'),
            'Preview': doc_text[:100] + '...' if len(doc_text) > 100 else doc_text
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
    
    # Delete options
    if allow_delete:
        st.divider()
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.warning("⚠️ Danger Zone")
        
        with col2:
            if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
                st.session_state['confirm_clear'] = True
        
        if st.session_state.get('confirm_clear', False):
            st.error("Are you sure you want to delete all documents? This cannot be undone!")
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("Yes, Delete", type="primary"):
                    return 'clear_all'
            with col2:
                if st.button("Cancel"):
                    st.session_state['confirm_clear'] = False
                    st.rerun()
    
    return None


def search_results_component(results: Dict[str, Any]):
    """
    Render search results component.
    
    Args:
        results: Dictionary with search results
    """
    st.subheader("🔍 Retrieved Context")
    
    if not results.get('documents'):
        st.info("No relevant documents found.")
        return
    
    for idx, (doc, metadata, distance) in enumerate(zip(
        results.get('documents', []),
        results.get('metadatas', []),
        results.get('distances', [])
    ), 1):
        with st.expander(f"📄 Document {idx} - {metadata.get('file_name', 'Unknown')} (Similarity: {1 - distance:.2%})"):
            st.markdown(f"**Source:** {metadata.get('file_name', 'Unknown')}")
            
            if 'page' in metadata:
                st.markdown(f"**Page:** {metadata.get('page')}")
            
            if 'content_type' in metadata:
                st.markdown(f"**Type:** {metadata.get('content_type')}")
            
            st.divider()
            st.text_area("Content", doc, height=150, disabled=True, key=f"doc_{idx}")


def loading_spinner(text: str = "Processing..."):
    """
    Context manager for loading spinner.
    
    Args:
        text: Loading text to display
    """
    return st.spinner(text)


def success_message(message: str, icon: str = "✅"):
    """
    Display a success message.
    
    Args:
        message: Success message
        icon: Icon to display
    """
    st.success(f"{icon} {message}", icon="✅")


def error_message(message: str, icon: str = "❌"):
    """
    Display an error message.
    
    Args:
        message: Error message
        icon: Icon to display
    """
    st.error(f"{icon} {message}", icon="❌")


def info_message(message: str, icon: str = "ℹ️"):
    """
    Display an info message.
    
    Args:
        message: Info message
        icon: Icon to display
    """
    st.info(f"{icon} {message}", icon="ℹ️")


def metrics_row(metrics: Dict[str, Any]):
    """
    Display a row of metrics.
    
    Args:
        metrics: Dictionary of metric name -> value
    """
    cols = st.columns(len(metrics))
    
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.metric(label, value)
