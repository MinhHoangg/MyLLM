"""
Main Streamlit Application: Entry point for the multimodal chatbot UI.
"""

import streamlit as st
import sys
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vector_db_clip import MultimodalVectorDatabase
from app.chatbot import MultimodalChatbot
from app.ingestion import DocumentIngestion
from models.dnotitia_model import DNotitiaModel
from ui.components import (
    render_chat_history,
    file_uploader_component,
    sidebar_settings_component,
    database_stats_component,
    document_list_component,
    search_results_component,
    loading_spinner,
    success_message,
    error_message,
    info_message,
    metrics_row
)
import tempfile
import os


# Page configuration
st.set_page_config(
    page_title="Multimodal RAG Chatbot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.messages = []
        st.session_state.vector_db = None
        st.session_state.chatbot = None
        st.session_state.model = None
        st.session_state.ingestion = None
        st.session_state.confirm_clear = False
        st.session_state.load_model_on_startup = True  # False = faster startup, loads on first chat
        st.session_state.high_parameter = False  # False = 2.4B model (default, balanced), True = 7.8B model


@st.cache_resource
def load_vector_db():
    """Load and cache the vector database."""
    try:
        # Initialize vector database with CLIP support
        vector_db = MultimodalVectorDatabase(
            persist_directory="data/embeddings_db",
            collection_name="multimodal_docs",
            use_clip=True
        )
        return vector_db
    except Exception as e:
        st.error(f"Error loading vector database: {str(e)}")
        return None


@st.cache_resource(show_spinner=False)
def load_model(_high_parameter: bool = False):
    """Load the chatbot model with caching
    
    Args:
        _high_parameter: If True, use 7.8B model. If False, use 2.4B model.
                        Underscore prefix makes it part of the cache key.
    """
    with st.spinner("Loading model... This may take a few minutes on first run."):
        model = DNotitiaModel(high_parameter=_high_parameter)
        return model


def initialize_app():
    """Initialize the application components."""
    if not st.session_state.initialized:
        with st.spinner("Initializing application..."):
            # Load vector database (fast)
            st.session_state.vector_db = load_vector_db()
            
            # Load model (slow - optional based on setting)
            if st.session_state.load_model_on_startup:
                high_param = st.session_state.get('high_parameter', False)
                param_size = "7.8B, ~90s" if high_param else "2.4B, ~40-50s"
                st.info(f" Loading AI model ({param_size})...")
                st.session_state.model = load_model(_high_parameter=high_param)
            else:
                st.info(" Skipped model loading for faster startup. Model will load on first chat message.")
                st.session_state.model = None
            
            # Initialize chatbot (disable DSPy to avoid threading issues and improve speed)
            if st.session_state.vector_db:
                st.session_state.chatbot = MultimodalChatbot(
                    vector_db=st.session_state.vector_db,
                    model=st.session_state.model,
                    use_dspy=False  # Disabled for performance and stability
                )
            
            # Initialize ingestion
            st.session_state.ingestion = DocumentIngestion()
            
            st.session_state.initialized = True


def chat_page():
    """Render the main chat page."""
    import logging
    logging.info(" STREAMLIT: Rendering chat page")
    
    st.title(" Multimodal RAG Chatbot")
    st.markdown("Ask questions about your uploaded documents!")
    
    # Sidebar settings
    settings = sidebar_settings_component()
    logging.info(f"STREAMLIT: Settings loaded: {json.dumps(settings, indent=2, default=str)}")
    
    # Sidebar info
    with st.sidebar:
        st.divider()
        
        # Model configuration section
        st.subheader("⚙️ Model Configuration")
        
        # High parameter toggle
        current_high_param = st.session_state.get('high_parameter', False)
        new_high_param = st.checkbox(
            "Use High-Performance Model (7.8B)",
            value=current_high_param,
            help=" True: EXAONE-3.5-7.8B (7.8B params, high quality, ~15.6GB VRAM, ~2-3 min load)\n"
                 " False: EXAONE-3.5-2.4B (2.4B params, balanced, ~4.8GB VRAM, ~40-50s load)",
            key="high_param_checkbox"
        )
        
        # Show current model size
        model_size_text = "7.8B (High Performance)" if new_high_param else "2.4B (Balanced)"
        st.caption(f" Selected: {model_size_text}")
        
        # Detect if setting changed - show reload button
        if new_high_param != current_high_param:
            st.warning("WARNING: Model setting changed. Click button below to reload.")
            if st.button(" Reload Model Now", type="primary", use_container_width=True):
                with st.spinner("Reloading model... Please wait."):
                    # Clear the cache
                    load_model.clear()
                    # Update session state
                    st.session_state.high_parameter = new_high_param
                    # Reload model with new setting
                    new_model = load_model(_high_parameter=new_high_param)
                    st.session_state.model = new_model
                    # Update chatbot's model reference
                    if st.session_state.chatbot:
                        st.session_state.chatbot.model = new_model
                    st.success(f" Model reloaded! Now using {'7.8B' if new_high_param else '2.4B'} variant.")
                    st.rerun()
        
        st.divider()
        st.subheader(" System Info")
        if st.session_state.model:
            try:
                model_info = st.session_state.model.get_model_info()
                model_name = model_info.get('model_name', 'Unknown')
                
                # Ensure model_name is a string (log errors to terminal)
                if not isinstance(model_name, str):
                    logging.error(f"WARNING: model_name is {type(model_name).__name__}, not string: {repr(model_name)}")
                    model_name = str(model_name) if model_name else 'Unknown'
                
                # Safely extract display name (log errors to terminal)
                try:
                    display_name = model_name.split('/')[-1] if '/' in model_name else model_name
                except AttributeError as e:
                    logging.error(f" Split Error: {str(e)} | Type: {type(model_name)}, Value: {repr(model_name)}")
                    display_name = 'Unknown'
                
                # Clean display
                st.text(f"Model: {display_name}")
                st.text(f"Device: {model_info.get('device', 'Unknown')}")
            except Exception as e:
                logging.error(f" Error getting model info: {str(e)}")
                logging.error(traceback.format_exc())
                st.text("Model: Loading...")
        else:
            st.text("Model: Not loaded")
        
        if st.session_state.vector_db:
            stats = st.session_state.vector_db.get_collection_stats()
            st.text(f"Documents: {stats['count']}")
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.chatbot:
                st.session_state.chatbot.clear_history()
            st.rerun()
    
    # Display chat history
    render_chat_history(st.session_state.messages)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        import logging
        logging.info("="*60)
        logging.info(f" STREAMLIT: User input: {prompt}")
        logging.info("="*60)
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with loading_spinner("Thinking..."):
                # Lazy load model if not loaded yet
                if st.session_state.model is None and st.session_state.chatbot:
                    high_param = st.session_state.get('high_parameter', False)
                    load_time = "~90 seconds" if high_param else "~40-50 seconds"
                    # Log to terminal instead of showing in UI
                    logging.info(f" Loading AI model for the first time ({load_time})...")
                    st.session_state.model = load_model(_high_parameter=high_param)
                    if st.session_state.model:
                        st.session_state.chatbot.model = st.session_state.model
                    logging.info(" Model loaded successfully!")
                
                if st.session_state.chatbot:
                    try:
                        logging.info(f" STREAMLIT: Calling chatbot.chat()")
                        
                        # Use RAG setting from sidebar (None = auto-detect, True = force on, False = force off)
                        use_rag_setting = settings.get('use_rag', None)
                        rag_mode_str = "auto-detect" if use_rag_setting is None else str(use_rag_setting)
                        logging.info(f"   use_rag={rag_mode_str}, n_results={settings['n_results']}")
                        
                        response = st.session_state.chatbot.chat(
                            question=prompt,
                            use_rag=use_rag_setting,  # None = auto-detect, True/False = force
                            n_results=settings['n_results'],
                            include_history=settings['include_history']
                        )
                        
                        # Log response details to terminal only
                        logging.info("="*60)
                        logging.info("STREAMLIT: Got response from chatbot.chat()")
                        
                        # Format response for logging
                        log_response = response.copy()
                        if 'answer' in log_response and len(str(log_response['answer'])) > 200:
                            log_response['answer'] = str(log_response['answer'])[:200] + f"... (total {len(str(response['answer']))} chars)"
                        if 'context' in log_response and len(str(log_response['context'])) > 200:
                            log_response['context'] = str(log_response['context'])[:200] + "..."
                        
                        logging.info(f"Response: {json.dumps(log_response, indent=2, default=str)}")
                        logging.info("="*60)
                        
                        # Get answer
                        answer = response.get('answer', 'No answer generated.')
                        
                        # Log errors to terminal only (don't show in UI)
                        if 'error' in response:
                            logging.error(f" Error in response: {response['error']}")
                        
                        # Display ONLY the answer - clean UI
                        st.markdown(answer)
                    except Exception as e:
                        import logging
                        import traceback
                        
                        # Log full error details to terminal
                        logging.error("="*60)
                        logging.error(f" STREAMLIT EXCEPTION: {type(e).__name__}")
                        logging.error(f"Message: {str(e)}")
                        logging.error("Traceback:")
                        logging.error(traceback.format_exc())
                        logging.error("="*60)
                        
                        # Show clean error message in UI only
                        answer = "I apologize, but I encountered an error processing your request. Please check the terminal logs for details."
                        st.markdown(answer)
                else:
                    answer = "Chatbot not initialized. Please check the configuration."
                    st.markdown(answer)
            
            # Add assistant message to chat
            st.session_state.messages.append({"role": "assistant", "content": answer})


def document_management_page():
    """Render the document management page."""
    st.title(" Document Management")
    st.markdown("Upload and manage documents for the chatbot's knowledge base.")
    
    # Two columns: upload and database
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Documents")
        
        if st.session_state.ingestion:
            supported_exts = st.session_state.ingestion.get_supported_extensions()
            uploaded_files = file_uploader_component(
                supported_extensions=supported_exts,
                accept_multiple=True,
                key="doc_uploader"
            )
            
            if uploaded_files:
                if st.button(" Process and Add to Knowledge Base", type="primary"):
                    with loading_spinner("Processing documents..."):
                        # Save uploaded files temporarily
                        temp_paths = []
                        for uploaded_file in uploaded_files:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                                tmp.write(uploaded_file.getbuffer())
                                temp_paths.append(tmp.name)
                        
                        # Process files
                        try:
                            if st.session_state.chatbot:
                                result = st.session_state.chatbot.add_documents_to_kb(temp_paths)
                                
                                # Show results
                                success_message(
                                    f"Successfully processed {result['files_processed']} files!"
                                )
                                
                                metrics_row({
                                    "Text Chunks": result['total_text_chunks'],
                                    "Images": result['total_images']
                                })
                                
                                # Show detailed results
                                with st.expander("📋 Processing Details"):
                                    for file_result in result['results']:
                                        if file_result.get('status') == 'success':
                                            st.success(f" {file_result['metadata']['file_name']}")
                                        else:
                                            st.error(f" {file_result.get('file_path')}: {file_result.get('error')}")
                            else:
                                error_message("Chatbot not initialized")
                        
                        except Exception as e:
                            error_message(f"Error processing files: {str(e)}")
                        
                        finally:
                            # Clean up temporary files
                            for path in temp_paths:
                                try:
                                    os.unlink(path)
                                except:
                                    pass
                        
                        st.rerun()
        else:
            error_message("Document ingestion system not initialized")
    
    with col2:
        st.subheader(" Vector Database")
        
        if st.session_state.vector_db:
            # Show database stats
            stats = st.session_state.vector_db.get_collection_stats()
            database_stats_component(stats)
            
            # Show documents
            st.divider()
            
            # Get documents with limit
            documents = st.session_state.vector_db.get_all_documents(limit=50)
            action = document_list_component(documents, allow_delete=True)
            
            if action == 'clear_all':
                with loading_spinner("Clearing database..."):
                    if st.session_state.vector_db.clear_collection():
                        success_message("Database cleared successfully!")
                        st.session_state['confirm_clear'] = False
                    else:
                        error_message("Failed to clear database")
                st.rerun()
        else:
            error_message("Vector database not initialized")


def search_page():
    """Render the search/test page."""
    st.title("🔍 Search & Test")
    st.markdown("Test the retrieval system and search your documents.")
    
    # Search input
    query = st.text_input(
        "Enter your search query:",
        placeholder="What would you like to search for?",
        key="search_query"
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        n_results = st.number_input(
            "Number of results:",
            min_value=1,
            max_value=20,
            value=5,
            key="search_n_results"
        )
    
    if query:
        if st.button("🔍 Search", type="primary"):
            with loading_spinner("Searching..."):
                if st.session_state.vector_db:
                    results = st.session_state.vector_db.search(
                        query=query,
                        n_results=n_results
                    )
                    
                    if results['documents']:
                        success_message(f"Found {len(results['documents'])} relevant documents")
                        search_results_component(results)
                    else:
                        info_message("No documents found. Try uploading some documents first.")
                else:
                    error_message("Vector database not initialized")


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Initialize app
    initialize_app()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Go to:",
        [" Chat", " Document Management", "🔍 Search & Test"],
        label_visibility="collapsed"
    )
    
    st.sidebar.divider()
    
    # Route to appropriate page
    if page == " Chat":
        chat_page()
    elif page == " Document Management":
        document_management_page()
    elif page == "🔍 Search & Test":
        search_page()
    
    # Footer
    st.sidebar.divider()
    st.sidebar.markdown("---")
    st.sidebar.caption(" Multimodal RAG Chatbot v1.0")
    st.sidebar.caption("Powered by DSPy & DNotitia DNA-2.0-8B")


if __name__ == "__main__":
    main()
