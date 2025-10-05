"""
Main Streamlit Application: Entry point for the multimodal chatbot UI.
"""

import streamlit as st
import sys
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
    page_icon="🤖",
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


@st.cache_resource
def load_model():
    """Load and cache the language model."""
    try:
        st.info("🔄 Loading model... This may take 1-2 minutes...")
        import logging
        logging.info("="*60)
        logging.info("STREAMLIT: Starting model load")
        logging.info("="*60)
        
        model = DNotitiaModel(
            model_name="dnotitia/DNA-2.0-8B",  # Primary model, will fallback to EXAONE if needed
            max_length=2048,
            temperature=0.7
        )
        
        logging.info("="*60)
        logging.info("STREAMLIT: Model loaded successfully")
        logging.info(f"Model name: {model.model_name}")
        logging.info(f"Model name type: {type(model.model_name)}")
        logging.info("="*60)
        
        st.success(f"✅ Model loaded: {model.model_name}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.error(f"Traceback:\n```\n{traceback.format_exc()}\n```")
        st.warning("⚠️ The chatbot will work in retrieval-only mode without generation.")
        return None


def initialize_app():
    """Initialize the application components."""
    if not st.session_state.initialized:
        with st.spinner("Initializing application..."):
            # Load vector database
            st.session_state.vector_db = load_vector_db()
            
            # Load model (optional - can work without it)
            st.session_state.model = load_model()
            
            # Initialize chatbot
            if st.session_state.vector_db:
                st.session_state.chatbot = MultimodalChatbot(
                    vector_db=st.session_state.vector_db,
                    model=st.session_state.model,
                    use_dspy=True
                )
            
            # Initialize ingestion
            st.session_state.ingestion = DocumentIngestion()
            
            st.session_state.initialized = True


def chat_page():
    """Render the main chat page."""
    import logging
    logging.info("📄 STREAMLIT: Rendering chat page")
    
    st.title("🤖 Multimodal RAG Chatbot")
    st.markdown("Ask questions about your uploaded documents!")
    
    # Sidebar settings
    settings = sidebar_settings_component()
    logging.info(f"⚙️ STREAMLIT: Settings loaded: {settings}")
    
    # Sidebar info
    with st.sidebar:
        st.divider()
        st.subheader("📊 System Info")
        if st.session_state.model:
            try:
                model_info = st.session_state.model.get_model_info()
                
                # Show active model with robust error handling
                model_name = model_info.get('model_name', 'Unknown')
                
                # Ensure model_name is a string
                if not isinstance(model_name, str):
                    st.error(f"⚠️ DEBUG: model_name is {type(model_name).__name__}, not string")
                    st.error(f"Value: {repr(model_name)}")
                    model_name = str(model_name) if model_name else 'Unknown'
                
                # Safely extract display name
                try:
                    display_name = model_name.split('/')[-1] if '/' in model_name else model_name
                except AttributeError as e:
                    st.error(f"❌ Split Error: {str(e)}")
                    st.error(f"Type: {type(model_name)}, Value: {repr(model_name)}")
                    display_name = 'Error'
                
                st.text(f"Model: {display_name}")
                st.text(f"Device: {model_info.get('device', 'Unknown')}")
                
                # Show if using fallback
                if model_info.get('is_fallback', False):
                    st.info(f"ℹ️ Using fallback model\n(Primary model pending approval)")
            except Exception as e:
                st.error(f"❌ Error getting model info: {str(e)}")
                st.error(f"Error type: {type(e).__name__}")
                import traceback
                st.error(f"Traceback:\n{traceback.format_exc()}")
        else:
            st.warning("Model not loaded")
        
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
        logging.info(f"💬 STREAMLIT: User input: {prompt}")
        logging.info("="*60)
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with loading_spinner("Thinking..."):
                if st.session_state.chatbot:
                    try:
                        logging.info(f"🤖 STREAMLIT: Calling chatbot.chat()")
                        logging.info(f"   use_rag={settings['use_rag']}, n_results={settings['n_results']}")
                        
                        response = st.session_state.chatbot.chat(
                            question=prompt,
                            use_rag=settings['use_rag'],
                            n_results=settings['n_results'],
                            include_history=settings['include_history']
                        )
                        
                        # DETAILED RESPONSE LOGGING
                        logging.info("="*60)
                        logging.info("✅ STREAMLIT: Got response from chatbot.chat()")
                        logging.info(f"Response keys: {list(response.keys())}")
                        logging.info(f"Response types: {[(k, type(v).__name__) for k, v in response.items()]}")
                        
                        # Log each field
                        for key, value in response.items():
                            if key == 'answer':
                                val_str = str(value)[:200] if len(str(value)) > 200 else str(value)
                                logging.info(f"  answer: {val_str}")
                            elif key == 'error':
                                logging.error(f"  ❌ ERROR in response.error: {value}")
                            else:
                                logging.info(f"  {key}: {value}")
                        logging.info("="*60)
                        
                        answer = response.get('answer', 'No answer generated.')
                        
                        # Check if error in response
                        if 'error' in response:
                            st.error(f"❌ Error: {response['error']}")
                            import traceback
                            if 'split' in str(response['error']).lower():
                                st.error("🐛 DEBUG: This is the SPLIT error!")
                                st.error(f"Full error: {response.get('error')}")
                        
                        st.markdown(answer)
                        
                        # Show sources if available
                        if response.get('sources'):
                            with st.expander("📚 View Sources"):
                                for idx, source in enumerate(response['sources'][:3], 1):
                                    st.text(f"{idx}. {source.get('file_name', 'Unknown')}")
                    except Exception as e:
                        import logging
                        import traceback
                        
                        logging.error("="*60)
                        logging.error(f"❌ STREAMLIT EXCEPTION: {type(e).__name__}")
                        logging.error(f"Message: {str(e)}")
                        logging.error("Traceback:")
                        logging.error(traceback.format_exc())
                        logging.error("="*60)
                        
                        answer = f"Error generating response: {str(e)}"
                        st.error(f"❌ Chat Error: {str(e)}")
                        st.error(f"Error type: {type(e).__name__}")
                        
                        # Special handling for split error
                        if "'DNotitiaModel' object has no attribute 'split'" in str(e):
                            st.error("🐛 **SPLIT ERROR DETECTED!**")
                            st.error("This means somewhere the code is trying to call .split() on the model object instead of a string")
                            st.error(f"Check terminal logs for full traceback")
                        
                        st.error(f"Traceback:\n```python\n{traceback.format_exc()}\n```")
                        st.markdown(answer)
                else:
                    answer = "Chatbot not initialized. Please check the configuration."
                    st.markdown(answer)
            
            # Add assistant message to chat
            st.session_state.messages.append({"role": "assistant", "content": answer})


def document_management_page():
    """Render the document management page."""
    st.title("📁 Document Management")
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
                if st.button("🚀 Process and Add to Knowledge Base", type="primary"):
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
                                            st.success(f"✅ {file_result['metadata']['file_name']}")
                                        else:
                                            st.error(f"❌ {file_result.get('file_path')}: {file_result.get('error')}")
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
        st.subheader("💾 Vector Database")
        
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
        ["💬 Chat", "📁 Document Management", "🔍 Search & Test"],
        label_visibility="collapsed"
    )
    
    st.sidebar.divider()
    
    # Route to appropriate page
    if page == "💬 Chat":
        chat_page()
    elif page == "📁 Document Management":
        document_management_page()
    elif page == "🔍 Search & Test":
        search_page()
    
    # Footer
    st.sidebar.divider()
    st.sidebar.markdown("---")
    st.sidebar.caption("🤖 Multimodal RAG Chatbot v1.0")
    st.sidebar.caption("Powered by DSPy & DNotitia DNA-2.0-8B")


if __name__ == "__main__":
    main()
