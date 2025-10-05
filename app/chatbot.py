"""
Chatbot Module: DSPy-based RAG pipeline for multimodal question answering.
"""

import dspy
from typing import List, Dict, Any, Optional
from app.vector_db_clip import MultimodalVectorDatabase
from models.dnotitia_model import DNotitiaModel


class RAGSignature(dspy.Signature):
    """Signature for RAG-based question answering."""
    context = dspy.InputField(desc="Retrieved context from documents")
    question = dspy.InputField(desc="User's question")
    answer = dspy.OutputField(desc="Answer to the question based on context")


class MultimodalRAG(dspy.Module):
    """DSPy module for multimodal RAG."""
    
    def __init__(self, vector_db: MultimodalVectorDatabase, llm):
        """
        Initialize MultimodalRAG module.
        
        Args:
            vector_db: MultimodalVectorDatabase instance for retrieval
            llm: Language model for generation
        """
        super().__init__()
        self.vector_db = vector_db
        self.retrieve = dspy.Retrieve(k=5)
        self.generate_answer = dspy.ChainOfThought(RAGSignature)
    
    def forward(self, question: str, n_results: int = 5):
        """
        Forward pass for RAG.
        
        Args:
            question: User's question
            n_results: Number of documents to retrieve
            
        Returns:
            Answer with context
        """
        # Retrieve relevant documents
        search_results = self.vector_db.search(question, n_results=n_results)
        
        # Format context from retrieved documents
        context_parts = []
        for doc, metadata in zip(search_results['documents'], search_results['metadatas']):
            source = metadata.get('file_name', 'unknown')
            context_parts.append(f"[Source: {source}]\n{doc}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer using DSPy
        prediction = self.generate_answer(context=context, question=question)
        
        return dspy.Prediction(
            answer=prediction.answer,
            context=context,
            sources=search_results['metadatas']
        )


class MultimodalChatbot:
    """Main chatbot class integrating DSPy, RAG, and the DNotitia model."""
    
    def __init__(
        self,
        vector_db: MultimodalVectorDatabase,
        model: Optional[DNotitiaModel] = None,
        use_dspy: bool = True
    ):
        """
        Initialize the multimodal chatbot.
        
        Args:
            vector_db: MultimodalVectorDatabase instance
            model: DNotitia model instance (optional)
            use_dspy: Whether to use DSPy for structured generation
        """
        self.vector_db = vector_db
        self.model = model
        self.use_dspy = use_dspy
        self.conversation_history = []
        
        # Configure DSPy if enabled
        if self.use_dspy and self.model:
            # Create a DSPy-compatible LM wrapper
            self.dspy_lm = self._create_dspy_lm()
            dspy.settings.configure(lm=self.dspy_lm)
            
            # Initialize RAG module
            self.rag_module = MultimodalRAG(vector_db=self.vector_db, llm=self.dspy_lm)
    
    def _create_dspy_lm(self):
        """Create a DSPy-compatible language model wrapper."""
        # Store model reference at class level to avoid passing in __call__
        actual_model = self.model
        
        class DNotitiaLM(dspy.LM):
            def __init__(self, model_name_str):
                # Pass only the string name to DSPy, NOT the model object
                super().__init__(model=model_name_str)
                
                # Store references
                self.model = model_name_str  # STRING for DSPy
                self._actual_model = actual_model  # Actual model object
                self.history = []
                self.kwargs = {
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "top_p": 0.9,
                    "model": model_name_str
                }
            
            def __call__(self, prompt=None, messages=None, **kwargs):
                """DSPy calls this with different signatures, handle all cases."""
                # DSPy might pass prompt or messages
                if messages:
                    # Handle messages format
                    if isinstance(messages, list) and len(messages) > 0:
                        prompt = messages[-1].get('content', '') if isinstance(messages[-1], dict) else str(messages[-1])
                
                if not prompt:
                    return [""]
                
                # Generate response
                response = self._actual_model.generate(prompt, **kwargs)
                return [response]
            
            def basic_request(self, prompt, **kwargs):
                """Basic request interface."""
                if not prompt:
                    return ""
                response = self._actual_model.generate(prompt, **kwargs)
                return response
        
        # Extract model name string
        if hasattr(self.model, 'model_name'):
            model_name_str = str(self.model.model_name) if self.model.model_name else "dnotitia"
        else:
            model_name_str = "dnotitia"
        
        return DNotitiaLM(model_name_str)
    
    def chat(
        self,
        question: str,
        use_rag: bool = True,
        n_results: int = 5,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """
        Process a chat message and generate a response.
        
        Args:
            question: User's question
            use_rag: Whether to use RAG for context retrieval
            n_results: Number of documents to retrieve
            include_history: Whether to include conversation history
            
        Returns:
            Dictionary containing answer and metadata
        """
        import logging
        logging.info("="*60)
        logging.info(f"📞 CHATBOT.chat() called")
        logging.info(f"   question: {question}")
        logging.info(f"   use_rag: {use_rag}, n_results: {n_results}")
        logging.info(f"   use_dspy: {self.use_dspy}")
        logging.info(f"   model: {self.model}")
        logging.info("="*60)
        
        response = {
            'question': question,
            'answer': '',
            'context': '',
            'sources': [],
            'method': 'rag' if use_rag else 'direct'
        }
        
        try:
            logging.info(f"🔧 Starting chat processing...")
            if use_rag:
                logging.info("📚 Using RAG pipeline...")
                # Use RAG pipeline
                if self.use_dspy and hasattr(self, 'rag_module'):
                    logging.info("🔮 Using DSPy RAG module...")
                    # Use DSPy RAG module
                    try:
                        prediction = self.rag_module(question, n_results=n_results)
                        response['answer'] = prediction.answer
                        response['context'] = prediction.context
                        response['sources'] = prediction.sources
                        logging.info(f"✅ DSPy RAG completed. Answer length: {len(response['answer'])}")
                    except Exception as e:
                        logging.error(f"❌ DSPy RAG failed: {type(e).__name__}: {str(e)}")
                        import traceback
                        logging.error(traceback.format_exc())
                        raise
                else:
                    logging.info("📝 Using manual RAG (no DSPy)...")
                    # Fallback to manual RAG
                    response = self._manual_rag(question, n_results)
            else:
                logging.info("💬 Direct generation (no RAG)...")
                # Direct generation without RAG
                if self.model:
                    prompt = self._build_prompt(question, include_history=include_history)
                    logging.info(f"   Calling model.generate()...")
                    answer = self.model.generate(prompt, max_new_tokens=512)
                    response['answer'] = answer
                    logging.info(f"✅ Generation completed. Answer length: {len(answer)}")
                else:
                    response['answer'] = "Model not initialized."
                    logging.warning("⚠️ Model not initialized")
            
            # Add to conversation history
            self.conversation_history.append({
                'question': question,
                'answer': response['answer']
            })
            logging.info(f"✅ Chat completed successfully. Returning response.")
            
        except Exception as e:
            logging.error("="*60)
            logging.error(f"❌ EXCEPTION in chatbot.chat(): {type(e).__name__}")
            logging.error(f"Message: {str(e)}")
            import traceback
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            logging.error("="*60)
            
            response['answer'] = f"Error generating response: {str(e)}"
            response['error'] = str(e)
        
        return response
    
    def _manual_rag(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Manual RAG implementation without DSPy.
        
        Args:
            question: User's question
            n_results: Number of documents to retrieve
            
        Returns:
            Response dictionary
        """
        # Retrieve relevant documents
        search_results = self.vector_db.search(question, n_results=n_results)
        
        # Format context
        context_parts = []
        for doc, metadata in zip(search_results['documents'], search_results['metadatas']):
            source = metadata.get('file_name', 'unknown')
            context_parts.append(f"[Source: {source}]\n{doc}")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt with context
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        # Generate answer
        if self.model:
            answer = self.model.generate(prompt, max_new_tokens=512)
        else:
            answer = "Model not initialized."
        
        return {
            'question': question,
            'answer': answer,
            'context': context,
            'sources': search_results['metadatas'],
            'method': 'rag'
        }
    
    def _build_prompt(self, question: str, include_history: bool = True) -> str:
        """
        Build a prompt for the model.
        
        Args:
            question: Current question
            include_history: Whether to include conversation history
            
        Returns:
            Formatted prompt
        """
        prompt_parts = []
        
        # Add conversation history if requested
        if include_history and self.conversation_history:
            prompt_parts.append("Conversation History:")
            for idx, turn in enumerate(self.conversation_history[-3:], 1):  # Last 3 turns
                prompt_parts.append(f"User: {turn['question']}")
                prompt_parts.append(f"Assistant: {turn['answer']}")
            prompt_parts.append("")
        
        # Add current question
        prompt_parts.append(f"User: {question}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def add_documents_to_kb(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Add documents to the knowledge base.
        
        Args:
            file_paths: List of file paths to ingest
            
        Returns:
            Summary of ingestion results
        """
        from app.ingestion import DocumentIngestion
        
        ingestion = DocumentIngestion()
        results = ingestion.ingest_multiple_files(file_paths)
        
        total_text_chunks = 0
        total_images = 0
        
        for result in results:
            if result.get('status') == 'success':
                # Add text documents
                text_docs = result.get('text_documents', [])
                if text_docs:
                    self.vector_db.add_text_documents(text_docs)
                    total_text_chunks += len(text_docs)
                
                # Add image documents
                image_docs = result.get('image_documents', [])
                if image_docs:
                    self.vector_db.add_image_documents(image_docs)
                    total_images += len(image_docs)
        
        return {
            'files_processed': len([r for r in results if r.get('status') == 'success']),
            'total_text_chunks': total_text_chunks,
            'total_images': total_images,
            'results': results
        }
