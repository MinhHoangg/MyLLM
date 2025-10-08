# Multimodal RAG Chatbot# Multimodal RAG Chatbot with DSPy



Production-ready multimodal chatbot with document understanding, powered by **EXAONE-3.5**, **CLIP embeddings**, and **RAG** (Retrieval-Augmented Generation).A sophisticated multimodal chatbot built with **DSPy**, powered by **Hugging Face language models** (dnotitia/DNA-2.0-1.7B with automatic fallback to upstage/SOLAR-10.7B-v1.0), featuring a complete **Retrieval-Augmented Generation (RAG)** pipeline that supports both text and image content.



## 🚀 Quick Start## 🎯 Project Overview



Choose your operating system:This project implements a production-ready multimodal chatbot that can:

- **macOS**: See [SETUP_MACOS.md](SETUP_MACOS.md)- Process multiple file formats (PDF, DOCX, XLSX, TXT, images, etc.)

- **Windows**: See [SETUP_WINDOWS.md](SETUP_WINDOWS.md)- Extract and process both text and images from documents

- **Linux**: See [SETUP_LINUX.md](SETUP_LINUX.md)- Store content in a vector database for efficient retrieval

- Answer questions using RAG with DSPy orchestration

## 🏗️ Architecture- Provide an intuitive Streamlit-based user interface

- **Automatic model fallback**: Works immediately with SOLAR-10.7B if primary model pending approval

**Backend:** FastAPI + Python (Port 8000)

- REST API for chat, documents, search### Key Features

- WebSocket support for streaming

- Offline-first model loading✅ **Multimodal Document Processing**: Handles text and images from various file formats  

- Adaptive content-aware chunking✅ **Advanced RAG Pipeline**: Uses DSPy for structured retrieval and generation  

- CLIP multimodal embeddings✅ **CLIP Integration**: True multimodal understanding with text-image embeddings ⭐ NEW  

✅ **Vector Database**: ChromaDB for efficient semantic search  

**Frontend:** React + Vite (Port 3000)✅ **LangChain Integration**: Intelligent document chunking  

- Modern React with hooks✅ **Modern UI**: Interactive Streamlit interface with chat and management features  

- Drag-and-drop file upload✅ **Production-Ready**: Modular architecture with proper error handling

- Real-time chat interface

- Document management---

- Search functionality

## 🏗️ Architecture

## ✨ Features

### System Components

- ✅ **Unlimited responses** - 4096 tokens (~3000 words)

- ✅ **RAG auto-detection** - Automatically uses documents when needed```

- ✅ **Multimodal** - Text + image understanding with CLIP┌─────────────────────────────────────────────────────────────┐

- ✅ **Adaptive chunking** - 5 content-aware strategies│                     Streamlit UI Layer                       │

- ✅ **Offline-first** - Models load from cache after first download│  (Chat Interface | Document Management | Search & Test)     │

- ✅ **Unlimited search** - All documents with similarity threshold└──────────────────────┬──────────────────────────────────────┘

- ✅ **Professional logging** - JSON-formatted                       │

- ✅ **Conversation history** - Multi-turn context-aware chat┌──────────────────────▼──────────────────────────────────────┐

- ✅ **Model switching** - 2.4B (fast) ↔ 7.8B (high-performance)│                  Application Layer                           │

│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │

## 🤖 Models│  │   Chatbot    │  │  Ingestion   │  │  Vector DB   │      │

│  │   (DSPy)     │  │  (LangChain) │  │  (ChromaDB)  │      │

| Model | Size | Load Time | Use Case |│  └──────────────┘  └──────────────┘  └──────────────┘      │

|-------|------|-----------|----------|└──────────────────────┬──────────────────────────────────────┘

| EXAONE-3.5-2.4B | ~4.8GB | 40-50s | Default, fast responses |                       │

| EXAONE-3.5-7.8B | ~15.6GB | 2-3min | High-performance mode |┌──────────────────────▼──────────────────────────────────────┐

| CLIP ViT-B/32 | ~600MB | <10s | Multimodal embeddings |│                  File Handlers Layer                         │

│  [PDF] [Images] [DOCX] [XLSX] [TXT/LOG]                    │

## 📁 Project Structure└──────────────────────┬──────────────────────────────────────┘

                       │

```┌──────────────────────▼──────────────────────────────────────┐

ChatBot/│                    Model Layer                               │

├── backend/               # FastAPI application│            DNotitia DNA-2.0-1.7B (Hugging Face)            │

│   ├── main.py           # App entry point└─────────────────────────────────────────────────────────────┘

│   ├── schemas.py        # Pydantic models```

│   ├── routes/           # API endpoints

│   │   ├── chat.py       # Chat endpoints### Data Flow

│   │   ├── documents.py  # Document management

│   │   ├── search.py     # Search endpoint1. **Document Upload** → File Handlers extract content (text + images)

│   │   └── model.py      # Model configuration2. **Chunking** → LangChain splits text into semantic chunks

│   └── requirements.txt3. **Embedding** → Sentence-Transformers generate vector embeddings

├── frontend/             # React application4. **Storage** → ChromaDB stores embeddings with metadata

│   ├── src/5. **Query** → User asks a question

│   │   ├── components/   # UI components6. **Retrieval** → ChromaDB finds relevant chunks (semantic search)

│   │   ├── pages/        # Page components7. **Generation** → DSPy orchestrates LLM to generate answer with context

│   │   ├── hooks/        # Custom hooks8. **Response** → Answer displayed with sources

│   │   ├── services/     # API client

│   │   └── context/      # Global state---

│   ├── package.json

│   └── vite.config.js## 📁 Project Structure

├── app/                  # Core business logic

│   ├── chatbot.py        # Chatbot with RAG```

│   ├── vector_db_clip.py # Vector databaseChatBot/

│   └── ingestion.py      # Document processing├── app/

├── models/               # Model wrappers│   ├── __init__.py

│   ├── dnotitia_model.py│   ├── chatbot.py              # DSPy RAG pipeline & chatbot logic

│   ├── clip_embedder.py│   ├── ingestion.py            # Document processing & chunking

│   └── exaone_fallback_model.py│   ├── vector_db.py            # ChromaDB operations

└── data/│   ├── file_handlers/

    └── embeddings_db/    # ChromaDB storage│   │   ├── __init__.py

```│   │   ├── pdf_handler.py      # PDF extraction (PyMuPDF)

│   │   ├── image_handler.py    # Image processing with OCR

## 🔌 API Endpoints│   │   ├── doc_handler.py      # DOCX processing

│   │   ├── xlsx_handler.py     # Excel spreadsheet processing

### Chat│   │   └── txt_handler.py      # Text files (TXT, LOG, etc.)

- `POST /api/chat/` - Send message│   └── utils.py                # Utility functions

- `POST /api/chat/clear-history` - Clear history│

- `WebSocket /api/chat/ws` - Streaming chat├── models/

│   ├── __init__.py

### Documents│   └── dnotitia_model.py       # Hugging Face model wrapper

- `POST /api/documents/upload` - Upload files│

- `GET /api/documents/` - List documents├── ui/

- `DELETE /api/documents/{id}` - Delete document│   ├── __init__.py

- `DELETE /api/documents/` - Clear all│   ├── app.py                  # Main Streamlit application

- `GET /api/documents/stats` - Statistics│   └── components.py           # Reusable UI components

│

### Search├── data/

- `POST /api/search/` - Search documents│   ├── uploads/                # Uploaded files

│   ├── chunks/                 # Processed chunks

### Model│   └── embeddings_db/          # ChromaDB persistence

- `GET /api/model/info` - Model info│

- `POST /api/model/config` - Update settings├── tests/

- `GET /api/model/settings` - Get settings│   └── test_app.py            # Unit tests

│

### Health├── .streamlit/

- `GET /api/health` - Health check│   └── config.toml            # Streamlit configuration

│

**API Documentation:** http://localhost:8000/docs (Swagger UI)├── requirements.txt            # Python dependencies

└── README.md                   # This file

## 💡 Usage Examples```



### Chat Request---

```bash

curl -X POST http://localhost:8000/api/chat/ \## 🔧 Technical Details

  -H "Content-Type: application/json" \

  -d '{### 1. File Handlers

    "question": "What is in my documents?",

    "use_rag": null,Each handler is specialized for specific file types:

    "similarity_threshold": 0.3

  }'#### **PDF Handler** (`pdf_handler.py`)

```- Uses **PyMuPDF (fitz)** for robust PDF processing

- Extracts text from each page

### Upload Documents- Extracts embedded images with metadata

```bash- Handles complex PDF structures

curl -X POST http://localhost:8000/api/documents/upload \

  -F "files=@document.pdf" \#### **Image Handler** (`image_handler.py`)

  -F "files=@image.jpg" \- Supports PNG, JPG, JPEG, BMP, TIFF, GIF

  -F "use_adaptive_chunking=true"- **OCR** with pytesseract to extract text from images

```- Image preprocessing (RGBA → RGB conversion)

- Stores PIL Image objects for display

### Search

```bash#### **DOC/DOCX Handler** (`doc_handler.py`)

curl -X POST http://localhost:8000/api/search/ \- Uses **python-docx** library

  -H "Content-Type: application/json" \- Extracts text from paragraphs

  -d '{- Processes tables and preserves structure

    "query": "machine learning",- Extracts document metadata (author, title)

    "similarity_threshold": 0.3

  }'#### **XLSX Handler** (`xlsx_handler.py`)

```- Uses **openpyxl** and **pandas**

- Processes multiple sheets

### Switch Model- Converts tabular data to readable text format

```bash- Handles formulas and calculated values

curl -X POST http://localhost:8000/api/model/config \

  -H "Content-Type: application/json" \#### **TXT/LOG Handler** (`txt_handler.py`)

  -d '{"high_parameter": true}'  # true=7.8B, false=2.4B- Universal text file processor

```- Automatic encoding detection with **chardet**

- Supports TXT, LOG, MD, CSV, JSON, XML, YAML

## ⚙️ Configuration- Handles large files efficiently



### Environment Variables### 2. Document Ingestion (`ingestion.py`)

Create `.env` in frontend directory:

```bashThe ingestion module orchestrates the entire document processing pipeline:

VITE_API_URL=http://localhost:8000

```1. **File Detection**: Automatically selects appropriate handler

2. **Content Extraction**: Extracts text and images

### Similarity Threshold3. **Chunking**: Uses LangChain's `RecursiveCharacterTextSplitter`

Controls search sensitivity (0.0-1.0):   - Chunk size: 1000 characters

- `0.1-0.2` - Very lenient (more results)   - Overlap: 200 characters

- `0.3` - Default (balanced)   - Smart splitting on natural boundaries (paragraphs, sentences)

- `0.4-0.5` - Strict (fewer, more relevant results)4. **Metadata Preservation**: Maintains source information, page numbers, etc.

5. **Image Processing**: Prepares images for embedding

## 🛠️ Development

### 3. Vector Database (`vector_db.py`)

### Backend (with hot reload)

```bash**ChromaDB** implementation with:

cd backend

uvicorn main:app --reload --port 8000- **Persistent Storage**: Data persists across sessions

```- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`

- **Cosine Similarity**: For semantic search

### Frontend (with HMR)- **Metadata Filtering**: Search by file, type, page, etc.

```bash- **CRUD Operations**: Add, search, delete, clear collections

cd frontend

npm run dev**Key Methods**:

```- `add_text_documents()`: Store text chunks with embeddings

- `add_image_documents()`: Store image metadata with embeddings

### Build Frontend for Production- `search()`: Semantic search with filters

```bash- `get_all_documents()`: Retrieve all stored documents

cd frontend- `clear_collection()`: Reset database

npm run build

npm run preview  # Preview production build### 3.5. CLIP Multimodal Embeddings ⭐ NEW (`vector_db_clip.py`, `clip_embedder.py`)

```

**Enhanced multimodal understanding** with OpenAI's CLIP:

## 📊 Performance

#### **Why CLIP?**

- **Response length:** 4096 tokens max (~3000 words)Traditional approaches store images as text descriptions. CLIP creates **aligned embeddings** where text and images share the same vector space, enabling:

- **Model loading:** 40-50s (2.4B), 2-3min (7.8B), cached after first load

- **Embeddings:** 512-dimensional CLIP vectors- **True Visual Understanding**: Images represented by what they contain, not just descriptions

- **Chunking:** Adaptive (100-1200 chars based on content type)- **Cross-Modal Retrieval**: Find images using text, or text using images

- **Search:** Unlimited results with similarity filtering- **Zero-Shot Classification**: Understand new concepts without additional training

- **Better Semantic Matching**: More accurate alignment between user intent and content

## 🧩 Adaptive Chunking Strategies

#### **CLIPEmbedder** (`models/clip_embedder.py`)

The system automatically selects the best chunking strategy:```python

# Encode text and images in the same space

| Content Type | Chunk Size | Overlap | Use Case |embedder = CLIPEmbedder(model_name="openai/clip-vit-base-patch32")

|--------------|------------|---------|----------|

| Structured Document | 400-800 | 100 | Headers, sections, bullets |text_emb = embedder.encode_text(["a red sunset over ocean"])

| List Data | 100-400 | 50 | Key-value pairs, lists |image_emb = embedder.encode_images([image])

| Dense Text | 600-1200 | 150 | Technical docs, articles |similarity = embedder.compute_similarity(text_emb, image_emb)

| Sparse Text | 150-300 | 50 | Short entries, logs |```

| Narrative Text | 512 | 128 | Stories, conversations |

**Features**:

## 🔧 Technologies- **Unified Embedding Space**: 512/768-dimensional vectors for both text and images

- **Multiple Models**: ViT-B/32 (600MB, fast) or ViT-L/14 (1.7GB, accurate)

- **Backend:** FastAPI, Uvicorn, ChromaDB, Transformers, PyTorch- **Auto Device Detection**: CUDA → MPS → CPU

- **Frontend:** React 18, Vite, React Query, React Router, Axios- **Batch Processing**: Efficient handling of multiple images

- **AI/ML:** EXAONE-3.5, CLIP, Hugging Face Transformers- **HybridEmbedder**: Combines CLIP + sentence-transformers

- **Vector DB:** ChromaDB with CLIP embeddings

- **Language Support:** Python 3.8+, Node.js 18+#### **MultimodalVectorDatabase** (`app/vector_db_clip.py`)



## 🎯 Key Improvements Over StreamlitEnhanced ChromaDB with CLIP support:



This is a complete architectural migration from Streamlit to FastAPI + React:```python

db = MultimodalVectorDatabase(use_clip=True)

1. **Separation of Concerns** - API backend separate from UI frontend

2. **Scalability** - RESTful API can serve multiple clients# Search with text (finds both text and images)

3. **Maintainability** - Modern React component structureresults = db.search("beautiful sunset")

4. **Developer Experience** - Hot reload for both backend and frontend

5. **Production Ready** - Proper error handling, logging, validation# Search with image (finds similar images and related text)

6. **API Documentation** - Automatic Swagger/OpenAPI docsresults = db.search(query_image, query_type='image')

7. **Type Safety** - Pydantic for backend, PropTypes for frontend

8. **Modern Tooling** - Vite for fast builds, React Query for caching# Multimodal fusion search

9. **WebSocket Ready** - Foundation for streaming responsesresults = db.search_multimodal(

10. **No Docker Complexity** - Direct execution, easier debugging    text_query="sunset over water",

    image_query=example_image,

## 🐛 Troubleshooting    alpha=0.5  # Balance text and image importance

)

### Backend won't start```

```bash

# Check Python version (3.8+ required)**Capabilities**:

python --version- **Text → Image**: Find images matching text description

- **Image → Image**: Find visually similar images

# Reinstall dependencies- **Image → Text**: Find text related to image content

pip install -r requirements.txt- **Multimodal Fusion**: Combined text + image queries with adjustable weighting

pip install -r backend/requirements.txt

> **📘 Learn More**: See [docs/general/CLIP_GUIDE.md](docs/general/CLIP_GUIDE.md) for detailed usage guide  

# Check if port 8000 is available> **🔧 Integration**: See [CLIP_INTEGRATION.md](CLIP_INTEGRATION.md) for step-by-step setup

lsof -i :8000  # macOS/Linux

netstat -ano | findstr :8000  # Windows### 4. DSPy RAG Pipeline (`chatbot.py`)

```

**DSPy** (Declarative Self-improving Language Programs) provides structured LM programming:

### Frontend won't start

```bash#### **RAG Components**:

# Check Node.js version (18+ required)

node --version```python

class RAGSignature(dspy.Signature):

# Clear cache and reinstall    context = dspy.InputField(desc="Retrieved context")

cd frontend    question = dspy.InputField(desc="User's question")

rm -rf node_modules package-lock.json    answer = dspy.OutputField(desc="Answer based on context")

npm install

```class MultimodalRAG(dspy.Module):

    def __init__(self, vector_db, llm):

### Model loading fails        self.retrieve = dspy.Retrieve(k=5)

```bash        self.generate_answer = dspy.ChainOfThought(RAGSignature)

# Check Hugging Face cache```

ls ~/.cache/huggingface/hub/

#### **Pipeline Flow**:

# Clear cache and re-download1. **Retrieval**: Search vector DB for relevant chunks

rm -rf ~/.cache/huggingface/hub/models--LGAI-EXAONE*2. **Context Formatting**: Combine retrieved documents

3. **Chain-of-Thought**: DSPy generates structured reasoning

# Verify internet connection for first download4. **Answer Generation**: LLM produces final answer

```5. **Source Attribution**: Returns relevant metadata



### CORS errors### 5. Model Integration (`dnotitia_model.py`)

```bash

# Verify proxy in vite.config.jsWrapper for **multiple language models** with automatic fallback:

# Verify CORS middleware in backend/main.py

# Check both servers are running on correct ports**Primary Model**: dnotitia/DNA-2.0-1.7B (1.7B params, requires approval)  

```**Fallback Model**: upstage/SOLAR-10.7B-v1.0 (10.7B params, open access)



## 📝 License**Features**:

- **Auto Device Selection**: GPU (CUDA/MPS) or CPU

MIT- **Automatic Fallback**: Uses SOLAR if primary model unavailable

- **Half-Precision**: FP16 on GPU for efficiency

## 🤝 Contributing- **Configurable Sampling**: Temperature, top_p, max_tokens

- **Batch Generation**: Process multiple prompts

Contributions welcome! Please:- **Token Management**: Handles padding, EOS tokens

1. Fork the repository- **Smart Error Handling**: Detects auth errors and switches models

2. Create a feature branch

3. Make your changes**Model Parameters**:

4. Submit a pull request- Max sequence length: 2048 tokens

- Temperature: 0.7 (configurable)

## 📧 Support- Top-p: 0.9 (nucleus sampling)



For issues or questions:> **Note**: If dnotitia model is pending approval, the app automatically uses SOLAR-10.7B. No manual intervention needed! See [docs/general/MODEL_GUIDE.md](docs/general/MODEL_GUIDE.md) for details.

- Create an issue on GitHub

- Check existing documentation### 6. Streamlit UI (`ui/app.py`, `ui/components.py`)

- Review API docs at http://localhost:8000/docs

**Three Main Pages**:

---

#### **💬 Chat Page**

**Note:** All existing business logic (chatbot, RAG, vector DB, ingestion) from the original Streamlit version has been preserved and wrapped with API endpoints. No changes to core functionality.- Real-time conversation interface

- Message history display
- Streaming responses
- Source attribution
- Settings sidebar (RAG, temperature, tokens)

#### **📁 Document Management**
- Multi-file upload
- Processing progress indicators
- Database statistics
- Document list with metadata
- Clear/delete operations

#### **🔍 Search & Test**
- Direct vector DB search
- Adjustable result count
- Similarity scores
- Context preview

**Reusable Components**:
- `render_chat_history()`: Display messages
- `file_uploader_component()`: Multi-format upload
- `sidebar_settings_component()`: Configuration panel
- `database_stats_component()`: DB metrics
- `search_results_component()`: Result display

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) CUDA-capable GPU for faster inference
- (Optional) Tesseract OCR for image text extraction

> **📝 Note for Windows Users**: See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for detailed Windows installation instructions and use the provided batch files (`setup_windows.bat` and `run_windows.bat`) for easy setup!

### Step 1: Install Tesseract (for OCR)

**macOS**:
```bash
brew install tesseract
```

**Ubuntu/Debian**:
```bash
sudo apt-get install tesseract-ocr
```

**Windows**:
Download installer from: https://github.com/UB-Mannheim/tesseract/wiki  
Or see [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for detailed instructions

### Step 2: Clone or Navigate to Project

```bash
cd "/Users/hoangquangminh/Library/CloudStorage/OneDrive-FPTCorporation/ChatBot"
```

### Step 3: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
source .venv/bin/activate && streamlit run ui/app.py  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: First run may take 10-15 minutes to download models and dependencies.

### Step 5: Run the Application

**macOS/Linux**:
```bash
streamlit run ui/app.py
```

**Windows**:
```cmd
streamlit run ui/app.py
```
Or simply double-click `run_windows.bat`

The application will open in your browser at `http://localhost:8501`

---

## 💡 Usage Guide

### 1. First Time Setup

When you first run the application:
- The system will attempt to download **dnotitia/DNA-2.0-1.7B** model (~3.5GB)
  - If unavailable/pending approval: automatically falls back to **SOLAR-10.7B** (~21GB)
- Initialize the **sentence-transformers** embedding model (~90MB)
- Create the ChromaDB database directory

**Expected Download Time**:
- dnotitia model: 5-10 minutes (if approved)
- SOLAR fallback: 10-15 minutes (open access, no approval needed)
- Embedding model: 1-2 minutes

> 💡 **No waiting for approval!** The app works immediately with SOLAR if dnotitia is pending.

### 2. Upload Documents

1. Navigate to **📁 Document Management** page
2. Click **"Upload documents"** button
3. Select one or more files (PDF, DOCX, XLSX, images, TXT, etc.)
4. Click **"🚀 Process and Add to Knowledge Base"**
5. Wait for processing (progress shown)
6. Check statistics and document list

### 3. Chat with Your Documents

1. Navigate to **💬 Chat** page
2. Type your question in the input box
3. The chatbot will:
   - Search the vector database for relevant content
   - Retrieve top-5 most similar chunks
   - Generate an answer using the DNotitia model
   - Show sources used for the answer
4. Continue the conversation naturally

### 4. Search & Test

1. Navigate to **🔍 Search & Test** page
2. Enter a search query
3. Adjust number of results
4. View retrieved documents with similarity scores

### 5. Adjust Settings

Use the sidebar to configure:
- **Use RAG**: Toggle retrieval-augmented generation
- **Number of Retrieved Documents**: 1-10
- **Temperature**: 0.1-2.0 (higher = more creative)
- **Max Tokens**: 128-2048 (response length)
- **Include Chat History**: Use conversation context

---

## 📊 Technical Specifications

### Supported File Formats

| Format | Extensions | Features |
|--------|------------|----------|
| PDF | `.pdf` | Text + embedded images extraction |
| Word | `.doc`, `.docx` | Text, tables, metadata |
| Excel | `.xlsx`, `.xls` | Multi-sheet, formulas, tables |
| Images | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.gif` | OCR text extraction |
| Text | `.txt`, `.log`, `.md`, `.csv`, `.json`, `.xml`, `.yaml` | Various encodings |

### Performance Metrics

- **Embedding Speed**: ~100 documents/second
- **Search Latency**: <100ms for 10,000 documents
- **Model Inference**: 
  - GPU: ~1-2 seconds per response
  - CPU: ~5-10 seconds per response
- **Max Document Size**: Limited by available memory

### Resource Requirements

| Component | CPU | GPU (Recommended) |
|-----------|-----|-------------------|
| Model Loading | 4GB RAM | 4GB VRAM |
| Inference | 2-4GB RAM | 2-3GB VRAM |
| Vector DB | 1GB per 100k docs | N/A |
| Total | 8GB+ RAM | 8GB+ VRAM |

---

## 🔬 Academic Explanation

### For Your Professor

This project demonstrates several advanced concepts in modern AI/ML systems:

#### 1. **Retrieval-Augmented Generation (RAG)**
- Combines information retrieval with language generation
- Reduces hallucination by grounding responses in actual documents
- Enables answering questions about private/custom data

#### 2. **DSPy Framework**
- **Declarative programming** for language models
- Separates concerns: retrieval vs. generation
- Enables optimization and self-improvement
- More maintainable than prompt engineering

#### 3. **Vector Embeddings & Semantic Search**
- Transforms text into high-dimensional vectors
- Cosine similarity for semantic matching
- Efficient approximate nearest neighbor search

#### 4. **Multimodal Processing**
- Handles heterogeneous data (text, images, tables)
- Unified embedding space for different modalities
- OCR integration for image-to-text conversion

#### 5. **Production Engineering**
- Modular architecture (separation of concerns)
- Error handling and logging
- Stateful UI with session management
- Persistent storage for long-term use

### Key Innovations

1. **Unified Multimodal Pipeline**: Single system handles all document types
2. **DSPy Integration**: Structured LM programming instead of prompt chains
3. **Modular File Handlers**: Easy to extend with new formats
4. **Real-time Processing**: Immediate feedback during document ingestion
5. **Interactive UI**: Non-technical users can operate the system

### Research Extensions

Possible future improvements:
- **Visual Embeddings**: CLIP-style image embeddings
- **Fine-tuning**: Optimize model for specific domains
- **Hybrid Search**: Combine semantic + keyword search
- **Query Expansion**: Improve retrieval with query rewriting
- **Evaluation**: Add metrics for answer quality

---

## 🐛 Troubleshooting

### Model Loading Issues

**Problem**: Out of memory when loading model  
**Solution**: 
```python
# In models/dnotitia_model.py, use quantization:
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
```

### OCR Not Working

**Problem**: `pytesseract` errors  
**Solution**: 
1. Ensure Tesseract is installed: `tesseract --version`
2. Set path explicitly:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
```

### ChromaDB Errors

**Problem**: Collection already exists  
**Solution**: Delete `data/embeddings_db/` directory or use different collection name

### Slow Performance

**Problem**: Generation takes too long  
**Solutions**:
1. Use GPU if available
2. Reduce `max_tokens` in settings
3. Enable model quantization
4. Use smaller embedding model

---

## 📚 References

- **DSPy**: https://github.com/stanfordnlp/dspy
- **LangChain**: https://python.langchain.com/
- **ChromaDB**: https://www.trychroma.com/
- **Sentence-Transformers**: https://www.sbert.net/
- **Streamlit**: https://streamlit.io/
- **DNotitia Model**: https://huggingface.co/dnotitia/DNA-2.0-1.7B
- **SOLAR Model**: https://huggingface.co/upstage/SOLAR-10.7B-v1.0

### Additional Documentation

- **[Documentation Index](docs/README.md)**: Complete documentation hub
- **[Windows Setup](docs/windows/WINDOWS_SETUP.md)**: Windows installation guide
- **[macOS Installation](docs/macos/INSTALLATION.md)**: macOS setup guide
- **[Linux Installation](docs/linux/INSTALLATION.md)**: Linux setup guide
- **[Model Guide](docs/general/MODEL_GUIDE.md)**: Detailed model configuration
- **[Technical Notebook](docs/general/TECHNICAL_EXPLANATION.ipynb)**: Academic deep-dive
- **[Quick Start](QUICKSTART.md)**: Fast-track installation guide

---

## 📄 License

This project is for educational purposes. Please refer to individual library licenses:
- DSPy: Apache 2.0
- Transformers: Apache 2.0
- LangChain: MIT
- ChromaDB: Apache 2.0

---

## 👥 Author

**Hoang Quang Minh**  
FPT Corporation

---

## 🙏 Acknowledgments

- Stanford NLP for DSPy framework
- Hugging Face for model hosting
- Open-source community for libraries and tools

---

**Note**: This project demonstrates a complete production-ready RAG system with multimodal capabilities, suitable for academic presentation and real-world deployment.
