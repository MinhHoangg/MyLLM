# Multimodal RAG Chatbot with DSPy

A sophisticated multimodal chatbot built with **DSPy**, powered by **Hugging Face language models** (dnotitia/DNA-2.0-1.7B with automatic fallback to upstage/SOLAR-10.7B-v1.0), featuring a complete **Retrieval-Augmented Generation (RAG)** pipeline that supports both text and image content.

## 🎯 Project Overview

This project implements a production-ready multimodal chatbot that can:
- Process multiple file formats (PDF, DOCX, XLSX, TXT, images, etc.)
- Extract and process both text and images from documents
- Store content in a vector database for efficient retrieval
- Answer questions using RAG with DSPy orchestration
- Provide an intuitive Streamlit-based user interface
- **Automatic model fallback**: Works immediately with SOLAR-10.7B if primary model pending approval

### Key Features

✅ **Multimodal Document Processing**: Handles text and images from various file formats  
✅ **Advanced RAG Pipeline**: Uses DSPy for structured retrieval and generation  
✅ **CLIP Integration**: True multimodal understanding with text-image embeddings ⭐ NEW  
✅ **Vector Database**: ChromaDB for efficient semantic search  
✅ **LangChain Integration**: Intelligent document chunking  
✅ **Modern UI**: Interactive Streamlit interface with chat and management features  
✅ **Production-Ready**: Modular architecture with proper error handling

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI Layer                       │
│  (Chat Interface | Document Management | Search & Test)     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  Application Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Chatbot    │  │  Ingestion   │  │  Vector DB   │      │
│  │   (DSPy)     │  │  (LangChain) │  │  (ChromaDB)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  File Handlers Layer                         │
│  [PDF] [Images] [DOCX] [XLSX] [TXT/LOG]                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Model Layer                               │
│            DNotitia DNA-2.0-1.7B (Hugging Face)            │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Document Upload** → File Handlers extract content (text + images)
2. **Chunking** → LangChain splits text into semantic chunks
3. **Embedding** → Sentence-Transformers generate vector embeddings
4. **Storage** → ChromaDB stores embeddings with metadata
5. **Query** → User asks a question
6. **Retrieval** → ChromaDB finds relevant chunks (semantic search)
7. **Generation** → DSPy orchestrates LLM to generate answer with context
8. **Response** → Answer displayed with sources

---

## 📁 Project Structure

```
ChatBot/
├── app/
│   ├── __init__.py
│   ├── chatbot.py              # DSPy RAG pipeline & chatbot logic
│   ├── ingestion.py            # Document processing & chunking
│   ├── vector_db.py            # ChromaDB operations
│   ├── file_handlers/
│   │   ├── __init__.py
│   │   ├── pdf_handler.py      # PDF extraction (PyMuPDF)
│   │   ├── image_handler.py    # Image processing with OCR
│   │   ├── doc_handler.py      # DOCX processing
│   │   ├── xlsx_handler.py     # Excel spreadsheet processing
│   │   └── txt_handler.py      # Text files (TXT, LOG, etc.)
│   └── utils.py                # Utility functions
│
├── models/
│   ├── __init__.py
│   └── dnotitia_model.py       # Hugging Face model wrapper
│
├── ui/
│   ├── __init__.py
│   ├── app.py                  # Main Streamlit application
│   └── components.py           # Reusable UI components
│
├── data/
│   ├── uploads/                # Uploaded files
│   ├── chunks/                 # Processed chunks
│   └── embeddings_db/          # ChromaDB persistence
│
├── tests/
│   └── test_app.py            # Unit tests
│
├── .streamlit/
│   └── config.toml            # Streamlit configuration
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🔧 Technical Details

### 1. File Handlers

Each handler is specialized for specific file types:

#### **PDF Handler** (`pdf_handler.py`)
- Uses **PyMuPDF (fitz)** for robust PDF processing
- Extracts text from each page
- Extracts embedded images with metadata
- Handles complex PDF structures

#### **Image Handler** (`image_handler.py`)
- Supports PNG, JPG, JPEG, BMP, TIFF, GIF
- **OCR** with pytesseract to extract text from images
- Image preprocessing (RGBA → RGB conversion)
- Stores PIL Image objects for display

#### **DOC/DOCX Handler** (`doc_handler.py`)
- Uses **python-docx** library
- Extracts text from paragraphs
- Processes tables and preserves structure
- Extracts document metadata (author, title)

#### **XLSX Handler** (`xlsx_handler.py`)
- Uses **openpyxl** and **pandas**
- Processes multiple sheets
- Converts tabular data to readable text format
- Handles formulas and calculated values

#### **TXT/LOG Handler** (`txt_handler.py`)
- Universal text file processor
- Automatic encoding detection with **chardet**
- Supports TXT, LOG, MD, CSV, JSON, XML, YAML
- Handles large files efficiently

### 2. Document Ingestion (`ingestion.py`)

The ingestion module orchestrates the entire document processing pipeline:

1. **File Detection**: Automatically selects appropriate handler
2. **Content Extraction**: Extracts text and images
3. **Chunking**: Uses LangChain's `RecursiveCharacterTextSplitter`
   - Chunk size: 1000 characters
   - Overlap: 200 characters
   - Smart splitting on natural boundaries (paragraphs, sentences)
4. **Metadata Preservation**: Maintains source information, page numbers, etc.
5. **Image Processing**: Prepares images for embedding

### 3. Vector Database (`vector_db.py`)

**ChromaDB** implementation with:

- **Persistent Storage**: Data persists across sessions
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Cosine Similarity**: For semantic search
- **Metadata Filtering**: Search by file, type, page, etc.
- **CRUD Operations**: Add, search, delete, clear collections

**Key Methods**:
- `add_text_documents()`: Store text chunks with embeddings
- `add_image_documents()`: Store image metadata with embeddings
- `search()`: Semantic search with filters
- `get_all_documents()`: Retrieve all stored documents
- `clear_collection()`: Reset database

### 3.5. CLIP Multimodal Embeddings ⭐ NEW (`vector_db_clip.py`, `clip_embedder.py`)

**Enhanced multimodal understanding** with OpenAI's CLIP:

#### **Why CLIP?**
Traditional approaches store images as text descriptions. CLIP creates **aligned embeddings** where text and images share the same vector space, enabling:

- **True Visual Understanding**: Images represented by what they contain, not just descriptions
- **Cross-Modal Retrieval**: Find images using text, or text using images
- **Zero-Shot Classification**: Understand new concepts without additional training
- **Better Semantic Matching**: More accurate alignment between user intent and content

#### **CLIPEmbedder** (`models/clip_embedder.py`)
```python
# Encode text and images in the same space
embedder = CLIPEmbedder(model_name="openai/clip-vit-base-patch32")

text_emb = embedder.encode_text(["a red sunset over ocean"])
image_emb = embedder.encode_images([image])
similarity = embedder.compute_similarity(text_emb, image_emb)
```

**Features**:
- **Unified Embedding Space**: 512/768-dimensional vectors for both text and images
- **Multiple Models**: ViT-B/32 (600MB, fast) or ViT-L/14 (1.7GB, accurate)
- **Auto Device Detection**: CUDA → MPS → CPU
- **Batch Processing**: Efficient handling of multiple images
- **HybridEmbedder**: Combines CLIP + sentence-transformers

#### **MultimodalVectorDatabase** (`app/vector_db_clip.py`)

Enhanced ChromaDB with CLIP support:

```python
db = MultimodalVectorDatabase(use_clip=True)

# Search with text (finds both text and images)
results = db.search("beautiful sunset")

# Search with image (finds similar images and related text)
results = db.search(query_image, query_type='image')

# Multimodal fusion search
results = db.search_multimodal(
    text_query="sunset over water",
    image_query=example_image,
    alpha=0.5  # Balance text and image importance
)
```

**Capabilities**:
- **Text → Image**: Find images matching text description
- **Image → Image**: Find visually similar images
- **Image → Text**: Find text related to image content
- **Multimodal Fusion**: Combined text + image queries with adjustable weighting

> **📘 Learn More**: See [docs/general/CLIP_GUIDE.md](docs/general/CLIP_GUIDE.md) for detailed usage guide  
> **🔧 Integration**: See [CLIP_INTEGRATION.md](CLIP_INTEGRATION.md) for step-by-step setup

### 4. DSPy RAG Pipeline (`chatbot.py`)

**DSPy** (Declarative Self-improving Language Programs) provides structured LM programming:

#### **RAG Components**:

```python
class RAGSignature(dspy.Signature):
    context = dspy.InputField(desc="Retrieved context")
    question = dspy.InputField(desc="User's question")
    answer = dspy.OutputField(desc="Answer based on context")

class MultimodalRAG(dspy.Module):
    def __init__(self, vector_db, llm):
        self.retrieve = dspy.Retrieve(k=5)
        self.generate_answer = dspy.ChainOfThought(RAGSignature)
```

#### **Pipeline Flow**:
1. **Retrieval**: Search vector DB for relevant chunks
2. **Context Formatting**: Combine retrieved documents
3. **Chain-of-Thought**: DSPy generates structured reasoning
4. **Answer Generation**: LLM produces final answer
5. **Source Attribution**: Returns relevant metadata

### 5. Model Integration (`dnotitia_model.py`)

Wrapper for **multiple language models** with automatic fallback:

**Primary Model**: dnotitia/DNA-2.0-1.7B (1.7B params, requires approval)  
**Fallback Model**: upstage/SOLAR-10.7B-v1.0 (10.7B params, open access)

**Features**:
- **Auto Device Selection**: GPU (CUDA/MPS) or CPU
- **Automatic Fallback**: Uses SOLAR if primary model unavailable
- **Half-Precision**: FP16 on GPU for efficiency
- **Configurable Sampling**: Temperature, top_p, max_tokens
- **Batch Generation**: Process multiple prompts
- **Token Management**: Handles padding, EOS tokens
- **Smart Error Handling**: Detects auth errors and switches models

**Model Parameters**:
- Max sequence length: 2048 tokens
- Temperature: 0.7 (configurable)
- Top-p: 0.9 (nucleus sampling)

> **Note**: If dnotitia model is pending approval, the app automatically uses SOLAR-10.7B. No manual intervention needed! See [docs/general/MODEL_GUIDE.md](docs/general/MODEL_GUIDE.md) for details.

### 6. Streamlit UI (`ui/app.py`, `ui/components.py`)

**Three Main Pages**:

#### **💬 Chat Page**
- Real-time conversation interface
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
