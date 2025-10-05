# Quick Start Guide

## 🚀 Get Started in 5 Minutes

> **🪟 Windows Users**: Use the provided batch files for easier setup!
> - Run `setup_windows.bat` for installation
> - Run `run_windows.bat` to start the app
> - See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for details

### 1. Install Dependencies

**macOS/Linux**:
```bash
# Navigate to project directory
cd ChatBot

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

**Windows** (Manual):
```cmd
# Navigate to project directory
cd ChatBot

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

**Windows** (Easy - Recommended):
```cmd
# Just double-click: setup_windows.bat
# Or run:
setup_windows.bat
```

### 2. Install Tesseract (for OCR)

**macOS**:
```bash
brew install tesseract
```

**Ubuntu/Debian**:
```bash
sudo apt-get install tesseract-ocr
```

**Windows**: 
- Download from https://github.com/UB-Mannheim/tesseract/wiki
- Install `tesseract-ocr-w64-setup-5.x.x.exe`
- Add to PATH: `C:\Program Files\Tesseract-OCR`
- See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for detailed instructions

### 3. Run the Application

**macOS/Linux**:
```bash
streamlit run ui/app.py
```

**Windows** (Manual):
```cmd
venv\Scripts\activate
streamlit run ui/app.py
```

**Windows** (Easy - Recommended):
```cmd
# Just double-click: run_windows.bat
# Or run:
run_windows.bat
```

The app will open at `http://localhost:8501`

---

## 🤖 About Model Loading (First Run)

**What happens on first run:**
1. App tries to load `dnotitia/DNA-2.0-1.7B` model
2. If unavailable (pending approval): **Automatically falls back to SOLAR-10.7B**
3. Model downloads (10-15 minutes, only first time)
4. App starts - ready to use!

**You don't need to wait for model approval!** The app works immediately with the fallback model.

> 💡 See [docs/general/MODEL_GUIDE.md](docs/general/MODEL_GUIDE.md) for detailed model information.
>
> 📚 **Platform-Specific Guides**:
> - Windows: [docs/windows/WINDOWS_SETUP.md](docs/windows/WINDOWS_SETUP.md)
> - macOS: [docs/macos/INSTALLATION.md](docs/macos/INSTALLATION.md)
> - Linux: [docs/linux/INSTALLATION.md](docs/linux/INSTALLATION.md)

---

### 4. Upload Your First Document

1. Go to **📁 Document Management** page
2. Click "Upload documents"
3. Select a PDF, DOCX, or any supported file
4. Click "🚀 Process and Add to Knowledge Base"
5. Wait for processing to complete

### 5. Start Chatting

1. Go to **💬 Chat** page
2. Ask a question about your document
3. Get an AI-generated answer with sources!

---

## 📝 Example Questions to Try

After uploading documents, try these questions:

- "What is this document about?"
- "Summarize the main points"
- "What are the key findings?"
- "Explain [specific topic] from the document"
- "What does the document say about [specific term]?"

---

## ⚙️ Configuration Tips

### Adjust Settings in Sidebar

- **Use RAG**: Enable/disable retrieval-augmented generation
- **Number of Retrieved Documents**: More = more context (but slower)
- **Temperature**: 
  - Low (0.1-0.3): More focused, factual
  - Medium (0.4-0.7): Balanced
  - High (0.8-2.0): More creative, varied
- **Max Tokens**: Response length (512 is good default)

### For Faster Performance

1. **Use GPU**: If available, PyTorch will auto-detect
2. **Reduce Max Tokens**: Lower = faster responses
3. **Fewer Retrieved Docs**: 3-5 is usually enough

### For Better Quality

1. **Increase Retrieved Docs**: Try 7-10 for complex questions
2. **Adjust Temperature**: Lower for factual, higher for creative
3. **Include Chat History**: Better for multi-turn conversations

---

## 🔍 Troubleshooting

### "Model loading failed"
- Check internet connection (first time downloads ~3.5GB)
- Ensure enough disk space (5GB+ recommended)
- Try restarting the application

### "OCR not working"
- Verify Tesseract is installed: `tesseract --version`
- Check PATH environment variable

### "Out of memory"
- Close other applications
- Use CPU mode (automatic if no GPU)
- Process documents in smaller batches

### "ChromaDB errors"
- Delete `data/embeddings_db/` folder and restart
- Check file permissions

---

## 📊 Understanding the Interface

### Chat Page (💬)
- Main conversation interface
- Shows chat history
- Displays sources for answers
- Sidebar settings for customization

### Document Management (📁)
- Upload new documents
- View database statistics
- See all stored documents
- Delete documents or clear database

### Search & Test (🔍)
- Direct vector database search
- Test retrieval without generation
- View similarity scores
- Debug retrieval issues

---

## 💡 Pro Tips

1. **Better Questions**: Be specific and clear
   - ❌ "Tell me about it"
   - ✅ "What are the main conclusions about climate change in this report?"

2. **Document Preparation**: Clean documents work better
   - Use searchable PDFs (not scanned images only)
   - Remove unnecessary pages
   - Ensure good image quality for OCR

3. **Batch Upload**: Upload related documents together
   - They can be cross-referenced
   - Better context for complex questions

4. **Clear Database**: Start fresh for unrelated topics
   - Prevents confusion between documents
   - Better performance with focused knowledge base

5. **Experiment with Settings**: Find what works for your use case
   - Academic: Lower temperature, more retrieved docs
   - Creative: Higher temperature, moderate retrieval
   - Quick facts: RAG on, few docs, low tokens

---

## 🎯 Use Cases

### Academic Research
- Analyze research papers
- Extract key findings
- Compare methodologies
- Summarize literature

### Business Documents
- Analyze reports and presentations
- Extract key metrics
- Summarize meetings
- Compare proposals

### Legal/Compliance
- Review contracts
- Extract clauses
- Compare documents
- Find specific terms

### Technical Documentation
- Query API docs
- Find code examples
- Understand specifications
- Troubleshooting guides

---

## 📞 Need Help?

- Check the main **README.md** for detailed documentation
- Review error messages in the terminal
- Ensure all dependencies are installed
- Try with a simple text file first

---

**Happy Chatting! 🤖**
