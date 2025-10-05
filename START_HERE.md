# 🚀 Start Here!

## Quick Navigation

**New to the project?** Start with one of these:

| I want to... | Go to... |
|--------------|----------|
| 🎯 **Understand the project** | [README.md](README.md) |
| ⚡ **Install quickly** | [QUICKSTART.md](QUICKSTART.md) |
| 📚 **Browse all docs** | [docs/README.md](docs/README.md) |
| 🪟 **Install on Windows** | [docs/windows/WINDOWS_SETUP.md](docs/windows/WINDOWS_SETUP.md) |
| 🍎 **Install on macOS** | [docs/macos/INSTALLATION.md](docs/macos/INSTALLATION.md) |
| 🐧 **Install on Linux** | [docs/linux/INSTALLATION.md](docs/linux/INSTALLATION.md) |
| 🤖 **Learn about models** | [docs/general/MODEL_GUIDE.md](docs/general/MODEL_GUIDE.md) |
| 🎓 **Technical deep-dive** | [docs/general/TECHNICAL_EXPLANATION.ipynb](docs/general/TECHNICAL_EXPLANATION.ipynb) |

---

## 🎯 Quick Start by Platform

### 🪟 Windows
```cmd
cd ChatBot
docs\windows\setup_windows.bat
docs\windows\run_windows.bat
```

### 🍎 macOS
```bash
cd ChatBot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run ui/app.py
```

### 🐧 Linux
```bash
cd ChatBot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run ui/app.py
```

---

## 📁 Project Structure

```
ChatBot/
├── 📄 README.md                # Main project documentation
├── 📄 QUICKSTART.md           # Quick installation guide
├── 📄 START_HERE.md           # This file!
│
├── 📁 docs/                    # All documentation
│   ├── windows/               # Windows guides & scripts
│   ├── macos/                 # macOS guides
│   ├── linux/                 # Linux guides
│   └── general/               # Platform-independent docs
│
├── 📁 app/                     # Application code
│   ├── file_handlers/         # PDF, Image, DOCX, XLSX, TXT handlers
│   ├── chatbot.py            # DSPy RAG pipeline
│   ├── ingestion.py          # Document processing
│   └── vector_db.py          # ChromaDB operations
│
├── 📁 models/                  # AI models
│   └── dnotitia_model.py     # Model wrapper with fallback
│
├── 📁 ui/                      # User interface
│   ├── app.py                # Main Streamlit app
│   └── components.py         # UI components
│
├── 📁 scripts/                 # Utility scripts
│   ├── check_model_status.py
│   └── test_model.py
│
└── 📁 data/                    # Your data (created on first run)
    ├── uploads/               # Uploaded documents
    └── embeddings_db/         # Vector database
```

---

## 🎓 For Students & Professors

### Recommended Reading Order

1. **[START_HERE.md](START_HERE.md)** ← You are here!
2. **[README.md](README.md)** - Complete overview
3. **[docs/general/TECHNICAL_EXPLANATION.ipynb](docs/general/TECHNICAL_EXPLANATION.ipynb)** - Deep dive
4. **[docs/general/IMPLEMENTATION_COMPLETE.md](docs/general/IMPLEMENTATION_COMPLETE.md)** - Implementation details

### Key Features to Highlight

- ✅ Multimodal RAG (text + images)
- ✅ DSPy framework integration
- ✅ Automatic model fallback system
- ✅ Production-ready architecture
- ✅ Comprehensive documentation
- ✅ Cross-platform support

---

## 🤖 About the Models

**Primary**: dnotitia/DNA-2.0-1.7B (requires approval)  
**Fallback**: upstage/SOLAR-10.7B-v1.0 (open access)

**The app automatically uses the fallback model if the primary is unavailable!**

See: [docs/general/MODEL_GUIDE.md](docs/general/MODEL_GUIDE.md)

---

## ✅ Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Tesseract OCR installed (for image text)
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] App runs successfully (`streamlit run ui/app.py`)

---

## 🆘 Need Help?

1. **Check troubleshooting** in your platform guide
2. **Run test scripts**:
   ```bash
   python scripts/check_model_status.py
   python scripts/test_model.py
   ```
3. **Read documentation**: [docs/README.md](docs/README.md)

---

## 💡 Pro Tips

### Windows Users
- Use the provided batch files for easy setup
- PowerShell scripts available as alternative
- Desktop shortcut script included

### macOS Users
- Create an alias for quick launch (see guide)
- Apple Silicon Macs get automatic MPS acceleration
- Homebrew makes installation simple

### Linux Users
- Distribution-specific instructions provided
- CUDA setup guide for GPU acceleration
- Desktop integration examples included

---

**Ready to start? Pick your platform above and follow the guide!** 🚀

**Questions? See [docs/README.md](docs/README.md) for complete documentation index.**
