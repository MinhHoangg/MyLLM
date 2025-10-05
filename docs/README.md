# 📚 Documentation Index

Welcome to the Multimodal RAG Chatbot documentation!

---

## 🚀 Quick Start

**Choose your operating system:**

| Platform | Installation Guide | Quick Scripts |
|----------|-------------------|---------------|
| 🪟 **Windows** | [Windows Setup](windows/WINDOWS_SETUP.md) | `setup_windows.bat`<br>`run_windows.bat` |
| 🍎 **macOS** | [macOS Installation](macos/INSTALLATION.md) | See guide for aliases |
| 🐧 **Linux** | [Linux Installation](linux/INSTALLATION.md) | See guide for aliases |

---

## 📖 Documentation Structure

```
docs/
├── windows/          # Windows-specific documentation
│   ├── WINDOWS_SETUP.md
│   ├── WINDOWS_README.md
│   ├── setup_windows.bat
│   ├── setup_windows.ps1
│   ├── run_windows.bat
│   ├── run_windows.ps1
│   ├── troubleshoot_windows.bat
│   └── create_shortcut.ps1
│
├── macos/            # macOS-specific documentation
│   └── INSTALLATION.md
│
├── linux/            # Linux-specific documentation
│   └── INSTALLATION.md
│
└── general/          # Platform-independent docs
    ├── MODEL_GUIDE.md
    ├── FALLBACK_SUMMARY.md
    ├── IMPLEMENTATION_COMPLETE.md
    ├── QUICK_REFERENCE.md
    └── TECHNICAL_EXPLANATION.ipynb
```

---

## 📋 By Topic

### Installation & Setup
- **Windows**: [windows/WINDOWS_SETUP.md](windows/WINDOWS_SETUP.md)
- **macOS**: [macos/INSTALLATION.md](macos/INSTALLATION.md)
- **Linux**: [linux/INSTALLATION.md](linux/INSTALLATION.md)

### Model Configuration
- **Complete Guide**: [general/MODEL_GUIDE.md](general/MODEL_GUIDE.md)
- **Fallback System**: [general/FALLBACK_SUMMARY.md](general/FALLBACK_SUMMARY.md)
- **Quick Reference**: [general/QUICK_REFERENCE.md](general/QUICK_REFERENCE.md)

### Technical Details
- **Main README**: [../README.md](../README.md)
- **Academic Notebook**: [general/TECHNICAL_EXPLANATION.ipynb](general/TECHNICAL_EXPLANATION.ipynb)
- **Implementation**: [general/IMPLEMENTATION_COMPLETE.md](general/IMPLEMENTATION_COMPLETE.md)

### Troubleshooting
- **Windows**: [windows/WINDOWS_SETUP.md#troubleshooting](windows/WINDOWS_SETUP.md#troubleshooting)
- **macOS**: [macos/INSTALLATION.md#troubleshooting](macos/INSTALLATION.md#troubleshooting)
- **Linux**: [linux/INSTALLATION.md#troubleshooting](linux/INSTALLATION.md#troubleshooting)

---

## 🎯 Common Tasks

### First Time Setup

**Windows:**
```cmd
cd ChatBot
setup_windows.bat
run_windows.bat
```

**macOS/Linux:**
```bash
cd ChatBot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run ui/app.py
```

### Running the App

**Windows:**
```cmd
run_windows.bat
```

**macOS/Linux:**
```bash
source .venv/bin/activate
streamlit run ui/app.py
```

### Testing

**All platforms:**
```bash
python scripts/check_model_status.py
python scripts/test_model.py
```

---

## 🤖 Understanding Models

### Available Models

1. **dnotitia/DNA-2.0-1.7B** (Primary)
   - 🔒 Requires approval
   - 3.5 GB download
   - Fast inference

2. **upstage/SOLAR-10.7B-v1.0** (Fallback)
   - ✅ Open access
   - 21 GB download
   - Better quality

### Automatic Fallback

The app automatically uses SOLAR if dnotitia is unavailable:

```
Try dnotitia → Failed (401) → Use SOLAR ✅
```

**See**: [general/MODEL_GUIDE.md](general/MODEL_GUIDE.md) for complete details

---

## 🆘 Getting Help

### 1. Check Platform Guide
- Windows: [windows/WINDOWS_SETUP.md](windows/WINDOWS_SETUP.md)
- macOS: [macos/INSTALLATION.md](macos/INSTALLATION.md)
- Linux: [linux/INSTALLATION.md](linux/INSTALLATION.md)

### 2. Run Test Scripts
```bash
python scripts/check_model_status.py
python scripts/test_model.py
```

### 3. Check Main README
[../README.md](../README.md) - Complete project documentation

### 4. Review Model Guide
[general/MODEL_GUIDE.md](general/MODEL_GUIDE.md) - Model configuration

---

## 📊 Documentation Stats

| Document | Type | Platform | Lines |
|----------|------|----------|-------|
| WINDOWS_SETUP.md | Installation | Windows | 485 |
| INSTALLATION.md | Installation | macOS | 380 |
| INSTALLATION.md | Installation | Linux | 420 |
| MODEL_GUIDE.md | Reference | All | 485 |
| TECHNICAL_EXPLANATION.ipynb | Tutorial | All | 850+ |
| README.md | Overview | All | 540 |

**Total**: 6+ comprehensive guides, 2,000+ lines of documentation

---

## 🎓 For Professors & Students

### Academic Resources

1. **Technical Deep Dive**
   - [general/TECHNICAL_EXPLANATION.ipynb](general/TECHNICAL_EXPLANATION.ipynb)
   - Jupyter notebook with detailed explanations
   - Code examples and architecture diagrams

2. **Implementation Details**
   - [general/IMPLEMENTATION_COMPLETE.md](general/IMPLEMENTATION_COMPLETE.md)
   - Complete development summary
   - Design decisions and patterns

3. **Main Documentation**
   - [../README.md](../README.md)
   - System architecture
   - Component descriptions

### Presentation Order

Recommended order for explaining to professor:

1. Start with [Quick Reference](general/QUICK_REFERENCE.md)
2. Show [Technical Notebook](general/TECHNICAL_EXPLANATION.ipynb)
3. Explain [Model Fallback](general/FALLBACK_SUMMARY.md)
4. Demonstrate the working app
5. Review [Main README](../README.md) for architecture

---

## 🔄 Recent Updates

### October 5, 2025
- ✅ Restructured documentation by platform
- ✅ Created dedicated folders (windows/macos/linux/general)
- ✅ Added comprehensive platform-specific guides
- ✅ Improved navigation and discoverability
- ✅ Enhanced troubleshooting sections

---

## 💡 Quick Tips

### Windows Users
- Use batch files for easy setup
- PowerShell scripts available as alternative
- Desktop shortcut creation script included

### macOS Users
- Create alias for quick launch
- MPS acceleration automatic on Apple Silicon
- Homebrew makes installation easy

### Linux Users
- Distribution-specific instructions provided
- CUDA setup guide for GPU acceleration
- Desktop integration examples included

---

## 📞 Support Resources

- **Main README**: [../README.md](../README.md)
- **Quick Start**: [../QUICKSTART.md](../QUICKSTART.md)
- **Test Scripts**: `scripts/check_model_status.py`, `scripts/test_model.py`

---

**Choose your platform above to get started!** 🚀
