# Installation Guide - All Platforms

Complete installation instructions for Windows, macOS, and Linux.

> **📚 New Documentation Structure**: Platform-specific guides are now in the `docs/` folder for better organization.

---

## 🎯 Choose Your Platform

**Select your operating system for detailed installation guide:**

| Platform | Quick Start | Full Guide |
|----------|-------------|------------|
| 🪟 **Windows** | Run `docs/windows/setup_windows.bat` | [Windows Setup Guide](docs/windows/WINDOWS_SETUP.md) |
| 🍎 **macOS** | [Quick Steps](#macos-quick) | [macOS Installation Guide](docs/macos/INSTALLATION.md) |
| 🐧 **Linux** | [Quick Steps](#linux-quick) | [Linux Installation Guide](docs/linux/INSTALLATION.md) |

---

## Choose Your Platform

- [Windows](#windows) - Most users start here
- [macOS](#macos) - Mac users  
- [Linux](#linux) - Ubuntu, Debian, etc.

**Or jump to detailed platform-specific guides:**
- [📘 Windows Full Guide](docs/windows/WINDOWS_SETUP.md)
- [📗 macOS Full Guide](docs/macos/INSTALLATION.md)
- [📕 Linux Full Guide](docs/linux/INSTALLATION.md)

---

## Windows <a name="windows"></a>

### 🎯 Easiest Method (Recommended)

**Step 1**: Install Python
- Download: https://www.python.org/downloads/
- ✅ **IMPORTANT**: Check "Add Python to PATH"
- Restart computer

**Step 2**: Install Tesseract OCR (optional)
- Download: https://github.com/UB-Mannheim/tesseract/wiki
- Run installer: `tesseract-ocr-w64-setup-5.x.x.exe`

**Step 3**: Run Setup
```cmd
Double-click: setup_windows.bat
```

**Step 4**: Start Application
```cmd
Double-click: run_windows.bat
```

### 📚 Detailed Windows Guide
See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) or [WINDOWS_README.md](WINDOWS_README.md)

### 🔧 Windows Scripts Available
- `setup_windows.bat` - Automated setup (Command Prompt)
- `run_windows.bat` - Start application (Command Prompt)
- `setup_windows.ps1` - Automated setup (PowerShell)
- `run_windows.ps1` - Start application (PowerShell)
- `troubleshoot_windows.bat` - Diagnostic tool
- `create_shortcut.ps1` - Create desktop shortcut

---

## macOS <a name="macos"></a>

### Prerequisites
- macOS 10.14 (Mojave) or higher
- Python 3.8+ (comes with macOS)
- Homebrew (recommended)

### Installation

**Step 1**: Install Homebrew (if not installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Step 2**: Install Dependencies
```bash
# Install Tesseract OCR
brew install tesseract

# Install Python (if needed)
brew install python@3.11
```

**Step 3**: Setup Project
```bash
# Navigate to project
cd ~/Library/CloudStorage/OneDrive-FPTCorporation/ChatBot

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

**Step 4**: Run Application
```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Start application
streamlit run ui/app.py
```

### Create Alias (Optional)
Add to `~/.zshrc` or `~/.bash_profile`:
```bash
alias chatbot='cd ~/Library/CloudStorage/OneDrive-FPTCorporation/ChatBot && source venv/bin/activate && streamlit run ui/app.py'
```

Then just type `chatbot` to start!

---

## Linux <a name="linux"></a>

### Ubuntu/Debian

**Step 1**: Update System
```bash
sudo apt update
sudo apt upgrade
```

**Step 2**: Install Dependencies
```bash
# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Install Tesseract OCR
sudo apt install tesseract-ocr

# Install system libraries
sudo apt install build-essential libssl-dev libffi-dev python3-dev
```

**Step 3**: Setup Project
```bash
# Navigate to project
cd ~/ChatBot

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

**Step 4**: Run Application
```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Start application
streamlit run ui/app.py
```

### Fedora/RHEL/CentOS

**Install Dependencies**:
```bash
# Install Python
sudo dnf install python3 python3-pip python3-devel

# Install Tesseract
sudo dnf install tesseract

# Install build tools
sudo dnf groupinstall "Development Tools"
```

Then follow Steps 3-4 from Ubuntu instructions above.

### Arch Linux

**Install Dependencies**:
```bash
# Install Python
sudo pacman -S python python-pip

# Install Tesseract
sudo pacman -S tesseract

# Install build tools
sudo pacman -S base-devel
```

Then follow Steps 3-4 from Ubuntu instructions above.

---

## GPU Support (All Platforms)

### NVIDIA GPU with CUDA

**Check GPU**:
```bash
nvidia-smi
```

**Install CUDA Toolkit**:
- Download from: https://developer.nvidia.com/cuda-downloads
- Choose your platform and follow instructions

**Install PyTorch with CUDA**:
```bash
# Activate virtual environment first
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate    # Windows

# Uninstall CPU version
pip uninstall torch torchvision torchaudio

# Install GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Verification

After installation, verify everything works:

### Test Python
```bash
python --version
# Should show: Python 3.8.x or higher
```

### Test Tesseract
```bash
tesseract --version
# Should show: tesseract 5.x.x
```

### Test Virtual Environment
```bash
# Activate venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Check Python in venv
which python  # macOS/Linux
where python  # Windows
```

### Test Dependencies
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers OK')"
python -c "import streamlit; print('Streamlit OK')"
python -c "import chromadb; print('ChromaDB OK')"
python -c "import langchain; print('LangChain OK')"
```

### Test Application
```bash
streamlit run ui/app.py
```

Browser should open at `http://localhost:8501`

---

## Troubleshooting

### Permission Errors (Linux/macOS)
```bash
# If you get permission errors, use --user flag
pip install --user -r requirements.txt

# Or fix ownership
sudo chown -R $USER:$USER ~/ChatBot
```

### Port Already in Use
```bash
# Use different port
streamlit run ui/app.py --server.port 8502
```

### Out of Memory
- Close other applications
- Reduce batch size in settings
- Use CPU if GPU memory insufficient
- Consider upgrading RAM

### Module Not Found
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

---

## Uninstallation

### Remove Project
```bash
# Just delete the folder
rm -rf ~/ChatBot  # macOS/Linux
# Or delete folder in File Explorer (Windows)
```

### Remove Dependencies (Optional)
```bash
# Remove virtual environment
rm -rf venv

# Uninstall Tesseract
brew uninstall tesseract  # macOS
sudo apt remove tesseract-ocr  # Ubuntu/Debian
```

---

## System Requirements

### Minimum
| Component | Requirement |
|-----------|-------------|
| OS | Windows 10, macOS 10.14, Ubuntu 20.04 |
| CPU | Intel Core i5 / AMD Ryzen 5 |
| RAM | 8GB |
| Storage | 10GB free |
| Python | 3.8+ |

### Recommended
| Component | Requirement |
|-----------|-------------|
| OS | Windows 11, macOS 12+, Ubuntu 22.04 |
| CPU | Intel Core i7 / AMD Ryzen 7 |
| RAM | 16GB |
| Storage | 20GB free (SSD) |
| GPU | NVIDIA with 6GB+ VRAM |
| Python | 3.10+ |

---

## First Run

The first time you run the application:
1. AI model will download (~3.5GB) - **takes 5-10 minutes**
2. Embedding model will download (~90MB)
3. ChromaDB will initialize
4. Browser will open automatically

**Subsequent runs**: Start in ~30 seconds!

---

## Getting Help

- **Windows**: See [WINDOWS_SETUP.md](WINDOWS_SETUP.md)
- **Quick Start**: See [QUICKSTART.md](QUICKSTART.md)
- **Full Guide**: See [README.md](README.md)
- **Academic**: See [TECHNICAL_EXPLANATION.ipynb](TECHNICAL_EXPLANATION.ipynb)

---

## Quick Command Reference

| Task | macOS/Linux | Windows |
|------|-------------|---------|
| Activate venv | `source venv/bin/activate` | `venv\Scripts\activate` |
| Deactivate venv | `deactivate` | `deactivate` |
| Run app | `streamlit run ui/app.py` | `streamlit run ui/app.py` |
| Install deps | `pip install -r requirements.txt` | `pip install -r requirements.txt` |
| Check Python | `python --version` | `python --version` |
| Check Tesseract | `tesseract --version` | `tesseract --version` |

---

**Installation complete! Ready to chat with your documents! 🤖**
