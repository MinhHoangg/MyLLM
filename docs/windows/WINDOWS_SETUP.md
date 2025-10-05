# Windows Setup Guide for Multimodal RAG Chatbot

## 🪟 Windows Installation Guide

Complete step-by-step instructions for setting up the chatbot on Windows.

---

## ✅ Prerequisites

- Windows 10 or Windows 11
- Python 3.8 or higher
- At least 8GB RAM (16GB recommended)
- 10GB free disk space
- Internet connection

---

## 📥 Step-by-Step Installation

### Step 1: Install Python

1. Download Python from https://www.python.org/downloads/
2. **Important**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```cmd
   python --version
   ```
   Should show Python 3.8 or higher

### Step 2: Install Tesseract OCR (for image text extraction)

**Option A - Using Installer (Recommended)**:
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Download `tesseract-ocr-w64-setup-5.x.x.exe` (64-bit)
3. Run installer
4. **Important**: Note the installation path (default: `C:\Program Files\Tesseract-OCR`)
5. Add Tesseract to PATH:
   - Open System Properties → Environment Variables
   - Edit "Path" under System Variables
   - Add: `C:\Program Files\Tesseract-OCR`
   - Click OK
6. Verify installation:
   ```cmd
   tesseract --version
   ```

**Option B - Using Chocolatey**:
```cmd
choco install tesseract
```

### Step 3: Download or Navigate to Project

Open Command Prompt or PowerShell and navigate to the project:

```cmd
cd "C:\Users\%USERNAME%\OneDrive - FPT Corporation\ChatBot"
```

Or if using PowerShell:
```powershell
cd "$env:USERPROFILE\OneDrive - FPT Corporation\ChatBot"
```

### Step 4: Create Virtual Environment

```cmd
python -m venv venv
```

### Step 5: Activate Virtual Environment

**Command Prompt**:
```cmd
venv\Scripts\activate
```

**PowerShell** (if you get execution policy error):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

You should see `(venv)` at the beginning of your command line.

### Step 6: Install Dependencies

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: This will take 10-15 minutes and download ~4GB of data.

### Step 7: Run the Application

```cmd
streamlit run ui/app.py
```

The application will automatically open in your default browser at `http://localhost:8501`

---

## 🚀 Quick Start Scripts

I've created batch files to make this easier!

### For First-Time Setup

Double-click `setup_windows.bat` or run:
```cmd
setup_windows.bat
```

This will:
- Check Python installation
- Create virtual environment
- Install all dependencies
- Configure Tesseract path

### To Run the Application

Double-click `run_windows.bat` or run:
```cmd
run_windows.bat
```

This will:
- Activate virtual environment
- Start the Streamlit application
- Open your browser automatically

---

## 🔧 GPU Support (Optional)

If you have an NVIDIA GPU with CUDA support:

### Step 1: Install CUDA Toolkit

1. Download from: https://developer.nvidia.com/cuda-downloads
2. Choose Windows → x86_64 → Version 11.8 or 12.x
3. Install the toolkit

### Step 2: Install GPU-enabled PyTorch

```cmd
venv\Scripts\activate
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Verify GPU

```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🐛 Common Windows Issues & Solutions

### Issue 1: "Python is not recognized"

**Solution**:
1. Reinstall Python with "Add to PATH" checked
2. Or manually add Python to PATH:
   - Open System Properties → Environment Variables
   - Edit "Path" under System Variables
   - Add: `C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python3xx`
   - Add: `C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python3xx\Scripts`

### Issue 2: "Cannot activate virtual environment" (PowerShell)

**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again:
```powershell
venv\Scripts\Activate.ps1
```

### Issue 3: "Tesseract not found" error

**Solution**:
Option A - Set environment variable:
```cmd
setx TESSERACT_PATH "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Option B - Edit `app/file_handlers/image_handler.py`:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Issue 4: "Microsoft Visual C++ 14.0 is required"

**Solution**:
Download and install Microsoft C++ Build Tools:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Issue 5: Long file path errors

**Solution**:
Enable long paths in Windows:
1. Open Registry Editor (regedit)
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Set `LongPathsEnabled` to `1`
4. Restart computer

Or use shorter path:
```cmd
cd C:\ChatBot
```

### Issue 6: Slow performance on CPU

**Solutions**:
1. Close other applications
2. Reduce "Max Tokens" in settings (256 instead of 512)
3. Use fewer retrieved documents (3 instead of 5)
4. Consider using GPU (see GPU Support section)

---

## 📊 System Requirements

### Minimum:
- CPU: Intel Core i5 or AMD Ryzen 5
- RAM: 8GB
- Storage: 10GB free space
- OS: Windows 10

### Recommended:
- CPU: Intel Core i7 or AMD Ryzen 7
- RAM: 16GB
- GPU: NVIDIA GTX 1660 or better with 6GB+ VRAM
- Storage: 20GB free space (SSD preferred)
- OS: Windows 11

---

## 🎯 Performance Tips for Windows

1. **Use SSD**: Install on SSD for faster model loading
2. **Close Background Apps**: Free up RAM
3. **Windows Defender**: Add project folder to exclusions
4. **Power Plan**: Use "High Performance" power plan
5. **Virtual Memory**: Increase page file size if needed

### Adjust Virtual Memory:
1. System Properties → Advanced → Performance Settings
2. Advanced → Virtual Memory → Change
3. Uncheck "Automatically manage"
4. Set Initial size: 16000 MB
5. Set Maximum size: 32000 MB

---

## 🔄 Updates and Maintenance

### Update Dependencies:
```cmd
venv\Scripts\activate
pip install --upgrade -r requirements.txt
```

### Clear Cache:
```cmd
rd /s /q data\embeddings_db
rd /s /q __pycache__
```

### Reinstall from Scratch:
```cmd
rd /s /q venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📝 Environment Variables (Optional)

Create a `.env` file in the project root for configuration:

```env
# Model settings
MODEL_NAME=dnotitia/DNA-2.0-1.7B
DEVICE=cuda
MAX_LENGTH=2048

# Database settings
DB_PATH=data/embeddings_db
COLLECTION_NAME=multimodal_docs

# Tesseract path (if not in PATH)
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe

# UI settings
STREAMLIT_PORT=8501
```

---

## 🆘 Getting Help

### Check Logs:
The terminal window shows error messages and logs.

### Common Commands:

**Check Python version**:
```cmd
python --version
```

**Check installed packages**:
```cmd
pip list
```

**Test imports**:
```cmd
python -c "import torch; import transformers; import streamlit"
```

**Check disk space**:
```cmd
wmic logicaldisk get size,freespace,caption
```

---

## 📚 Additional Resources

- Python for Windows: https://www.python.org/downloads/windows/
- Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/
- CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki

---

## ✅ Verification Checklist

After installation, verify everything works:

- [ ] Python installed and in PATH
- [ ] Virtual environment created
- [ ] All dependencies installed
- [ ] Tesseract OCR working
- [ ] Streamlit application starts
- [ ] Can upload a document
- [ ] Can ask questions and get answers

---

## 🎓 For Students

If you're presenting this at university:

1. **Demo Preparation**:
   - Upload sample documents before presentation
   - Prepare example questions
   - Test everything the day before

2. **Offline Mode**:
   - Download model beforehand (first run)
   - No internet needed after setup

3. **Presentation Tips**:
   - Use "High Performance" power mode
   - Close unnecessary applications
   - Have backup slides in case of issues

---

**Need help?** Check the main README.md or QUICKSTART.md for more information.

**Happy Chatting! 🤖**
