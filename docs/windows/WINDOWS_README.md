# 🪟 WINDOWS USERS - START HERE!

## Quick Setup for Windows

This project includes **easy-to-use batch files** for Windows users. No complex commands needed!

---

## ⚡ Super Quick Start (3 Steps)

### 1️⃣ Install Python
- Download from: https://www.python.org/downloads/
- **IMPORTANT**: Check "Add Python to PATH" during installation
- Restart your computer after installation

### 2️⃣ Install Tesseract OCR (Optional but recommended)
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Download `tesseract-ocr-w64-setup-5.x.x.exe`
- Run installer
- Keep default installation path

### 3️⃣ Run Setup
**Just double-click this file**: `setup_windows.bat`

Wait for installation to complete (10-15 minutes).

---

## 🚀 Running the Application

**Just double-click this file**: `run_windows.bat`

The chatbot will open in your browser!

---

## 📁 Important Files for Windows Users

| File | Purpose | When to Use |
|------|---------|-------------|
| `setup_windows.bat` | Install everything | **First time only** |
| `run_windows.bat` | Start the chatbot | **Every time you want to use it** |
| `troubleshoot_windows.bat` | Check for problems | If something isn't working |
| `WINDOWS_SETUP.md` | Detailed guide | For more information |
| `setup_windows.ps1` | PowerShell version | Alternative to .bat file |
| `run_windows.ps1` | PowerShell version | Alternative to .bat file |

---

## 🎯 First Time Setup Checklist

- [ ] Install Python 3.8+ (with "Add to PATH")
- [ ] Install Tesseract OCR (optional)
- [ ] Double-click `setup_windows.bat`
- [ ] Wait for installation (10-15 minutes)
- [ ] See "Setup Complete!" message
- [ ] Double-click `run_windows.bat`
- [ ] Browser opens with chatbot interface
- [ ] Upload a document and start chatting!

---

## 🆘 Troubleshooting

### Problem: "Python is not recognized"

**Solution**:
1. Reinstall Python
2. **CHECK** "Add Python to PATH" option
3. Restart computer
4. Try again

### Problem: "Cannot activate virtual environment"

**Solution**:
Right-click PowerShell → Run as Administrator:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Problem: Batch file closes immediately

**Solution**:
1. Right-click `setup_windows.bat`
2. Choose "Edit" (opens in Notepad)
3. Check if there are any error messages
4. Or run: `troubleshoot_windows.bat` to diagnose

### Problem: "Tesseract not found"

**Solution**:
Edit `app/file_handlers/image_handler.py` and add this line after imports:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Problem: Application is slow

**Solutions**:
- Close other applications
- In chatbot settings: reduce "Max Tokens" to 256
- In chatbot settings: reduce "Retrieved Documents" to 3
- Consider upgrading RAM or using a GPU

---

## 💻 System Requirements

### Minimum (Works but slow):
- Windows 10
- Intel Core i5 / AMD Ryzen 5
- 8GB RAM
- 10GB free disk space

### Recommended (Good performance):
- Windows 11
- Intel Core i7 / AMD Ryzen 7  
- 16GB RAM
- 20GB free disk space (SSD)
- NVIDIA GPU with 6GB+ VRAM (optional)

---

## 🎓 For Students

### Quick Demo Setup:
1. Run `setup_windows.bat` the night before your presentation
2. Test with a sample document
3. Prepare 2-3 questions to ask
4. On presentation day, just run `run_windows.bat`

### Offline Use:
✅ **Good news!** After first run, the chatbot works offline!
- The AI model is downloaded once (first time)
- No internet needed after that
- Perfect for presentations!

---

## 📚 Need More Help?

1. **Detailed Instructions**: Read `WINDOWS_SETUP.md`
2. **General Info**: Read `README.md`
3. **Quick Reference**: Read `QUICKSTART.md`
4. **Academic Details**: Open `TECHNICAL_EXPLANATION.ipynb`

---

## 🎉 What You Get

After setup, you'll have:
- ✅ AI Chatbot that answers questions about YOUR documents
- ✅ Support for PDF, Word, Excel, Images, Text files
- ✅ Extracts text from images (OCR)
- ✅ Shows sources for answers
- ✅ Easy-to-use web interface
- ✅ All running on your computer (private and secure)

---

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Install Python | 5 minutes |
| Install Tesseract | 3 minutes |
| Run setup_windows.bat | 10-15 minutes |
| First run (download model) | 5-10 minutes |
| Every subsequent run | 30 seconds |
| Upload & process document | 10-60 seconds |
| Ask question & get answer | 2-10 seconds |

---

## 🔐 Privacy Note

✅ **Everything runs on YOUR computer**
- No data sent to cloud
- No accounts needed
- Your documents stay private
- No internet required after setup

---

## 📞 Getting Started Command

Open Command Prompt (cmd) and type:
```cmd
cd "%USERPROFILE%\OneDrive - FPT Corporation\ChatBot"
setup_windows.bat
```

Or just navigate to the folder in File Explorer and double-click `setup_windows.bat`!

---

**Happy Chatting! 🤖**

*Made easy for Windows by including automated setup scripts!*
