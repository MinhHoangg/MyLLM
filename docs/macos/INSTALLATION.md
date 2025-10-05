# macOS Installation Guide

Complete installation guide for macOS users.

---

## 🖥️ System Requirements

### Minimum
- **macOS**: 10.14 (Mojave) or higher
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **Python**: 3.8 or higher

### Recommended
- **macOS**: 12 (Monterey) or higher
- **RAM**: 16 GB (32 GB for SOLAR model)
- **Storage**: 30 GB free (SSD preferred)
- **Python**: 3.10 or higher
- **Chip**: Apple Silicon (M1/M2/M3) for better performance

---

## 📋 Prerequisites

### 1. Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Python 3.10+

```bash
# Check if Python is installed
python3 --version

# If not installed or version < 3.8
brew install python@3.11
```

### 3. Install Tesseract OCR (for image text recognition)

```bash
brew install tesseract
```

Verify installation:
```bash
tesseract --version
```

---

## 🚀 Quick Installation (5 Minutes)

### Step 1: Navigate to Project
```bash
cd ~/Library/CloudStorage/OneDrive-FPTCorporation/ChatBot
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv .venv
```

### Step 3: Activate Virtual Environment
```bash
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

### Step 4: Upgrade pip
```bash
pip install --upgrade pip
```

### Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

This will take 5-10 minutes depending on your internet speed.

### Step 6: Run the Application
```bash
streamlit run ui/app.py
```

The browser will automatically open to `http://localhost:8501`

---

## 🎯 First Run Experience

### What Happens on First Launch:

1. **Embedding Model Download** (~90 MB, 1-2 minutes)
   ```
   Downloading: sentence-transformers/all-MiniLM-L6-v2
   ```

2. **Language Model Download** (automatic fallback)
   ```
   Trying: dnotitia/DNA-2.0-1.7B
   → Not approved, falling back...
   Downloading: upstage/SOLAR-10.7B-v1.0 (~21 GB, 10-15 minutes)
   ```

3. **Database Initialization** (few seconds)
   ```
   Creating ChromaDB collection
   ```

4. **App Ready!** 🎉

### Performance on Apple Silicon (M1/M2/M3)

✅ **Automatic MPS (Metal Performance Shaders) detection**
- Faster inference than Intel Macs
- Better memory efficiency
- Cool & quiet operation

---

## 📝 Daily Usage

### Starting the App

**Option 1: Manual activation**
```bash
cd ~/Library/CloudStorage/OneDrive-FPTCorporation/ChatBot
source .venv/bin/activate
streamlit run ui/app.py
```

**Option 2: Create an alias (recommended)**

Add to `~/.zshrc`:
```bash
alias chatbot='cd ~/Library/CloudStorage/OneDrive-FPTCorporation/ChatBot && source .venv/bin/activate && streamlit run ui/app.py'
```

Apply changes:
```bash
source ~/.zshrc
```

Now just type:
```bash
chatbot
```

**Option 3: Create a launcher script**

Create `~/chatbot.sh`:
```bash
#!/bin/bash
cd ~/Library/CloudStorage/OneDrive-FPTCorporation/ChatBot
source .venv/bin/activate
streamlit run ui/app.py
```

Make it executable:
```bash
chmod +x ~/chatbot.sh
```

Run with:
```bash
~/chatbot.sh
```

---

## 🔧 Troubleshooting

### Issue: "command not found: streamlit"

**Cause**: Virtual environment not activated

**Solution**:
```bash
source .venv/bin/activate
streamlit run ui/app.py
```

### Issue: "ModuleNotFoundError: No module named 'langchain'"

**Cause**: Using system Python instead of venv

**Solution**:
```bash
# Make sure venv is activated (you should see (.venv))
source .venv/bin/activate

# Verify correct Python
which python
# Should show: /path/to/ChatBot/.venv/bin/python

# Reinstall if needed
pip install -r requirements.txt
```

### Issue: "Port 8501 is already in use"

**Cause**: Another instance is running

**Solutions**:

**Option 1: Use different port**
```bash
streamlit run ui/app.py --server.port 8502
```

**Option 2: Kill existing process**
```bash
# Find the process
lsof -i :8501

# Kill it (replace PID with actual process ID)
kill -9 <PID>
```

### Issue: Slow performance on Intel Mac

**Solutions**:
1. Close other applications
2. Reduce max tokens in settings
3. Use CPU mode (automatic fallback)
4. Consider upgrading to Apple Silicon Mac

### Issue: "zsh: permission denied"

**Solution**:
```bash
chmod +x .venv/bin/activate
source .venv/bin/activate
```

---

## 🎮 GPU Acceleration (Apple Silicon)

### Check if MPS is available:
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

Should return: `MPS available: True` on M1/M2/M3 Macs

### Verify model is using MPS:
In the app sidebar, you should see:
```
Device: mps
```

---

## 📦 Updating the Application

### Update Python packages:
```bash
source .venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Update to latest code:
```bash
git pull origin main  # If using git
```

---

## 🗑️ Uninstallation

### Remove application:
```bash
rm -rf ~/Library/CloudStorage/OneDrive-FPTCorporation/ChatBot
```

### Remove Homebrew packages (optional):
```bash
brew uninstall tesseract python@3.11
```

### Remove cached models (optional):
```bash
rm -rf ~/.cache/huggingface/
```

---

## 💡 Tips & Tricks

### 1. Speed up model loading
Pre-download models:
```bash
python scripts/test_model.py
```

### 2. Monitor resource usage
```bash
# Open Activity Monitor
# Search for "python" or "streamlit"
# Check CPU/Memory usage
```

### 3. Better terminal experience
Use iTerm2 for split panes:
```bash
brew install --cask iterm2
```

### 4. Keyboard shortcut
Create macOS app launcher:
- Open Automator
- New → Application
- Add "Run Shell Script"
- Paste: `~/chatbot.sh`
- Save as "Chatbot.app"
- Add to Dock

---

## 🔐 Security Notes

### Models are cached at:
```
~/.cache/huggingface/hub/
```

### Database stored at:
```
~/Library/CloudStorage/OneDrive-FPTCorporation/ChatBot/data/embeddings_db/
```

### Your documents:
```
~/Library/CloudStorage/OneDrive-FPTCorporation/ChatBot/data/uploads/
```

**All data stays local on your Mac** - nothing is sent to external servers except model downloads from Hugging Face.

---

## 📊 Disk Space Management

### Check space usage:
```bash
du -sh ~/Library/CloudStorage/OneDrive-FPTCorporation/ChatBot
du -sh ~/.cache/huggingface/
```

### Clean up:
```bash
# Remove old uploaded files
rm -rf data/uploads/*

# Clear model cache (will re-download on next run)
rm -rf ~/.cache/huggingface/hub/models--upstage--SOLAR-10.7B-v1.0
```

---

## 🆘 Getting Help

### Check logs:
```bash
# Streamlit logs are shown in terminal
# Look for errors in red text
```

### Test scripts:
```bash
# Check model status
python scripts/check_model_status.py

# Test model loading
python scripts/test_model.py
```

### Documentation:
- [Main README](../../README.md)
- [Model Guide](../general/MODEL_GUIDE.md)
- [Quick Start](../general/QUICKSTART.md)

---

## ✅ Success Checklist

- [ ] Homebrew installed
- [ ] Python 3.8+ installed
- [ ] Tesseract OCR installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] App runs without errors
- [ ] Browser opens to localhost:8501
- [ ] Can upload documents
- [ ] Can chat with documents

**Once all checked, you're ready to use the chatbot!** 🎉

---

**Last Updated**: October 5, 2025  
**macOS Version**: Tested on Monterey, Ventura, Sonoma
