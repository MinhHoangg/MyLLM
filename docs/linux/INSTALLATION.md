# Linux Installation Guide

Complete installation guide for Linux users (Ubuntu, Debian, Fedora, Arch).

---

## 🐧 System Requirements

### Minimum
- **Distribution**: Ubuntu 20.04, Debian 10, Fedora 33, Arch (current)
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **Python**: 3.8 or higher

### Recommended
- **Distribution**: Ubuntu 22.04+, Debian 11+, Fedora 35+
- **RAM**: 16 GB (32 GB for SOLAR model)
- **Storage**: 30 GB free (SSD preferred)
- **Python**: 3.10 or higher
- **GPU**: NVIDIA with CUDA support (optional)

---

## 📋 Installation by Distribution

### Ubuntu / Debian

#### Step 1: Update System
```bash
sudo apt update
sudo apt upgrade -y
```

#### Step 2: Install Dependencies
```bash
# Python and development tools
sudo apt install -y python3 python3-pip python3-venv python3-dev

# Tesseract OCR
sudo apt install -y tesseract-ocr

# Build essentials
sudo apt install -y build-essential libssl-dev libffi-dev

# Optional: Git
sudo apt install -y git
```

#### Step 3: Verify Installation
```bash
python3 --version  # Should be 3.8+
tesseract --version  # Should show version
```

---

### Fedora / RHEL / CentOS

#### Step 1: Update System
```bash
sudo dnf update -y
```

#### Step 2: Install Dependencies
```bash
# Python and development tools
sudo dnf install -y python3 python3-pip python3-devel

# Tesseract OCR
sudo dnf install -y tesseract

# Build tools
sudo dnf groupinstall -y "Development Tools"

# Optional: Git
sudo dnf install -y git
```

#### Step 3: Verify Installation
```bash
python3 --version
tesseract --version
```

---

### Arch Linux

#### Step 1: Update System
```bash
sudo pacman -Syu
```

#### Step 2: Install Dependencies
```bash
# Python and tools
sudo pacman -S python python-pip

# Tesseract OCR
sudo pacman -S tesseract tesseract-data-eng

# Build tools
sudo pacman -S base-devel

# Optional: Git
sudo pacman -S git
```

#### Step 3: Verify Installation
```bash
python --version
tesseract --version
```

---

## 🚀 Application Setup

### Step 1: Navigate to Project
```bash
cd ~/ChatBot  # Or your project location
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

### Step 5: Install Python Packages
```bash
pip install -r requirements.txt
```

This takes 5-10 minutes depending on your internet speed.

### Step 6: Run the Application
```bash
streamlit run ui/app.py
```

Browser opens to `http://localhost:8501`

---

## 🎯 First Run Experience

### What Happens:

1. **Embedding Model** (~90 MB, 1-2 min)
2. **Language Model** (~21 GB SOLAR, 10-15 min)
3. **Database Init** (few seconds)
4. **Ready!** 🎉

### With NVIDIA GPU:

The app automatically detects CUDA:
```
Device: cuda
```

### CPU Only:

Works fine, just slower:
```
Device: cpu
```

---

## 🎮 GPU Support (NVIDIA)

### Check GPU
```bash
nvidia-smi
```

### Install CUDA Toolkit

**Ubuntu/Debian:**
```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update

# Install CUDA
sudo apt install -y cuda
```

**Fedora:**
```bash
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora37/x86_64/cuda-fedora37.repo
sudo dnf install -y cuda
```

### Install PyTorch with CUDA

```bash
# Activate venv first
source .venv/bin/activate

# Uninstall CPU version
pip uninstall torch torchvision torchaudio

# Install GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU Support
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📝 Daily Usage

### Starting the App

**Option 1: Manual**
```bash
cd ~/ChatBot
source .venv/bin/activate
streamlit run ui/app.py
```

**Option 2: Create alias**

Add to `~/.bashrc` or `~/.zshrc`:
```bash
alias chatbot='cd ~/ChatBot && source .venv/bin/activate && streamlit run ui/app.py'
```

Reload:
```bash
source ~/.bashrc  # or ~/.zshrc
```

Now just type:
```bash
chatbot
```

**Option 3: Create script**

Create `~/chatbot.sh`:
```bash
#!/bin/bash
cd ~/ChatBot
source .venv/bin/activate
streamlit run ui/app.py
```

Make executable:
```bash
chmod +x ~/chatbot.sh
```

Run:
```bash
~/chatbot.sh
```

---

## 🔧 Troubleshooting

### Issue: "command not found: streamlit"

**Solution**:
```bash
source .venv/bin/activate
which streamlit  # Should show path in .venv
```

### Issue: "ModuleNotFoundError"

**Solution**:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Issue: "Permission denied"

**Solution**:
```bash
chmod +x .venv/bin/activate
chmod +x scripts/*.py
```

### Issue: "Port 8501 already in use"

**Solutions**:

**Option 1: Different port**
```bash
streamlit run ui/app.py --server.port 8502
```

**Option 2: Kill process**
```bash
# Find process
lsof -i :8501

# Kill it
kill -9 <PID>

# Or use fuser
sudo fuser -k 8501/tcp
```

### Issue: "CUDA out of memory"

**Solutions**:
1. Use smaller model (dnotitia instead of SOLAR)
2. Reduce max tokens in settings
3. Close other GPU applications
4. Use CPU mode (remove GPU packages)

### Issue: Slow on CPU

**Solutions**:
1. Use SOLAR model (better quality)
2. Reduce max_tokens to 256-512
3. Process documents in smaller batches
4. Consider adding GPU

---

## 🖥️ Desktop Integration

### Create Desktop Launcher (Ubuntu/GNOME)

Create `~/.local/share/applications/chatbot.desktop`:
```ini
[Desktop Entry]
Name=Multimodal Chatbot
Comment=RAG Chatbot with DSPy
Exec=/home/USERNAME/chatbot.sh
Icon=applications-science
Terminal=true
Type=Application
Categories=Development;Science;
```

Replace `USERNAME` with your username.

### Create Desktop Launcher (KDE)

Right-click desktop → Create New → Link to Application
- Name: Chatbot
- Command: `/home/USERNAME/chatbot.sh`
- Terminal: Yes

---

## 📊 System Monitoring

### Check resource usage:
```bash
# CPU and memory
htop

# GPU (NVIDIA)
nvidia-smi

# Disk usage
df -h
du -sh ~/ChatBot ~/.cache/huggingface/
```

---

## 🔐 Security & Privacy

### Firewall (optional)
```bash
# Allow Streamlit port
sudo ufw allow 8501/tcp

# Or restrict to localhost only (default)
# No changes needed
```

### SELinux (Fedora/RHEL)
```bash
# If you encounter permission issues
sudo setenforce 0  # Temporary
# Or configure SELinux policies
```

### Data locations:
```
~/ChatBot/data/uploads/          # Your documents
~/ChatBot/data/embeddings_db/    # Vector database
~/.cache/huggingface/hub/        # AI models
```

**All data stays local** - no external servers except model downloads.

---

## 📦 Updating

### Update packages:
```bash
source .venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Update system:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade

# Fedora
sudo dnf upgrade

# Arch
sudo pacman -Syu
```

---

## 🗑️ Uninstallation

### Remove application:
```bash
rm -rf ~/ChatBot
```

### Remove dependencies (optional):
```bash
# Ubuntu/Debian
sudo apt remove tesseract-ocr python3-venv

# Fedora
sudo dnf remove tesseract python3-devel

# Arch
sudo pacman -R tesseract python-pip
```

### Remove cached models:
```bash
rm -rf ~/.cache/huggingface/
```

---

## 🆘 Getting Help

### Test scripts:
```bash
python scripts/check_model_status.py
python scripts/test_model.py
```

### Check logs:
```bash
# Terminal shows all logs
# Look for ERROR or WARNING messages
```

### Documentation:
- [Main README](../../README.md)
- [Model Guide](../general/MODEL_GUIDE.md)
- [Quick Start](../general/QUICKSTART.md)

---

## ✅ Success Checklist

- [ ] System updated
- [ ] Python 3.8+ installed
- [ ] Tesseract installed
- [ ] Build tools installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] App runs without errors
- [ ] Browser opens to localhost:8501
- [ ] Can upload documents
- [ ] Can chat with documents

**All checked? You're ready!** 🎉

---

**Last Updated**: October 5, 2025  
**Tested on**: Ubuntu 22.04, Debian 11, Fedora 38, Arch Linux
