# Linux Setup Guide

Complete setup instructions for running the Multimodal RAG Chatbot on Linux (Ubuntu, Debian, Fedora, Arch, etc.).

## Prerequisites

### 1. Install Python 3.8+

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**Fedora:**
```bash
sudo dnf install python3 python3-pip
```

**Arch Linux:**
```bash
sudo pacman -S python python-pip
```

**Verify installation:**
```bash
python3 --version  # Should show 3.8 or higher
```

### 2. Install Node.js 18+

**Ubuntu/Debian:**
```bash
# Using NodeSource repository
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install nodejs
```

**Fedora:**
```bash
sudo dnf install nodejs npm
```

**Arch Linux:**
```bash
sudo pacman -S nodejs npm
```

**Verify installation:**
```bash
node --version   # Should show 18 or higher
npm --version
```

### 3. Install Build Tools

**Ubuntu/Debian:**
```bash
sudo apt install build-essential git
```

**Fedora:**
```bash
sudo dnf groupinstall "Development Tools"
sudo dnf install git
```

**Arch Linux:**
```bash
sudo pacman -S base-devel git
```

## Installation

### Step 1: Navigate to Project
```bash
cd /path/to/ChatBot
```

### Step 2: Install Python Dependencies

#### Option A: System-wide Installation
```bash
# Install main dependencies
pip3 install -r requirements.txt

# Install backend dependencies
pip3 install -r backend/requirements.txt
```

#### Option B: Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### Step 3: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

## Running the Application

### Start Backend (Terminal 1)
```bash
cd backend
python3 main.py
```

Expected output:
```
{"timestamp": "2025-10-08 12:00:00", "level": "INFO", "module": "main", "message": "Initializing application..."}
{"timestamp": "2025-10-08 12:00:05", "level": "INFO", "module": "main", "message": "Loading vector database..."}
{"timestamp": "2025-10-08 12:00:10", "level": "INFO", "module": "main", "message": "Model will be loaded on first request..."}
{"timestamp": "2025-10-08 12:00:10", "level": "INFO", "module": "main", "message": "Application initialized successfully!"}

INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Start Frontend (Terminal 2)
```bash
cd frontend
npm run dev
```

Expected output:
```
  VITE v5.0.8  ready in 1234 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

## Access the Application

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/health

## First Run - Model Download

On first run, models will be downloaded from Hugging Face:

1. **CLIP Model** (~600MB) - Downloads in ~30s on good connection
2. **EXAONE-3.5-2.4B** (~4.8GB) - Downloads in ~2-5 minutes
3. **Models cached at:** `~/.cache/huggingface/hub/`

**Subsequent runs:** Models load from cache (no download needed)

## Verify Installation

### 1. Check Backend Health
```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "vector_db": true,
  "chatbot": true,
  "model_loaded": false
}
```

### 2. Test Frontend
Open http://localhost:3000 in browser - you should see the chat interface.

### 3. Upload a Test Document
- Click "Documents" tab
- Drag and drop a text file or image
- Verify it appears in the document list

### 4. Test Chat
- Go to "Chat" tab
- Ask: "What documents do I have?"
- Should see a response using RAG

## Troubleshooting

### Port Already in Use

**Backend (8000):**
```bash
# Find process using port 8000
sudo lsof -i :8000
# or
sudo netstat -tulpn | grep :8000

# Kill the process
sudo kill -9 <PID>
```

**Frontend (3000):**
```bash
# Find process using port 3000
sudo lsof -i :3000

# Kill the process
sudo kill -9 <PID>
```

### Python Module Not Found
```bash
# Ensure you're in the project root
cd /path/to/ChatBot

# Reinstall dependencies
pip3 install -r requirements.txt --upgrade
pip3 install -r backend/requirements.txt --upgrade
```

### Permission Denied Errors
```bash
# Install with --user flag
pip3 install --user -r requirements.txt
pip3 install --user -r backend/requirements.txt

# Or fix npm permissions
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Node Module Errors
```bash
cd frontend

# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

### Model Download Fails
```bash
# Check internet connection
ping huggingface.co

# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/models--LGAI-EXAONE*
# Restart backend - it will re-download
```

### ChromaDB SQLite Errors

If you get SQLite version errors:
```bash
# Ubuntu/Debian
sudo apt install libsqlite3-dev

# Fedora
sudo dnf install sqlite-devel

# Arch
sudo pacman -S sqlite

# Rebuild ChromaDB
pip3 install --force-reinstall chromadb
```

### Memory Issues

If system runs out of memory:
```bash
# Check available memory
free -h

# For 2.4B model: Need at least 6GB RAM
# For 7.8B model: Need at least 18GB RAM

# Close other applications or use smaller model
```

## Performance Tips

### Use High-Performance Model (7.8B)
After first message, switch to larger model:
```bash
curl -X POST http://localhost:8000/api/model/config \
  -H "Content-Type: application/json" \
  -d '{"high_parameter": true}'
```

### CUDA Support (NVIDIA GPU)

If you have NVIDIA GPU:
```bash
# Check CUDA availability
nvidia-smi

# Install CUDA-enabled PyTorch
pip3 uninstall torch torchvision
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### AMD GPU (ROCm)

For AMD GPUs on supported hardware:
```bash
# Install ROCm PyTorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
```

### Monitor Resources
```bash
# CPU and Memory
htop
# or
top

# GPU usage (NVIDIA)
watch -n 1 nvidia-smi

# Disk I/O
iotop
```

## Development Mode

### Backend with Auto-Reload
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### Frontend with HMR (Hot Module Replacement)
```bash
cd frontend
npm run dev
```

Changes to code will automatically reload!

## Running as System Service (systemd)

### Create Backend Service

Create `/etc/systemd/system/chatbot-backend.service`:
```ini
[Unit]
Description=Chatbot Backend Service
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/ChatBot/backend
Environment="PATH=/usr/bin:/usr/local/bin"
ExecStart=/usr/bin/python3 main.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

### Create Frontend Service

Create `/etc/systemd/system/chatbot-frontend.service`:
```ini
[Unit]
Description=Chatbot Frontend Service
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/ChatBot/frontend
Environment="PATH=/usr/bin:/usr/local/bin"
ExecStart=/usr/bin/npm run dev
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

### Enable and Start Services
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services (start on boot)
sudo systemctl enable chatbot-backend
sudo systemctl enable chatbot-frontend

# Start services
sudo systemctl start chatbot-backend
sudo systemctl start chatbot-frontend

# Check status
sudo systemctl status chatbot-backend
sudo systemctl status chatbot-frontend

# View logs
sudo journalctl -u chatbot-backend -f
sudo journalctl -u chatbot-frontend -f
```

## Firewall Configuration

### UFW (Ubuntu/Debian)
```bash
# Allow ports
sudo ufw allow 8000/tcp
sudo ufw allow 3000/tcp

# Check status
sudo ufw status
```

### firewalld (Fedora/RHEL)
```bash
# Allow ports
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --permanent --add-port=3000/tcp
sudo firewall-cmd --reload

# Check status
sudo firewall-cmd --list-all
```

## Stopping the Application

### Stop Backend
Press `Ctrl + C` in Terminal 1

### Stop Frontend
Press `Ctrl + C` in Terminal 2

### Force Kill (if frozen)
```bash
# Kill backend
pkill -f "python3 main.py"

# Kill frontend
pkill -f "vite"
```

### Stop Services (if using systemd)
```bash
sudo systemctl stop chatbot-backend
sudo systemctl stop chatbot-frontend
```

## Uninstallation

### Remove Application
```bash
# Delete the folder
rm -rf /path/to/ChatBot
```

### Remove Cached Models (Optional)
```bash
# This will free up ~5-16GB of disk space
rm -rf ~/.cache/huggingface/
```

### Remove Virtual Environment (if created)
```bash
rm -rf /path/to/ChatBot/venv/
```

### Remove System Services (if created)
```bash
sudo systemctl stop chatbot-backend chatbot-frontend
sudo systemctl disable chatbot-backend chatbot-frontend
sudo rm /etc/systemd/system/chatbot-*.service
sudo systemctl daemon-reload
```

## Distribution-Specific Notes

### Ubuntu/Debian
- Package manager: `apt`
- Service manager: `systemd`
- Python command: `python3`

### Fedora/RHEL/CentOS
- Package manager: `dnf` (or `yum` for older versions)
- Service manager: `systemd`
- Python command: `python3`
- May need to enable SELinux exceptions

### Arch Linux
- Package manager: `pacman`
- Service manager: `systemd`
- Python command: `python`
- Cutting-edge packages

### Gentoo
- Package manager: `emerge`
- Compile from source
- May need USE flags for Python features

## Next Steps

1. **Upload Documents:** Go to http://localhost:3000/documents
2. **Test Chat:** Ask questions about your documents
3. **Try Search:** Use the search feature to find specific content
4. **Switch Models:** Try the 7.8B model for better responses
5. **Check API Docs:** Explore http://localhost:8000/docs

## Support

- See main [README.md](README.md) for general documentation
- Check API documentation at http://localhost:8000/docs
- Review logs in terminal for debugging

---

**Quick Reference:**

| Action | Command |
|--------|---------|
| Start Backend | `cd backend && python3 main.py` |
| Start Frontend | `cd frontend && npm run dev` |
| View API Docs | http://localhost:8000/docs |
| View Frontend | http://localhost:3000 |
| Check Health | `curl http://localhost:8000/api/health` |
| Stop Servers | Press `Ctrl + C` in each terminal |
| Force Kill | `pkill -f "python3 main.py"` (backend)<br>`pkill -f "vite"` (frontend) |
| View Logs (systemd) | `journalctl -u chatbot-backend -f` |
