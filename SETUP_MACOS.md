# macOS Setup Guide

Complete setup instructions for running the Multimodal RAG Chatbot on macOS.

## Prerequisites

### 1. Install Homebrew (if not installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Python 3.8+
```bash
# Install Python via Homebrew
brew install python@3.11

# Verify installation
python3 --version  # Should show 3.8 or higher
```

### 3. Install Node.js 18+
```bash
# Install Node.js via Homebrew
brew install node

# Verify installation
node --version   # Should show 18 or higher
npm --version
```

### 4. Install Git (if not installed)
```bash
brew install git
```

## Installation

### Step 1: Clone or Navigate to Project
```bash
cd /Users/hoangquangminh/Library/CloudStorage/OneDrive-FPTCorporation/ChatBot
```

### Step 2: Install Python Dependencies

#### Option A: Using pip (Recommended)
```bash
# Install main dependencies
pip3 install -r requirements.txt

# Install backend dependencies
pip3 install -r backend/requirements.txt
```

#### Option B: Using virtual environment
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
lsof -i :8000

# Kill the process
kill -9 <PID>
```

**Frontend (3000):**
```bash
# Find process using port 3000
lsof -i :3000

# Kill the process
kill -9 <PID>
```

### Python Module Not Found
```bash
# Ensure you're in the project root
cd /Users/hoangquangminh/Library/CloudStorage/OneDrive-FPTCorporation/ChatBot

# Reinstall dependencies
pip3 install -r requirements.txt --upgrade
pip3 install -r backend/requirements.txt --upgrade
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

### Permission Denied Errors
```bash
# Fix Python permissions
pip3 install --user -r requirements.txt

# Fix npm permissions
npm config set prefix ~/.npm-global
export PATH=~/.npm-global/bin:$PATH
```

### ChromaDB Errors
```bash
# Delete and recreate database
rm -rf data/embeddings_db/

# Restart backend - it will recreate the database
```

## Performance Tips

### Use High-Performance Model (7.8B)
After first message, switch to larger model:
```bash
curl -X POST http://localhost:8000/api/model/config \
  -H "Content-Type: application/json" \
  -d '{"high_parameter": true}'
```

### Optimize for Apple Silicon (M1/M2/M3)
Models automatically use Metal Performance Shaders (MPS) for GPU acceleration on Apple Silicon Macs.

### Monitor Resources
```bash
# Check CPU/Memory usage
top -o cpu

# Check GPU usage (Apple Silicon only)
sudo powermetrics --samplers gpu_power -i 1000
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

## Uninstallation

### Remove Application
```bash
# Just delete the folder
rm -rf /Users/hoangquangminh/Library/CloudStorage/OneDrive-FPTCorporation/ChatBot
```

### Remove Cached Models (Optional)
```bash
# This will free up ~5-16GB of disk space
rm -rf ~/.cache/huggingface/
```

### Remove Virtual Environment (if created)
```bash
cd /Users/hoangquangminh/Library/CloudStorage/OneDrive-FPTCorporation/ChatBot
rm -rf venv/
```

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
