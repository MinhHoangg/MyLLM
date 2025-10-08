# Windows Setup Guide

Complete setup instructions for running the Multimodal RAG Chatbot on Windows 10/11.

## Prerequisites

### 1. Install Python 3.8+

**Option A: Microsoft Store (Recommended)**
1. Open Microsoft Store
2. Search for "Python 3.11"
3. Click "Get" to install

**Option B: Python.org**
1. Download from https://www.python.org/downloads/
2. Run installer
3. ✅ **IMPORTANT:** Check "Add Python to PATH"
4. Click "Install Now"

**Verify installation:**
```powershell
python --version
```

### 2. Install Node.js 18+

1. Download from https://nodejs.org/ (LTS version)
2. Run installer with default settings
3. Restart your terminal

**Verify installation:**
```powershell
node --version
npm --version
```

### 3. Install Git (Optional)

Download from https://git-scm.com/download/win and install with default settings.

## Installation

### Step 1: Navigate to Project

Open PowerShell or Command Prompt:
```powershell
cd C:\Users\YourUsername\path\to\ChatBot
```

Or if on OneDrive:
```powershell
cd "C:\Users\YourUsername\OneDrive\ChatBot"
```

### Step 2: Install Python Dependencies

```powershell
# Install main dependencies
pip install -r requirements.txt

# Install backend dependencies
pip install -r backend\requirements.txt
```

**If you get SSL errors:**
```powershell
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

**If you get permission errors:**
```powershell
pip install --user -r requirements.txt
pip install --user -r backend\requirements.txt
```

### Step 3: Install Frontend Dependencies

```powershell
cd frontend
npm install
cd ..
```

**If npm install fails:**
```powershell
# Clear cache and retry
npm cache clean --force
npm install
```

## Running the Application

### Start Backend (Terminal 1)

Open PowerShell/Command Prompt:
```powershell
cd backend
python main.py
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

Open another PowerShell/Command Prompt:
```powershell
cd frontend
npm run dev
```

Expected output:
```
  VITE v5.0.8  ready in 1234 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
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
3. **Models cached at:** `C:\Users\YourUsername\.cache\huggingface\hub\`

**Subsequent runs:** Models load from cache (no download needed)

## Verify Installation

### 1. Check Backend Health

In PowerShell:
```powershell
curl http://localhost:8000/api/health
```

Or open in browser: http://localhost:8000/api/health

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

Open http://localhost:3000 in your browser - you should see the chat interface.

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
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace <PID> with actual process ID)
taskkill /PID <PID> /F
```

**Frontend (3000):**
```powershell
# Find process using port 3000
netstat -ano | findstr :3000

# Kill the process
taskkill /PID <PID> /F
```

### Python Not Found

If you get `'python' is not recognized`:
```powershell
# Try python3 instead
python3 --version

# Or py launcher
py --version

# Use whichever works in the commands
```

### Module Not Found Errors

```powershell
# Ensure you're in the project root
cd C:\path\to\ChatBot

# Reinstall dependencies
pip install -r requirements.txt --upgrade
pip install -r backend\requirements.txt --upgrade
```

### Node Module Errors

```powershell
cd frontend

# Clear cache and reinstall
rmdir /s /q node_modules
del package-lock.json
npm cache clean --force
npm install
```

### Model Download Fails

```powershell
# Check internet connection
ping huggingface.co

# Clear cache and retry
rmdir /s /q %USERPROFILE%\.cache\huggingface\hub\models--LGAI-EXAONE*
# Restart backend - it will re-download
```

### Windows Defender / Antivirus Blocking

If Windows Defender blocks Python or Node:
1. Open Windows Security
2. Go to "Virus & threat protection"
3. Click "Manage settings"
4. Add exclusions for:
   - Python installation folder
   - Node.js installation folder
   - Project folder

### Long Path Issues

If you get "path too long" errors:
1. Open Group Policy Editor: `gpedit.msc`
2. Navigate to: Computer Configuration > Administrative Templates > System > Filesystem
3. Enable "Enable Win32 long paths"
4. Restart

Or use PowerShell as Administrator:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### ChromaDB Errors

```powershell
# Delete and recreate database
rmdir /s /q data\embeddings_db

# Restart backend - it will recreate the database
```

### PyTorch CUDA (Optional - For NVIDIA GPUs)

To use GPU acceleration with NVIDIA graphics card:
```powershell
# Install CUDA-enabled PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Performance Tips

### Use High-Performance Model (7.8B)

After first message, switch to larger model:
```powershell
curl -X POST http://localhost:8000/api/model/config -H "Content-Type: application/json" -d "{\"high_parameter\": true}"
```

Or use PowerShell's Invoke-RestMethod:
```powershell
Invoke-RestMethod -Uri http://localhost:8000/api/model/config -Method POST -ContentType "application/json" -Body '{"high_parameter": true}'
```

### Monitor Resources

Open Task Manager (`Ctrl + Shift + Esc`):
- Check Python.exe CPU/Memory usage
- Check node.exe CPU/Memory usage

## Development Mode

### Backend with Auto-Reload

```powershell
cd backend
uvicorn main:app --reload --port 8000
```

### Frontend with HMR (Hot Module Replacement)

```powershell
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

```powershell
# Kill backend
taskkill /F /IM python.exe

# Kill frontend
taskkill /F /IM node.exe
```

## Running as Windows Services (Optional)

### Using NSSM (Non-Sucking Service Manager)

1. Download NSSM from https://nssm.cc/download
2. Extract to a folder

**Install Backend Service:**
```powershell
nssm install ChatbotBackend "C:\Python311\python.exe" "C:\path\to\ChatBot\backend\main.py"
nssm start ChatbotBackend
```

**Install Frontend Service:**
```powershell
nssm install ChatbotFrontend "C:\Program Files\nodejs\npm.cmd" "run dev"
nssm set ChatbotFrontend AppDirectory "C:\path\to\ChatBot\frontend"
nssm start ChatbotFrontend
```

## Uninstallation

### Remove Application
Simply delete the ChatBot folder

### Remove Cached Models (Optional)
```powershell
# This will free up ~5-16GB of disk space
rmdir /s /q %USERPROFILE%\.cache\huggingface
```

### Remove Python Packages
```powershell
pip uninstall -r requirements.txt -y
pip uninstall -r backend\requirements.txt -y
```

## Windows-Specific Notes

### File Paths
- Use backslashes: `C:\path\to\file`
- Or forward slashes: `C:/path/to/file`
- Spaces in paths need quotes: `"C:\Program Files\..."`

### Terminal Options
- **PowerShell** (Recommended for Windows 10/11)
- **Command Prompt** (cmd.exe)
- **Windows Terminal** (Best experience - install from Microsoft Store)
- **Git Bash** (If Git installed)

### Environment Variables
To set permanently:
```powershell
# Open System Properties
rundll32 sysdm.cpl,EditEnvironmentVariables

# Add to PATH or create new variables
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
| Start Backend | `cd backend && python main.py` |
| Start Frontend | `cd frontend && npm run dev` |
| View API Docs | http://localhost:8000/docs |
| View Frontend | http://localhost:3000 |
| Check Health | `curl http://localhost:8000/api/health` |
| Stop Servers | Press `Ctrl + C` in each terminal |
| Force Kill | `taskkill /F /IM python.exe` (backend)<br>`taskkill /F /IM node.exe` (frontend) |
