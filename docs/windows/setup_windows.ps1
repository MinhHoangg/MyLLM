# PowerShell Setup Script for Multimodal RAG Chatbot
# Run this with: PowerShell -ExecutionPolicy Bypass -File setup_windows.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Multimodal RAG Chatbot Setup (PowerShell)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host $pythonVersion -ForegroundColor Green
    Write-Host "Python is installed!" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Red
    Pause
    exit 1
}
Write-Host ""

# Check Tesseract installation
Write-Host "[2/6] Checking Tesseract OCR installation..." -ForegroundColor Yellow
try {
    $tesseractVersion = tesseract --version 2>&1
    Write-Host "Tesseract OCR is installed!" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Tesseract OCR is not installed" -ForegroundColor Yellow
    Write-Host "Please install from: https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
    Write-Host "The chatbot will work but OCR from images won't be available" -ForegroundColor Yellow
    Write-Host ""
    Pause
}
Write-Host ""

# Check if virtual environment exists
Write-Host "[3/6] Checking virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists. Skipping creation." -ForegroundColor Green
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        Pause
        exit 1
    }
    Write-Host "Virtual environment created!" -ForegroundColor Green
}
Write-Host ""

# Activate virtual environment
Write-Host "[4/6] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "You may need to run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    Pause
    exit 1
}
Write-Host "Virtual environment activated!" -ForegroundColor Green
Write-Host ""

# Upgrade pip
Write-Host "[5/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "Pip upgraded!" -ForegroundColor Green
Write-Host ""

# Install dependencies
Write-Host "[6/6] Installing dependencies (this may take 10-15 minutes)..." -ForegroundColor Yellow
Write-Host "Please be patient, downloading ~4GB of data..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    Write-Host "Try running: pip install -r requirements.txt" -ForegroundColor Yellow
    Pause
    exit 1
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Run the application using: .\run_windows.ps1" -ForegroundColor White
Write-Host "2. Or manually: .\venv\Scripts\Activate.ps1 followed by streamlit run ui/app.py" -ForegroundColor White
Write-Host ""
Write-Host "The first run will download the AI model (~3.5GB)" -ForegroundColor Yellow
Write-Host "This only happens once." -ForegroundColor Yellow
Write-Host ""
Pause
