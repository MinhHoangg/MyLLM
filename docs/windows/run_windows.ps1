# PowerShell Run Script for Multimodal RAG Chatbot
# Run this with: PowerShell -ExecutionPolicy Bypass -File run_windows.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Multimodal RAG Chatbot" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup_windows.ps1 first" -ForegroundColor Yellow
    Write-Host ""
    Pause
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "You may need to run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    Pause
    exit 1
}

# Check if Streamlit is installed
python -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Streamlit not installed" -ForegroundColor Red
    Write-Host "Please run setup_windows.ps1 first" -ForegroundColor Yellow
    Pause
    exit 1
}

# Start the application
Write-Host ""
Write-Host "Starting Multimodal RAG Chatbot..." -ForegroundColor Green
Write-Host ""
Write-Host "The application will open in your default browser" -ForegroundColor Cyan
Write-Host "URL: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Run Streamlit
streamlit run ui/app.py

# If streamlit exits
Write-Host ""
Write-Host "Application stopped." -ForegroundColor Yellow
Pause
