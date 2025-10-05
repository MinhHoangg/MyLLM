@echo off
REM Windows Setup Script for Multimodal RAG Chatbot
REM This script will set up the entire environment on Windows

echo ========================================
echo Multimodal RAG Chatbot Setup
echo ========================================
echo.

REM Check if Python is installed
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)
python --version
echo Python is installed!
echo.

REM Check if Tesseract is installed
echo [2/6] Checking Tesseract OCR installation...
tesseract --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Tesseract OCR is not installed
    echo Please install from: https://github.com/UB-Mannheim/tesseract/wiki
    echo The chatbot will work but OCR from images won't be available
    echo.
    pause
) else (
    echo Tesseract OCR is installed!
    echo.
)

REM Check if virtual environment exists
echo [3/6] Checking virtual environment...
if exist "venv\" (
    echo Virtual environment already exists. Skipping creation.
) else (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created!
)
echo.

REM Activate virtual environment
echo [4/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated!
echo.

REM Upgrade pip
echo [5/6] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo Pip upgraded!
echo.

REM Install dependencies
echo [6/6] Installing dependencies (this may take 10-15 minutes)...
echo Please be patient, downloading ~4GB of data...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Try running: pip install -r requirements.txt
    pause
    exit /b 1
)
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Run the application using: run_windows.bat
echo 2. Or manually: venv\Scripts\activate followed by streamlit run ui/app.py
echo.
echo The first run will download the AI model (~3.5GB)
echo This only happens once.
echo.
pause
