@echo off
REM Windows Run Script for Multimodal RAG Chatbot
REM Double-click this file to start the application

echo ========================================
echo Multimodal RAG Chatbot
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_windows.bat first
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Check if Streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ERROR: Streamlit not installed
    echo Please run setup_windows.bat first
    pause
    exit /b 1
)

REM Start the application
echo.
echo Starting Multimodal RAG Chatbot...
echo.
echo The application will open in your default browser
echo URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.
echo ========================================
echo.

REM Run Streamlit
streamlit run ui/app.py

REM If streamlit exits
echo.
echo Application stopped.
pause
