@echo off
REM Quick troubleshooting script for Windows

echo ========================================
echo Windows Troubleshooting Utility
echo ========================================
echo.

echo Running diagnostic checks...
echo.

echo [Check 1] Python Installation
python --version
if errorlevel 1 (
    echo FAILED: Python not found
    echo Solution: Install Python from https://www.python.org/downloads/
) else (
    echo PASSED
)
echo.

echo [Check 2] Python in PATH
where python
if errorlevel 1 (
    echo FAILED: Python not in PATH
    echo Solution: Add Python to system PATH
) else (
    echo PASSED
)
echo.

echo [Check 3] Pip Installation
pip --version
if errorlevel 1 (
    echo FAILED: pip not found
    echo Solution: Reinstall Python with pip
) else (
    echo PASSED
)
echo.

echo [Check 4] Virtual Environment
if exist "venv\" (
    echo PASSED: Virtual environment exists
) else (
    echo FAILED: Virtual environment not found
    echo Solution: Run setup_windows.bat
)
echo.

echo [Check 5] Tesseract OCR
tesseract --version
if errorlevel 1 (
    echo WARNING: Tesseract not found
    echo Solution: Install from https://github.com/UB-Mannheim/tesseract/wiki
) else (
    echo PASSED
)
echo.

echo [Check 6] Required Packages
if exist "venv\" (
    call venv\Scripts\activate.bat
    python -c "import torch; print('PyTorch:', torch.__version__)"
    python -c "import transformers; print('Transformers:', transformers.__version__)"
    python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
    python -c "import chromadb; print('ChromaDB:', chromadb.__version__)"
    python -c "import langchain; print('LangChain:', langchain.__version__)"
    if errorlevel 1 (
        echo FAILED: Some packages missing
        echo Solution: Run setup_windows.bat
    ) else (
        echo PASSED: All core packages installed
    )
) else (
    echo SKIPPED: Virtual environment not found
)
echo.

echo [Check 7] Disk Space
wmic logicaldisk get caption,freespace,size | find "C:"
echo Ensure at least 10GB free space
echo.

echo [Check 8] System Memory
wmic computersystem get totalphysicalmemory
echo Recommended: 8GB+ RAM
echo.

echo ========================================
echo Diagnostic Complete
echo ========================================
echo.
echo If you see any FAILED checks, follow the solutions above.
echo For more help, see WINDOWS_SETUP.md
echo.
pause
