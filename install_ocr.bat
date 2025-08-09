@echo off
echo ========================================
echo   OCR Setup for Menu Intelligence
echo   (Updated for Gemini Vision API)
echo ========================================
echo.

cd /d "C:\Users\rohan\Desktop\Puch AI"

echo [1/4] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Please ensure the virtual environment exists in .venv folder
    pause
    exit /b 1
)

echo [2/4] Installing Google Generative AI (Gemini)...
.venv\Scripts\pip.exe install google-generativeai
if errorlevel 1 (
    echo WARNING: Failed to install google-generativeai
    echo You may need to check your internet connection
)

echo [3/4] Installing EasyOCR (offline fallback)...
.venv\Scripts\pip.exe install easyocr
if errorlevel 1 (
    echo WARNING: Failed to install easyocr
    echo This is optional but recommended for offline OCR
)

echo [4/4] Installing additional dependencies...
.venv\Scripts\pip.exe install torch torchvision Pillow
if errorlevel 1 (
    echo WARNING: Some dependencies failed to install
)

echo.
echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo Setup Information:
echo - Gemini Vision API: Primary OCR method (requires API key)
echo - EasyOCR: Offline fallback (no API key needed)
echo.
echo Next Steps:
echo 1. Get Gemini API key from: https://aistudio.google.com/app/apikey
echo 2. Add to .env file: GEMINI_API_KEY=your_api_key_here
echo 3. Start your MCP server
echo 4. Test menu intelligence with an image
echo.
echo ========================================

pause
