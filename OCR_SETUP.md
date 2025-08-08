# OCR Setup for Menu Intelligence

## Install Dependencies

1. **Install Python packages:**
   ```cmd
   cd "C:\Users\rohan\Desktop\Puch AI"
   .venv\Scripts\pip.exe install pytesseract opencv-python numpy
   ```

2. **Install Tesseract OCR (Windows):**
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install the .exe file (usually to `C:\Program Files\Tesseract-OCR\`)
   - Add to Windows PATH or set in your code

3. **Alternative: Use install_ocr.bat**
   - Double-click `install_ocr.bat` to install Python packages

## Configuration

If Tesseract is not in PATH, add this to your .env file:
```
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Testing Menu Intelligence

After setup, test with:
```
menu_intelligence menu_image_base64="[your_base64_image]" allergies=["nuts"] preferences=["vegetarian"]
```

The tool will now:
- Extract text from menu images using OCR
- Detect menu items and prices
- Identify allergens in the text
- Provide recommendations based on preferences
- Give culturally appropriate etiquette tips
