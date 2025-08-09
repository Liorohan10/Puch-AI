# OCR Setup Instructions for Menu Intelligence

This document explains how to set up OCR (Optical Character Recognition) for the menu intelligence feature in your MCP Travel Assistant server.

## Overview

The menu intelligence tool now uses **Google Gemini Vision API** as the primary method with EasyOCR as fallback:

1. **Google Gemini Vision API** (Primary - highest accuracy with intelligent analysis)
2. **EasyOCR** (Offline fallback - no API key needed)

## Quick Setup Options

### Option 1: Google Gemini API (Recommended)

The best option - provides intelligent menu analysis with OCR:

1. **Get Gemini API Key:**
   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the API key

2. **Add to Environment:**
   ```bash
   # Add to your .env file
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Install Dependencies:**
   ```bash
   pip install google-generativeai
   ```

### Option 2: EasyOCR (Offline Fallback)

Works offline without any API keys:

```bash
pip install easyocr torch torchvision
```

That's it! EasyOCR will be used automatically if Gemini API is not configured.

## What Makes Gemini Vision Special

Unlike traditional OCR, Gemini Vision provides:

- **Intelligent text extraction** with context understanding
- **Menu item identification** with automatic categorization
- **Price detection** and formatting
- **Allergen analysis** with ingredient recognition
- **Cuisine type detection** 
- **Smart recommendations** based on preferences
- **Multi-language support** with natural understanding

## Testing OCR Setup

Run this script to test your OCR setup:

```python
import base64
import asyncio
from your_mcp_server import menu_intelligence

async def test_ocr():
    # Test with a sample image (replace with actual menu image)
    with open("test_menu.jpg", "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    result = await menu_intelligence(
        menu_image_base64=image_base64,
        allergies=["nuts", "dairy"],
        preferences=["vegetarian", "spicy"]
    )
    
    print("OCR Result:", result)
    print("Detected Items:", result.get("menu_items", []))
    print("Recommendations:", result.get("recommendations", []))
    print("Allergen Warnings:", result.get("allergen_warnings", []))

asyncio.run(test_ocr())
```

## Environment Configuration

Your `.env` file should contain:

```bash
# Authentication
AUTH_TOKEN=your_bearer_token_here
MY_NUMBER=your_phone_number_here

# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here
```

## Troubleshooting

### Common Issues:

1. **"Import google.generativeai could not be resolved"**
   - Solution: `pip install google-generativeai`

2. **"GEMINI_API_KEY not found in environment"**
   - Check your `.env` file has the correct key
   - Ensure `.env` file is in the project root
   - Restart your server after adding the key

3. **"No module named 'easyocr'"**
   - Solution: `pip install easyocr`

4. **"All OCR methods failed"**
   - Check internet connection for Gemini API
   - Verify API key is valid
   - Ensure image is properly base64 encoded

### Image Quality Tips:

- **Good lighting:** Ensure menu is well-lit
- **Straight angle:** Hold camera parallel to menu
- **High resolution:** Use camera's highest quality setting
- **Stable shot:** Avoid blurry images
- **Clean surface:** Remove reflections and shadows
- **Full menu:** Capture complete sections for better context

## Supported Languages

- **Gemini Vision:** 100+ languages with intelligent context understanding
- **EasyOCR:** 80+ languages including English, Spanish, French, German, Chinese, Japanese, Arabic

## Cost Considerations

- **Gemini API:** Free tier includes generous limits for testing
- **EasyOCR:** Free (offline processing)

## Security Notes

- Keep API keys secure and never commit them to code
- Use environment variables for API keys
- Consider rate limiting for production use
- Monitor API usage

## Performance Tips

1. **Image optimization:** Resize large images to max 2048px width
2. **Format:** Use JPEG for photos, PNG for screenshots
3. **Quality:** Higher quality images = better analysis
4. **Context:** Include full menu sections for better understanding

## Advanced Features

Gemini Vision provides advanced menu analysis:

```json
{
    "extracted_text": "Full menu text",
    "menu_items": [
        {
            "name": "Margherita Pizza",
            "price": "$12.99",
            "description": "Fresh mozzarella, tomato sauce, basil",
            "category": "main"
        }
    ],
    "cuisine_type": "Italian",
    "price_range": "moderate",
    "vegetarian_options": ["Margherita Pizza", "Caprese Salad"],
    "allergen_warnings": ["Contains dairy (mozzarella)"],
    "recommendations": ["Margherita Pizza (vegetarian option)"]
}
```

## Migration from Previous OCR

If upgrading from Google Vision/Azure setup:

1. Remove old dependencies:
   ```bash
   pip uninstall google-cloud-vision azure-cognitiveservices-vision-computervision
   ```

2. Install Gemini:
   ```bash
   pip install google-generativeai
   ```

3. Update environment variables:
   ```bash
   # Remove these old keys
   # GOOGLE_VISION_API_KEY=...
   # AZURE_VISION_ENDPOINT=...
   # AZURE_VISION_KEY=...
   
   # Add this new key
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Quick Install Script

Run the updated installation script:

```cmd
install_ocr.bat
```

This script now installs:
- `google-generativeai` for Gemini Vision
- `easyocr` for offline fallback
- All required dependencies

## Testing with Claude Desktop

Once your server is running with ngrok:

```
Test the menu intelligence:
menu_intelligence menu_image_base64="[paste_base64_image_here]" allergies=["nuts"] preferences=["vegetarian"]
```

Expected intelligent response from Gemini:
```json
{
  "extracted_text": "Complete menu text...",
  "menu_items": [
    {
      "name": "Margherita Pizza", 
      "price": "$12.99",
      "category": "main",
      "description": "Fresh mozzarella, tomato, basil"
    }
  ],
  "cuisine_type": "Italian",
  "price_range": "moderate", 
  "recommendations": ["Margherita Pizza (vegetarian, no nuts)"],
  "allergen_warnings": [],
  "vegetarian_options": ["Margherita Pizza"],
  "ocr_method": "Gemini Vision API",
  "success": true
}
```
