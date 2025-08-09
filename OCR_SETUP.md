# Enhanced Menu Intelligence & Local Cuisine Discovery

## 🚀 Powered by Gemini 2.0 Pro

Your MCP Travel Assistant now features advanced menu intelligence and local cuisine discovery powered by Google's latest Gemini 2.0 Pro model.

## 🌟 New Capabilities

### **Enhanced Menu Analysis**
- **Gemini 2.0 Pro Vision**: Advanced image analysis with cultural context
- **Comprehensive Menu Parsing**: Automatic categorization and ingredient analysis
- **Intelligent Recommendations**: Personalized suggestions based on preferences
- **Cultural Dining Insights**: Local customs and etiquette guidance
- **Advanced Allergen Detection**: Safety analysis with confidence scoring

### **Local Cuisine Discovery** ⭐ **NEW**
- **Must-Try Dishes**: Authentic local specialties with cultural significance
- **Restaurant Recommendations**: Local favorites vs tourist traps
- **Food Districts**: Best areas for authentic dining experiences
- **Seasonal Specialties**: What's available when you visit
- **Budget Guidance**: From street food to fine dining options

## 🔧 Setup Instructions

### Option 1: Gemini 2.0 Pro (Recommended)

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

### Option 2: EasyOCR (Fallback)

Works offline without any API keys:
```bash
pip install easyocr torch torchvision
```

## 🍽️ Usage Examples

### **Menu Analysis with Image**
```json
{
  "menu_image_base64": "[base64_image_data]",
  "allergies": ["nuts", "dairy"],
  "preferences": ["vegetarian", "spicy"],
  "location": "Tokyo, Japan",
  "discovery_mode": false
}
```

### **Local Cuisine Discovery** 
```json
{
  "location": "Bangkok, Thailand",
  "allergies": ["shellfish"],
  "preferences": ["authentic", "street food"],
  "discovery_mode": true
}
```

## 📊 Enhanced Response Format

### **Menu Analysis Response**
```json
{
  "mode": "enhanced_menu_analysis",
  "menu_analysis": {
    "cuisine_type": "Traditional Thai",
    "restaurant_style": "street food",
    "price_range": "budget"
  },
  "menu_categories": [
    {
      "category": "mains",
      "items": [
        {
          "name": "ผัดไทย",
          "english_translation": "Pad Thai",
          "cultural_significance": "Thailand's national dish",
          "allergen_warnings": ["fish sauce", "tamarind"],
          "authenticity_score": 9
        }
      ]
    }
  ],
  "personalized_recommendations": [
    {
      "dish_name": "Pad Thai",
      "order_priority": "must-try",
      "cultural_notes": "Best when made with fresh tamarind",
      "pairing_suggestions": "Thai iced tea"
    }
  ],
  "cultural_dining_insights": {
    "dining_etiquette": ["Use fork and spoon, not chopsticks"],
    "sharing_culture": "Family style sharing expected"
  }
}
```

### **Cuisine Discovery Response**
```json
{
  "mode": "cuisine_discovery",
  "location": "Bangkok, Thailand",
  "must_try_dishes": [
    {
      "name": "ส้มตำ",
      "english_name": "Som Tam",
      "cultural_significance": "Northeast Thailand's most famous salad",
      "spice_level": "very hot",
      "best_time_to_eat": "lunch",
      "price_range": "30-60 THB"
    }
  ],
  "recommended_restaurants": [
    {
      "name": "Thip Samai",
      "type": "local institution", 
      "specialties": ["Pad Thai"],
      "local_popularity": "Famous for original Pad Thai recipe since 1966"
    }
  ],
  "food_districts": [
    {
      "area_name": "Chinatown (Yaowarat)",
      "specialty": "Street food and Chinese-Thai fusion",
      "best_time_to_visit": "Evening after 6 PM"
    }
  ]
}
```

## 🛠️ Advanced Features

### **Smart Allergen Analysis**
- **Safety Scoring**: 1-10 rating for allergen-friendliness
- **Risk Categories**: Safe, risky, unclear dishes
- **Communication Help**: How to explain allergies to staff

### **Cultural Intelligence**
- **Regional Specialties**: Location-specific dishes
- **Seasonal Awareness**: What's in season
- **Local Ingredients**: Unique regional components
- **Fusion Elements**: Cultural influences in dishes

### **Value Analysis**
- **Best Value Picks**: Maximum taste for money
- **Splurge Worthy**: Expensive but exceptional dishes
- **Hidden Costs**: Service charges and extras
- **Portion Expectations**: Local serving sizes

## 🌍 Supported Locations

Works globally with deep knowledge of:
- **Asian Cuisines**: Thai, Japanese, Chinese, Korean, Vietnamese, Indian
- **European Cuisines**: Italian, French, Spanish, German, Greek
- **Latin American**: Mexican, Peruvian, Argentinian, Brazilian
- **Middle Eastern**: Lebanese, Turkish, Moroccan, Persian
- **African**: Ethiopian, South African, West African
- **And many more...**

## 🔍 Troubleshooting

### **Common Issues:**

1. **"Gemini 2.0 Pro model not found"**
   - Update: `pip install --upgrade google-generativeai`
   - The system will fallback to available models automatically

2. **"Discovery mode not working"**
   - Ensure `location` parameter is provided in format "City, Country"
   - Set `discovery_mode: true`

3. **"Enhanced analysis unavailable"**
   - Check GEMINI_API_KEY is valid
   - Verify internet connection
   - System will fallback to basic OCR

### **Optimization Tips:**

- **Image Quality**: Higher resolution = better analysis
- **Location Specificity**: "Tokyo, Japan" better than just "Japan"  
- **Preference Details**: "authentic street food" better than "traditional"
- **Context Matters**: Include dining occasion (business, casual, date)

## 🚀 Quick Test

Run this to test your enhanced setup:

```bash
# Start your MCP server
python mcp_starter.py

# Test cuisine discovery (no image needed)
menu_intelligence location="Paris, France" discovery_mode=true preferences=["authentic", "pastries"]

# Test menu analysis (with image)
menu_intelligence menu_image_base64="[your_image]" location="Rome, Italy" allergies=["gluten"]
```

## 🎯 Pro Tips

1. **Combine Both Modes**: Use discovery first to find restaurants, then analyze their menus
2. **Location Context**: Always include location for better cultural insights
3. **Specific Preferences**: The more specific, the better the recommendations
4. **Ask Follow-ups**: Use the cultural notes to ask informed questions to staff

## 🔐 Security & Privacy

- **API Keys**: Keep GEMINI_API_KEY secure
- **Image Data**: Images are processed securely through Google's APIs
- **No Storage**: Images are not stored after analysis
- **Rate Limits**: Gemini API has generous free tier limits
