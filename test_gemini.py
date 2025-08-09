import google.generativeai as genai
import os

# Load API key from environment
api_key = "AIzaSyCmf2hYWYn0q4frv1J_55dXvmt3edDrNS4"
print(f"🔑 Testing API key: {api_key[:10]}...")

try:
    genai.configure(api_key=api_key)
    print("✅ API configured successfully")
    
    # Test available models
    models = genai.list_models()
    available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
    print(f"📝 Available models: {available_models}")
    
    # Test with gemini-2.5-pro
    print("🧪 Testing gemini-2.5-pro...")
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content("Hello, can you see this text?")
        print(f"✅ gemini-2.5-pro works! Response: {response.text[:50]}...")
    except Exception as e:
        print(f"❌ gemini-2.5-pro failed: {e}")
        
        # Try gemini-1.5-pro as fallback
        print("🧪 Testing gemini-1.5-pro as fallback...")
        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content("Hello, can you see this text?")
            print(f"✅ gemini-1.5-pro works! Response: {response.text[:50]}...")
        except Exception as e2:
            print(f"❌ gemini-1.5-pro also failed: {e2}")

except Exception as e:
    print(f"❌ API configuration failed: {e}")
