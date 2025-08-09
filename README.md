# Puch AI - Intelligent Travel Assistant

This is a starter template for creating your own Model Context Protocol (MCP) server that works with Puch AI, supercharged with a suite of travel tools and an intelligent orchestrating agent.

## What is MCP?

MCP (Model Context Protocol) allows AI assistants like Puch to connect to external tools and data sources safely. Think of it as giving your AI extra superpowers without compromising security.

## What's Included in This Starter?

This starter has been transformed into a powerful **AI Travel Assistant** with a suite of tools designed to help you navigate the world with confidence.

### 🧠 Intelligent Travel Agent
The core of this assistant. It understands complex travel requests, automatically selects the right tools for the job, and provides a unified, actionable plan.

### �️ Core Travel Tools
- **Cultural Context Predictor**: Get crucial do's and don'ts, etiquette, and behavioral insights for any country.
- **Local Social Dynamics Decoder**: Understand local norms and get advice based on your specific location and time of day.
- **Emergency Phrase Generator**: Instantly get essential emergency phrases in the local language.
- **Menu Intelligence**: Analyze a photo of a menu to get translations, allergen warnings, and recommendations. Powered by Google Gemini Vision with a Tesseract OCR fallback.
- **Local Navigation with Social Intelligence**: Plan walking, driving, or transit routes with added safety and social context.
- **Travel Memory Archive**: Save and retrieve your travel experiences, photos, and AI-generated insights.

## Quick Setup Guide

### Step 1: Install Dependencies

First, make sure you have Python 3.11 or higher installed.

```bash
# Create virtual environment (if you haven't already)
python -m venv .venv

# Activate the environment
# On Windows (Git Bash/WSL):
source .venv/bin/activate
# On macOS/Linux:
source .venv/bin/activate

# Install all required packages
pip install -r requirements.txt 
# Or if using uv:
# uv sync
```
*Note: A `pyproject.toml` is provided, which can be used with tools like `uv` or `pip`.*

### Step 2: Set Up Environment Variables

Create a `.env` file in the project root (`Puch-AI/.env`):

```bash
# Create an empty .env file
touch .env
```

Then edit `.env` and add your details:

```env
# Required by Puch AI
AUTH_TOKEN="your_secret_token_here"
MY_NUMBER="Your Number with countrycode attached(eg: 918366XXXXXX)"

# For Menu Intelligence Tool (Optional but Recommended)
GEMINI_API_KEY="your_google_ai_gemini_api_key"

# For Tesseract OCR Fallback (Optional)
# Only needed if 'tesseract' is not in your system's PATH
# TESSERACT_CMD="/path/to/your/tesseract"
```

**Important Notes:**
- `AUTH_TOKEN`: Your secret token for authentication. Keep it safe!
- `MY_NUMBER`: Your WhatsApp number in the format `{country_code}{number}`.
- `GEMINI_API_KEY`: Highly recommended for the best menu analysis results. Get one from [Google AI Studio](https://aistudio.google.com/app/apikey).
- `TESSERACT_CMD`: Only necessary if the Tesseract OCR engine isn't globally installed and accessible from your terminal.

### Step 3: Run the Server

```bash
# From the root Puch-AI directory
python mcp-bearer-token/mcp_starter.py
```

You'll see: `🚀 Starting MCP server on http://0.0.0.0:8086`

### Step 4: Make It Public (Required by Puch)

Since Puch needs to access your server over HTTPS, you need to expose your local server.

#### Using ngrok (Recommended)

1. **Install ngrok:**
   Download from https://ngrok.com/download

2. **Get your authtoken:**
   - Go to https://dashboard.ngrok.com/get-started/your-authtoken
   - Copy your authtoken
   - Run: `ngrok config add-authtoken YOUR_AUTHTOKEN`

3. **Start the tunnel:**
   ```bash
   ngrok http 8086
   ```
You will get a public HTTPS URL. This is what you'll use to connect with Puch AI.

## How to Use the Intelligent Travel Agent

Once your server is running and exposed via ngrok, you can interact with the agent using `curl` or any HTTP client.

### Example Request

Here's an example of a complex travel request that uses multiple tools at once. Replace `YOUR_NGROK_URL` and `YOUR_AUTH_TOKEN` accordingly.

```bash
curl -X POST YOUR_NGROK_URL/mcp/tools/intelligent_travel_agent \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "travel_request": "I am planning to visit Tokyo tomorrow from the USA. I want to walk from Shibuya to Harajuku and need cultural etiquette tips and emergency phrases.",
    "user_id": "demo_user",
    "home_country": "USA",
    "current_location": "Shibuya Station"
  }'
```

### Expected Response

The agent will analyze the request, call the necessary tools in the background, and return a synthesized plan:

```json
{
  "ok": true,
  "data": {
    "request_analysis": {
      "identified_needs": {
        "cultural_guidance": true,
        "navigation_help": true,
        "emergency_preparation": true
      }
    },
    "orchestrated_results": {
      "cultural_context": { "... (cultural insights) ..." },
      "navigation": { "... (route steps) ..." },
      "emergency_phrases": { "... (translated phrase) ..." }
    },
    "summary": "🏛️ Cultural Tips: ... \n🗺️ Route: ... \n🆘 Emergency Phrase: ...",
    "next_steps": [ "📸 Take a photo of any menu for analysis..." ]
  }
}
```

## How to Connect with Puch AI

1. **[Open Puch AI](https://wa.me/+919998881729)** in your browser.
2. **Start a new conversation.**
3. **Use the connect command** with your public ngrok URL:
   `/connect YOUR_NGROK_URL/mcp`
4. **Provide your AUTH_TOKEN** when prompted.

Once connected, you can interact with your travel assistant directly through Puch AI!
   ```
   /mcp connect https://your-domain.ngrok.app/mcp your_secret_token_here
   ```

### Debug Mode

To get more detailed error messages:

```
/mcp diagnostics-level debug
```

## Customizing the Starter

### Adding New Tools

1. **Create a new tool function:**
   ```python
   @mcp.tool(description="Your tool description")
   async def your_tool_name(
       parameter: Annotated[str, Field(description="Parameter description")]
   ) -> str:
       # Your tool logic here
       return "Tool result"
   ```

2. **Add required imports** if needed


## 📚 **Additional Documentation Resources**

### **Official Puch AI MCP Documentation**
- **Main Documentation**: https://puch.ai/mcp
- **Protocol Compatibility**: Core MCP specification with Bearer & OAuth support
- **Command Reference**: Complete MCP command documentation
- **Server Requirements**: Tool registration, validation, HTTPS requirements

### **Technical Specifications**
- **JSON-RPC 2.0 Specification**: https://www.jsonrpc.org/specification (for error handling)
- **MCP Protocol**: Core protocol messages, tool definitions, authentication

### **Supported vs Unsupported Features**

**✓ Supported:**
- Core protocol messages
- Tool definitions and calls
- Authentication (Bearer & OAuth)
- Error handling

**✗ Not Supported:**
- Videos extension
- Resources extension
- Prompts extension

## Getting Help

- **Join Puch AI Discord:** https://discord.gg/VMCnMvYx
- **Check Puch AI MCP docs:** https://puch.ai/mcp
- **Puch WhatsApp Number:** +91 99988 81729

---

**Happy coding! 🚀**

Use the hashtag `#BuildWithPuch` in your posts about your MCP!
