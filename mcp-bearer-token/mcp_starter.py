import asyncio
from typing import Annotated, Literal
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
import httpx
import readabilipy
import time
import uuid
from datetime import datetime, timedelta

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of URLs.
        (Using DuckDuckGo because Google blocks most programmatic scraping.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]

# --- Simple JSON helper ---
def ok(data: dict | list | str | int | float | None = None):
    return {"ok": True, "data": data, "ts": datetime.utcnow().isoformat()}

def err(message: str, code: str = "error"):
    return {"ok": False, "error": {"code": code, "message": message}, "ts": datetime.utcnow().isoformat()}

# --- In-memory stores (can be swapped for a DB) ---
MEMORIES: dict[str, list[dict]] = {}
USAGE: dict[str, int] = {}

# --- MCP Server Setup ---
mcp = FastMCP(
    "Puch Travel MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    # Return phone number in {country_code}{number}
    return MY_NUMBER

# --- Usage tracking ---
USAGE: dict[str, int] = {}

def track_tool_usage(tool_name: str):
    USAGE[tool_name] = USAGE.get(tool_name, 0) + 1

# --- Travel Tools ---

CULTURAL_CONTEXT_DESCRIPTION = RichToolDescription(
    description="Cultural Context Predictor: etiquette, taboos, behavioral insights between a home country and a destination.",
    use_when="User plans to visit a destination and needs cultural do's and don'ts.",
)

@mcp.tool(description=CULTURAL_CONTEXT_DESCRIPTION.model_dump_json())
async def cultural_context_predictor(
    home_country: Annotated[str, Field(description="ISO country name or code of the user's home country")],
    destination_country: Annotated[str, Field(description="ISO country name or code of the destination country")],
    traveler_profile: Annotated[str | None, Field(description="Interests or context to tailor guidance")] = None,
) -> dict:
    track_tool_usage("cultural_context_predictor")
    # Placeholder logic; in real deployment you might call an LLM with curated prompts
    insights = {
        "greetings": "Handshake is common; slight bow may be appreciated depending on region.",
        "taboos": [
            "Avoid loud public behavior",
            "Respect religious sites dress codes",
        ],
        "etiquette": [
            "Use polite forms of address",
            "Punctuality is valued",
        ],
        "business": [
            "Exchange business cards with both hands",
        ],
    }
    return ok({
        "home_country": home_country,
        "destination_country": destination_country,
        "traveler_profile": traveler_profile,
        "insights": insights,
    })

LOCAL_SOCIAL_DESCRIPTION = RichToolDescription(
    description="Local Social Dynamics Decoder: advice based on time, location, and norms.",
    use_when="User needs to behave appropriately in a specific local context.",
)

@mcp.tool(description=LOCAL_SOCIAL_DESCRIPTION.model_dump_json())
async def local_social_dynamics_decoder(
    city: Annotated[str, Field(description="City or locality name")],
    country: Annotated[str, Field(description="Country name or code")],
    time_of_day: Annotated[str, Field(description="Morning/Afternoon/Evening/Night or 24h time")],
    context: Annotated[str | None, Field(description="Situational context, e.g., market, metro, nightlife")] = None,
) -> dict:
    track_tool_usage("local_social_dynamics_decoder")
    advice = [
        "Stay aware of personal space in crowded areas",
        "In markets, friendly bargaining is common",
        "Avoid displaying valuables",
    ]
    return ok({"city": city, "country": country, "time_of_day": time_of_day, "context": context, "advice": advice})

## Removed: religious_and_festival_calendar (not needed)

## Removed: crowd_sourced_safety_intel (not needed)

EMERGENCY_PHRASE_DESCRIPTION = RichToolDescription(
    description="Emergency Phrase Generator: respectful phrases in local language.",
    use_when="User needs quick emergency phrases.",
)

@mcp.tool(description=EMERGENCY_PHRASE_DESCRIPTION.model_dump_json())
async def emergency_phrase_generator(
    intent: Annotated[str, Field(description="Help intent, e.g., need_doctor, lost, police, embassy")],
    language: Annotated[str, Field(description="Target language name or code")],
    politeness_level: Annotated[str | None, Field(description="formal/informal/neutral")] = "formal",
) -> dict:
    phrases = {
        "need_doctor": {
            "en": "I need a doctor, please.",
        },
        "police": {
            "en": "Please call the police.",
        },
        "lost": {
            "en": "I'm lost, can you help me?",
        },
    }
    chosen = phrases.get(intent, {"en": "Please help me."})
    # Placeholder translation
    return ok({"intent": intent, "language": language, "phrase": chosen.get("en")})

## Removed: predictive_risk_assessment (not needed)

## Removed: digital_safety_net (not needed)

## Removed: contextual_visual_storytelling (not needed)

MENU_INTEL_DESCRIPTION = RichToolDescription(
    description="Menu Intelligence: analyze a menu photo for allergens, recommendations, and etiquette.",
    use_when="User shares a menu photo and dietary preferences.",
)

@mcp.tool(description=MENU_INTEL_DESCRIPTION.model_dump_json())
async def menu_intelligence(
    menu_image_base64: Annotated[str, Field(description="Base64-encoded image of the menu")],
    allergies: Annotated[list[str] | None, Field(description="List of allergens to avoid")]=None,
    preferences: Annotated[list[str] | None, Field(description="Cuisine/diet preferences, e.g., vegetarian, spicy")]=None,
    language: Annotated[str | None, Field(description="Language for output")]="en",
) -> dict:
    import base64
    import io
    import re
    import json
    from PIL import Image
    
    async def extract_with_gemini(image_bytes, allergies, preferences):
        """Extract menu information using Gemini Vision API"""
        try:
            import google.generativeai as genai
            
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise Exception("GEMINI_API_KEY not found in environment")
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-pro')
            
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Create comprehensive prompt for menu analysis
            allergen_text = f"Allergies to check for: {', '.join(allergies)}" if allergies else "No specific allergies to check"
            preference_text = f"Dietary preferences: {', '.join(preferences)}" if preferences else "No specific preferences"
            
            prompt = f"""
Analyze this menu image and extract information in the following JSON format:

{{
    "extracted_text": "Full text content from the menu",
    "menu_items": [
        {{
            "name": "dish name",
            "price": "price if visible",
            "description": "description if available",
            "category": "appetizer/main/dessert/drink"
        }}
    ],
    "detected_prices": ["list of all prices found"],
    "recommendations": [
        "recommended items based on preferences with explanations"
    ],
    "allergen_warnings": [
        "specific allergen warnings based on menu content"
    ],
    "cuisine_type": "type of cuisine (Italian, Chinese, etc.)",
    "price_range": "budget/moderate/expensive",
    "vegetarian_options": ["list of vegetarian dishes if any"],
    "spicy_dishes": ["list of spicy dishes if any"]
}}

User Context:
{allergen_text}
{preference_text}

Instructions:
1. Extract ALL visible text from the menu
2. Identify individual menu items with prices where visible
3. Look for allergen ingredients in item descriptions
4. Recommend items based on user preferences
5. Categorize dishes appropriately
6. Provide helpful warnings for allergies
7. Be thorough but concise
8. If text is unclear, indicate uncertainty

Please analyze this menu image:
"""
            
            # Generate response
            response = model.generate_content([prompt, pil_image])
            
            if not response.text:
                raise Exception("Gemini returned empty response")
            
            # Try to parse JSON response
            try:
                # Clean the response text (remove markdown code blocks if present)
                clean_text = response.text.strip()
                if clean_text.startswith('```json'):
                    clean_text = clean_text[7:]
                if clean_text.endswith('```'):
                    clean_text = clean_text[:-3]
                clean_text = clean_text.strip()
                
                result = json.loads(clean_text)
                return result, "Gemini Vision API"
                
            except json.JSONDecodeError:
                # If JSON parsing fails, create structured response from text
                response_text = response.text
                return {
                    "extracted_text": response_text,
                    "menu_items": [],
                    "detected_prices": [],
                    "recommendations": [response_text[:200] + "..."],
                    "allergen_warnings": [],
                    "cuisine_type": "Unknown",
                    "price_range": "Unknown"
                }, "Gemini Vision API (text mode)"
                
        except Exception as e:
            raise Exception(f"Gemini API failed: {str(e)}")
    
    async def extract_with_tesseract(image_bytes):
        """Fallback OCR using Tesseract with OpenCV preprocessing (lightweight)."""
        try:
            import numpy as np
            import cv2
            import pytesseract

            # Configure tesseract path if specified in environment
            tesseract_cmd = os.environ.get("TESSERACT_CMD")
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

            # Load image
            pil_image = Image.open(io.BytesIO(image_bytes))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            processed_image = Image.fromarray(thresh)

            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?$€£¥₹₽₩₪₨₡₵₹()-/:;&@#%+'
            extracted_text = pytesseract.image_to_string(processed_image, config=custom_config).strip()

            if not extracted_text:
                raise Exception("No text detected by Tesseract")

            return {
                "extracted_text": extracted_text,
                "menu_items": [],
                "detected_prices": [],
                "recommendations": ["Basic OCR text extraction - manual analysis needed"],
                "allergen_warnings": [],
                "cuisine_type": "Unknown",
                "price_range": "Unknown"
            }, "Tesseract OCR (fallback)"

        except Exception as e:
            raise Exception(f"Tesseract OCR failed: {str(e)}")
    
    try:
        # Decode and validate image
        try:
            # Handle data URL prefix
            if menu_image_base64.startswith('data:image'):
                menu_image_base64 = menu_image_base64.split(',')[1]
            
            menu_image_base64 = menu_image_base64.strip()
            image_bytes = base64.b64decode(menu_image_base64)
            
            # Validate image
            pil_image = Image.open(io.BytesIO(image_bytes))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                
        except Exception as decode_error:
            return ok({
                "language": language,
                "error": f"Failed to decode image: {str(decode_error)}",
                "extracted_text": "Image decode failed",
                "recommendations": ["Please check the image format and try again"],
                "allergen_warnings": [],
                "etiquette": ["Ask staff for menu assistance"],
                "translation": "Image processing failed",
                "success": False
            })
        
        # Try Gemini first, then fallback to Tesseract OCR
        result = {}
        ocr_method = ""

        try:
            # Method 1: Gemini Vision API (best accuracy and intelligence)
            result, ocr_method = await extract_with_gemini(image_bytes, allergies, preferences)
        except Exception as gemini_error:
            # Method 2: Tesseract OCR fallback (lightweight)
            try:
                result, ocr_method = await extract_with_tesseract(image_bytes)

                # If Tesseract worked, perform simple post-processing
                if result.get("extracted_text"):
                    text = result["extracted_text"]
                    lines = [line.strip() for line in text.split('\n') if line.strip()]

                    # Basic price detection
                    price_patterns = [
                        r'[\$€£¥₹]\s*\d+[.,]?\d*',
                        r'\d+[.,]\d{2}(?!\d)',
                        r'Rs\.?\s*\d+',
                    ]

                    detected_prices = []
                    menu_items = []

                    for line in lines:
                        has_price = any(re.search(pattern, line, re.IGNORECASE) for pattern in price_patterns)
                        if has_price:
                            for pattern in price_patterns:
                                matches = re.findall(pattern, line, re.IGNORECASE)
                                detected_prices.extend(matches)
                            menu_items.append({"name": line, "price": "", "description": "", "category": "unknown"})
                        elif len(line) > 3:
                            menu_items.append({"name": line, "price": "", "description": "", "category": "unknown"})

                    result["detected_prices"] = list(set(detected_prices))[:10]
                    result["menu_items"] = menu_items[:15]

                    # Basic allergen checking
                    if allergies:
                        allergen_warnings = []
                        text_lower = text.lower()
                        allergen_keywords = {
                            'nuts': ['nut', 'almond', 'walnut', 'peanut', 'cashew'],
                            'dairy': ['milk', 'cheese', 'butter', 'cream', 'yogurt'],
                            'gluten': ['wheat', 'bread', 'pasta', 'flour', 'gluten'],
                            'shellfish': ['shrimp', 'crab', 'lobster', 'shellfish'],
                            'eggs': ['egg', 'mayo', 'mayonnaise'],
                            'soy': ['soy', 'tofu', 'soybean']
                        }

                        for allergy in allergies:
                            keywords = allergen_keywords.get(allergy.lower(), [allergy.lower()])
                            if any(keyword in text_lower for keyword in keywords):
                                allergen_warnings.append(f"⚠️ {allergy.title()} may be present")

                        result["allergen_warnings"] = allergen_warnings

            except Exception as tesseract_error:
                return ok({
                    "language": language,
                    "error": "All OCR methods failed",
                    "extracted_text": "OCR processing failed",
                    "recommendations": ["Please try a clearer image or install OCR dependencies"],
                    "allergen_warnings": [],
                    "etiquette": ["Ask staff for menu translation"],
                    "translation": "OCR unavailable",
                    "ocr_method": "All methods failed",
                    "success": False,
                    "error_details": {
                        "gemini": str(gemini_error),
                        "tesseract": str(tesseract_error)
                    }
                })
        
        # Process Gemini results or enhance EasyOCR results
        if not result.get("extracted_text") or len(result.get("extracted_text", "").strip()) < 3:
            return ok({
                "language": language,
                "extracted_text": "No meaningful text detected",
                "recommendations": ["Image may be unclear or contain no readable text"],
                "allergen_warnings": [],
                "etiquette": ["Point to menu items when ordering"],
                "translation": "No text to analyze",
                "ocr_method": ocr_method,
                "success": False
            })
        
        # Enhanced recommendations if Gemini didn't provide them
        if not result.get("recommendations") and preferences:
            recommendations = []
            menu_items = result.get("menu_items", [])
            
            for pref in preferences:
                pref_lower = pref.lower()
                matching_items = []
                
                for item in menu_items:
                    item_name = item.get("name", "") if isinstance(item, dict) else str(item)
                    if pref_lower in item_name.lower():
                        matching_items.append(item_name)
                
                if matching_items:
                    recommendations.extend([f"✓ {item} (matches {pref})" for item in matching_items[:2]])
            
            if not recommendations:
                recommendations = ["Ask server for recommendations based on your preferences"]
            
            result["recommendations"] = recommendations
        
        # Cultural etiquette suggestions
        etiquette = [
            "Point to menu items if there's a language barrier",
            "Ask server 'What do you recommend?' in local language",
            "Check if service charge is included before tipping",
            "Say 'thank you' in the local language after ordering"
        ]
        
        # Prepare final result
        final_result = {
            "language": language,
            "extracted_text": result.get("extracted_text", "")[:800] + "..." if len(result.get("extracted_text", "")) > 800 else result.get("extracted_text", ""),
            "menu_items": result.get("menu_items", [])[:15],
            "detected_prices": result.get("detected_prices", [])[:10],
            "recommendations": result.get("recommendations", [])[:6],
            "allergen_warnings": result.get("allergen_warnings", []),
            "cuisine_type": result.get("cuisine_type", "Unknown"),
            "price_range": result.get("price_range", "Unknown"),
            "etiquette": etiquette,
            "translation": f"Menu analyzed using {ocr_method}",
            "ocr_method": ocr_method,
            "success": True
        }
        
        track_tool_usage("menu_intelligence")
        return ok(final_result)
        
    except Exception as e:
        return ok({
            "language": language,
            "error": f"Menu analysis failed: {str(e)}",
            "extracted_text": "Processing failed",
            "recommendations": ["Please try again with a clearer image"],
            "allergen_warnings": ["Unable to detect allergens - ask staff"],
            "etiquette": ["Use translation app as backup"],
            "translation": "Analysis failed",
            "ocr_method": "Error occurred",
            "success": False
        })

NAV_SOCIAL_DESCRIPTION = RichToolDescription(
    description="Local Navigation with Social Intelligence: safety and tourist-awareness context for routes.",
    use_when="User wants to navigate and avoid unsafe or overly crowded segments.",
)

@mcp.tool(description=NAV_SOCIAL_DESCRIPTION.model_dump_json())
async def local_navigation_social_intelligence(
    origin: Annotated[str, Field(description="Start location (address or lat,lng)")],
    destination: Annotated[str, Field(description="End location (address or lat,lng)")],
    mode: Annotated[Literal["walking", "driving", "transit"], Field(description="Travel mode")]="walking",
    caution_preference: Annotated[Literal["low", "medium", "high"], Field(description="How cautious to be")]="medium",
) -> dict:
    # Placeholder routing; optionally use Google Directions API with GOOGLE_MAPS_API_KEY
    steps = [
        {"instruction": "Head north 200m", "risk": "low"},
        {"instruction": "Through market lane", "risk": "medium", "note": "crowded at evenings"},
        {"instruction": "Arrive at destination", "risk": "low"},
    ]
    score = 0.15 if caution_preference == "high" else 0.25
    return ok({"mode": mode, "safety_score": score, "steps": steps})

## Removed: accent_and_dialect_training (not needed)

## Removed: two_way_live_voice_interpreter (not needed)

## Removed: message_relay_audio_translation (not needed)

## Removed: expense_cultural_context (not needed)

TRAVEL_MEMORY_DESCRIPTION = RichToolDescription(
    description="Travel Memory Archive: save cultural experiences, photos, and AI insights.",
    use_when="User wants to store and retrieve travel memories.",
    side_effects="Stores memories in-memory by user id.",
)

@mcp.tool(description=TRAVEL_MEMORY_DESCRIPTION.model_dump_json())
async def travel_memory_archive(
    action: Annotated[Literal["save", "list"], Field(description="Save a memory or list memories")],
    user_id: Annotated[str, Field(description="User identifier")],
    title: Annotated[str | None, Field(description="Memory title")]=None,
    text: Annotated[str | None, Field(description="Narrative text")]=None,
    image_base64: Annotated[str | None, Field(description="Optional photo base64")]=None,
    tags: Annotated[list[str] | None, Field(description="Optional tags")]=None,
) -> dict:
    if action == "save":
        if not (title or text or image_base64):
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Provide at least title, text, or image_base64 to save"))
        item = {
            "id": str(uuid.uuid4()),
            "ts": datetime.utcnow().isoformat(),
            "title": title,
            "text": text,
            "image_base64": image_base64,
            "tags": tags or [],
        }
        MEMORIES.setdefault(user_id, []).append(item)
        return ok({"saved": item})
    # list
    return ok({"memories": list(reversed(MEMORIES.get(user_id, [])))})

INTELLIGENT_AGENT_DESCRIPTION = RichToolDescription(
    description="Intelligent Travel Agent: analyzes complex travel requests and orchestrates multiple tools to provide comprehensive travel assistance.",
    use_when="User has complex travel planning needs, multi-step itineraries, or wants unified travel advice.",
    side_effects="May call multiple underlying tools and save memories based on request context.",
)

@mcp.tool(description=INTELLIGENT_AGENT_DESCRIPTION.model_dump_json())
async def intelligent_travel_agent(
    travel_request: Annotated[str, Field(description="Natural language travel request or question")],
    user_id: Annotated[str, Field(description="User identifier for memory storage")],
    home_country: Annotated[str | None, Field(description="User's home country (if known)")] = None,
    current_location: Annotated[str | None, Field(description="Current location (city, country)")] = None,
    dietary_restrictions: Annotated[list[str] | None, Field(description="Known allergies or dietary preferences")] = None,
) -> dict:
    """
    Intelligent agent that analyzes travel requests and orchestrates appropriate tools.
    """
    track_tool_usage("intelligent_travel_agent")
    
    try:
        # Parse the request to determine intent and extract key information
        request_lower = travel_request.lower()
        orchestrated_results = {}
        
        # Extract locations mentioned in the request
        def extract_locations():
            locations = []
            # Simple keyword extraction - in production, use NER
            location_indicators = ["to ", "in ", "from ", "visit ", "going to ", "traveling to "]
            for indicator in location_indicators:
                if indicator in request_lower:
                    parts = request_lower.split(indicator)
                    if len(parts) > 1:
                        potential_location = parts[1].split()[0:3]  # Take next 1-3 words
                        locations.append(" ".join(potential_location))
            return locations
        
        locations = extract_locations()
        destination = locations[0] if locations else current_location
        
        # Determine which tools to use based on request content
        needs_cultural_context = any(word in request_lower for word in [
            "culture", "etiquette", "customs", "do's", "don'ts", "taboo", "manners", "behavior"
        ])
        
        needs_navigation = any(word in request_lower for word in [
            "route", "directions", "navigate", "walk", "drive", "transit", "from", "to", "path"
        ])
        
        needs_emergency_phrases = any(word in request_lower for word in [
            "emergency", "help", "lost", "phrase", "language", "translate", "say"
        ])
        
        needs_social_dynamics = any(word in request_lower for word in [
            "local", "social", "crowd", "busy", "time", "when", "market", "area"
        ])
        
        needs_menu_help = any(word in request_lower for word in [
            "menu", "food", "restaurant", "eat", "dining", "allergies", "vegetarian"
        ])
        
        wants_to_save_memory = any(word in request_lower for word in [
            "remember", "save", "memory", "experience", "note"
        ])
        
        # Prepare tool suggestions and structured guidance instead of direct execution
        suggested_tools = []
        guidance = {}
        
        # 1. Cultural Context Analysis
        if needs_cultural_context and home_country and destination:
            suggested_tools.append({
                "tool": "cultural_context_predictor",
                "parameters": {
                    "home_country": home_country,
                    "destination_country": destination,
                    "traveler_profile": f"Extracted from request: {travel_request[:100]}"
                },
                "reason": "Get cultural etiquette and taboos guidance"
            })
            
            # Provide immediate basic guidance
            guidance["cultural_tips"] = {
                "general_advice": [
                    "Research local customs and dress codes",
                    "Learn basic greetings and polite phrases",
                    "Be respectful of religious and cultural sites",
                    "Observe and follow local social norms"
                ],
                "suggested_research": [
                    "Business etiquette if traveling for work",
                    "Dining customs and table manners", 
                    "Tipping practices and gift-giving customs"
                ]
            }
        
        # 2. Navigation Planning
        if needs_navigation:
            # Extract origin and destination from request
            origin = current_location or "current location"
            nav_destination = destination or "destination"
            
            # Determine travel mode
            mode = "walking"
            if any(word in request_lower for word in ["drive", "car", "driving"]):
                mode = "driving"
            elif any(word in request_lower for word in ["transit", "train", "bus", "metro"]):
                mode = "transit"
            
            suggested_tools.append({
                "tool": "local_navigation_social_intelligence",
                "parameters": {
                    "origin": origin,
                    "destination": nav_destination,
                    "mode": mode,
                    "caution_preference": "medium"
                },
                "reason": f"Get safe {mode} directions with social context"
            })
            
            # Provide immediate basic guidance
            guidance["navigation_tips"] = {
                "mode": mode,
                "general_advice": [
                    "Download offline maps before traveling",
                    "Keep emergency contacts readily available",
                    "Stay aware of your surroundings",
                    "Have backup navigation methods ready"
                ],
                "safety_reminders": [
                    "Avoid displaying expensive items",
                    "Trust your instincts about unsafe areas",
                    "Keep someone informed of your itinerary"
                ]
            }
        
        # 3. Emergency Preparedness
        if needs_emergency_phrases:
            # Determine intent
            intent = "lost"  # default
            if any(word in request_lower for word in ["doctor", "medical", "sick"]):
                intent = "need_doctor"
            elif any(word in request_lower for word in ["police", "danger", "unsafe"]):
                intent = "police"
            
            # Determine language from destination
            language = "english"  # default
            if destination:
                # Simple mapping - in production, use proper language detection
                lang_map = {
                    "japan": "japanese", "tokyo": "japanese", "osaka": "japanese",
                    "france": "french", "paris": "french",
                    "spain": "spanish", "madrid": "spanish",
                    "germany": "german", "berlin": "german",
                    "italy": "italian", "rome": "italian"
                }
                for place, lang in lang_map.items():
                    if place in destination.lower():
                        language = lang
                        break
            
            suggested_tools.append({
                "tool": "emergency_phrase_generator",
                "parameters": {
                    "intent": intent,
                    "language": language,
                    "politeness_level": "formal"
                },
                "reason": f"Learn essential {language} phrases for {intent} situations"
            })
            
            # Provide immediate basic guidance
            guidance["emergency_preparation"] = {
                "language": language,
                "essential_info": [
                    "Save local emergency numbers in your phone",
                    "Keep your embassy contact information handy",
                    "Have your accommodation address written down",
                    "Carry identification and important documents"
                ],
                "basic_phrases_needed": [
                    "Help/Emergency", "I'm lost", "Call police/doctor",
                    "I don't speak [local language]", "Thank you"
                ]
            }
        
        # 4. Social Dynamics Awareness
        if needs_social_dynamics and destination:
            # Extract time context
            time_context = "daytime"
            if any(t in request_lower for t in ["evening", "night", "late"]):
                time_context = "evening"
            elif any(t in request_lower for t in ["morning", "early"]):
                time_context = "morning"
            
            suggested_tools.append({
                "tool": "local_social_dynamics_decoder",
                "parameters": {
                    "city": destination,
                    "country": destination,  # Simplified - in production, separate city/country
                    "time_of_day": time_context,
                    "context": "General travel context"
                },
                "reason": f"Understand {destination} social norms for {time_context}"
            })
            
            # Provide immediate basic guidance
            guidance["social_awareness"] = {
                "time_context": time_context,
                "general_advice": [
                    "Observe local behavior and follow suit",
                    "Be respectful of personal space norms",
                    "Learn appropriate greeting customs",
                    "Understand local communication styles"
                ]
            }
        
        # 5. Memory Management
        if wants_to_save_memory:
            suggested_tools.append({
                "tool": "travel_memory_archive",
                "parameters": {
                    "action": "save",
                    "user_id": user_id,
                    "title": f"Travel Plan: {travel_request[:50]}...",
                    "text": f"Travel request: {travel_request}",
                    "tags": ["intelligent_agent", "travel_planning"]
                },
                "reason": "Save this travel plan for future reference"
            })
        
        # Add menu analysis suggestion if dietary restrictions mentioned
        if dietary_restrictions or any(word in request_lower for word in ["food", "restaurant", "eat", "menu", "dining"]):
            guidance["dining_assistance"] = {
                "dietary_info": dietary_restrictions or ["No specific restrictions mentioned"],
                "recommendations": [
                    "Use the menu_intelligence tool when you find restaurants",
                    "Take photos of menus for allergen analysis",
                    "Learn how to communicate dietary restrictions in local language",
                    "Research local cuisine that matches your preferences"
                ]
            }
        
        # Synthesize unified response
        unified_response = {
            "request_analysis": {
                "original_request": travel_request,
                "detected_locations": locations,
                "identified_needs": {
                    "cultural_guidance": needs_cultural_context,
                    "navigation_help": needs_navigation,
                    "emergency_preparation": needs_emergency_phrases,
                    "social_awareness": needs_social_dynamics,
                    "dining_assistance": needs_menu_help,
                    "memory_keeping": wants_to_save_memory
                }
            },
            "suggested_tools": suggested_tools,
            "immediate_guidance": guidance
        }
        
        # Create a human-friendly summary
        summary_parts = []
        
        if "cultural_tips" in guidance:
            summary_parts.append("🏛️ **Cultural Preparation**: Research local customs, greetings, and etiquette")
        
        if "navigation_tips" in guidance:
            nav_info = guidance["navigation_tips"]
            summary_parts.append(f"🗺️ **Navigation**: Plan {nav_info['mode']} route with safety considerations")
        
        if "emergency_preparation" in guidance:
            emerg_info = guidance["emergency_preparation"]
            summary_parts.append(f"🆘 **Emergency Ready**: Learn key {emerg_info['language']} phrases and safety info")
        
        if "social_awareness" in guidance:
            social_info = guidance["social_awareness"]
            summary_parts.append(f"👥 **Social Awareness**: Understand {social_info['time_context']} social norms")
        
        if "dining_assistance" in guidance:
            dietary_info = guidance["dining_assistance"]["dietary_info"]
            summary_parts.append(f"🍽️ **Dining**: Prepared for {', '.join(dietary_info)} dietary needs")
        
        unified_response["summary"] = "\n".join(summary_parts) if summary_parts else "✈️ Travel assistance provided based on your request."
        
        # Add next steps for tools to use
        next_steps = []
        for tool_suggestion in suggested_tools:
            tool_name = tool_suggestion["tool"]
            reason = tool_suggestion["reason"]
            next_steps.append(f"� Use `{tool_name}` to {reason.lower()}")
        
        # Add general next steps
        if not any("menu_intelligence" in step for step in next_steps):
            next_steps.append("� Use `menu_intelligence` when you find restaurants - take photos for analysis")
        
        if not wants_to_save_memory and not any("travel_memory_archive" in step for step in next_steps):
            next_steps.append("💾 Use `travel_memory_archive` to save important experiences")
        
        unified_response["next_steps"] = next_steps
        
        return ok(unified_response)
        
    except Exception as e:
        return ok({
            "request_analysis": {
                "original_request": travel_request,
                "error": f"Failed to process request: {str(e)}"
            },
            "suggested_tools": [],
            "immediate_guidance": {},
            "summary": f"❌ Error processing travel request: {str(e)}",
            "next_steps": ["Try simplifying your request or use individual tools directly"]
        })

## Removed: job_finder (not needed)

## Removed: make_img_black_and_white (not needed)

# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
