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
SAFETY_ALERTS: list[dict] = []
CHECK_INS: dict[str, list[dict]] = {}
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

RELIGIOUS_FEST_DESCRIPTION = RichToolDescription(
    description="Religious and Festival Calendar Integration: cross-check travel dates with local events.",
    use_when="User has travel dates and wants to understand festivals and closures.",
)

@mcp.tool(description=RELIGIOUS_FEST_DESCRIPTION.model_dump_json())
async def religious_and_festival_calendar(
    destination: Annotated[str, Field(description="City/Country")],
    start_date: Annotated[str, Field(description="Trip start date in ISO format YYYY-MM-DD")],
    end_date: Annotated[str, Field(description="Trip end date in ISO format YYYY-MM-DD")],
) -> dict:
    # Minimal placeholder: suggest using public calendars or APIs
    events = [
        {"name": "Local Market Day", "date": start_date, "impact": "Road closures possible"},
        {"name": "Religious Observance", "date": end_date, "impact": "Some restaurants may be closed"},
    ]
    return ok({"destination": destination, "events": events})

CROWD_SAFETY_DESCRIPTION = RichToolDescription(
    description="Crowd-Sourced Safety Intelligence: submit and receive real-time safety alerts.",
    use_when="User wants to report or check local safety alerts.",
    side_effects="Stores minimal alert data in memory.",
)

class SafetyAlert(BaseModel):
    id: str
    user: str
    location: str
    type: str
    details: str | None = None
    ts: str

@mcp.tool(description=CROWD_SAFETY_DESCRIPTION.model_dump_json())
async def crowd_sourced_safety_intel(
    action: Annotated[Literal["submit", "list"], Field(description="Submit a new alert or list recent alerts")],
    user_id: Annotated[str, Field(description="User identifier, e.g., phone or session id")],
    location: Annotated[str | None, Field(description="City/Area name or coordinates")] = None,
    alert_type: Annotated[str | None, Field(description="Type: scam/unsafe_area/police_check/etc")] = None,
    details: Annotated[str | None, Field(description="Optional details")]=None,
    limit: Annotated[int | None, Field(description="Max number of alerts to return for list")]=10,
) -> dict:
    if action == "submit":
        if not (location and alert_type):
            raise McpError(ErrorData(code=INVALID_PARAMS, message="location and alert_type required for submit"))
        item = SafetyAlert(id=str(uuid.uuid4()), user=user_id, location=location, type=alert_type, details=details, ts=datetime.utcnow().isoformat()).model_dump()
        SAFETY_ALERTS.append(item)
        return ok({"submitted": item})
    # list
    recent = list(reversed(SAFETY_ALERTS))[: (limit or 10)]
    return ok({"alerts": recent})

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

PREDICTIVE_RISK_DESCRIPTION = RichToolDescription(
    description="Predictive Risk Assessment: analyzes news/weather/social to predict risks 24–48h ahead.",
    use_when="User wants short-term risk forecast.",
)

@mcp.tool(description=PREDICTIVE_RISK_DESCRIPTION.model_dump_json())
async def predictive_risk_assessment(
    destination: Annotated[str, Field(description="City/Country")],
    horizon_hours: Annotated[int, Field(description="Forecast horizon in hours (24-48 recommended)")] = 36,
) -> dict:
    # Placeholder scoring
    score = 0.2 if horizon_hours <= 24 else 0.35
    drivers = ["Weather variability", "Political demonstrations", "Holiday congestion"]
    return ok({"destination": destination, "horizon_hours": horizon_hours, "risk_score": score, "drivers": drivers})

DIGITAL_SAFETY_NET_DESCRIPTION = RichToolDescription(
    description="Digital Safety Net: monitor user check-ins and detect inactivity.",
    use_when="User wants periodic safety pings and alerts on inactivity.",
    side_effects="Stores check-in timestamps in memory.",
)

@mcp.tool(description=DIGITAL_SAFETY_NET_DESCRIPTION.model_dump_json())
async def digital_safety_net(
    action: Annotated[Literal["check_in", "status"], Field(description="Check in now or get status")],
    user_id: Annotated[str, Field(description="User identifier, e.g., phone or session id")],
    inactivity_threshold_minutes: Annotated[int, Field(description="Minutes of inactivity before raising alert")]=120,
) -> dict:
    now = datetime.utcnow()
    if action == "check_in":
        CHECK_INS.setdefault(user_id, []).append({"ts": now.isoformat()})
        return ok({"checked_in_at": now.isoformat()})
    # status
    last = (CHECK_INS.get(user_id) or [])[-1]["ts"] if CHECK_INS.get(user_id) else None
    alert = None
    if last:
        last_dt = datetime.fromisoformat(last)
        mins = (now - last_dt).total_seconds() / 60
        if mins > inactivity_threshold_minutes:
            alert = {"type": "inactivity_alert", "minutes_inactive": round(mins, 1)}
    return ok({"last_check_in": last, "alert": alert})

VISUAL_STORY_DESCRIPTION = RichToolDescription(
    description="Contextual Visual Storytelling: given a landmark image, return engaging stories.",
    use_when="User shares a landmark photo and wants a story.",
)

@mcp.tool(description=VISUAL_STORY_DESCRIPTION.model_dump_json())
async def contextual_visual_storytelling(
    landmark_image_base64: Annotated[str, Field(description="Base64 image of the landmark")],
    interests: Annotated[str | None, Field(description="User interests to tailor the story")]=None,
    language: Annotated[str | None, Field(description="Output language")]="en",
) -> dict:
    # Placeholder: In real system, send to VLM for captioning and story generation
    story = "A storied landmark with centuries of history, echoing tales of trade, culture, and resilience."
    return ok({"language": language, "story": story})

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
    import cv2
    import numpy as np
    import pytesseract
    from PIL import Image
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(menu_image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to OpenCV format for preprocessing
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocess image for better OCR
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to PIL for tesseract
        processed_image = Image.fromarray(thresh)
        
        # Extract text using OCR with custom config for better accuracy
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?$€£¥₹₽₩₪₨₡₵₹()-/:;&@#%+='
        extracted_text = pytesseract.image_to_string(processed_image, config=custom_config).strip()
        
        if not extracted_text:
            return ok({
                "language": language,
                "extracted_text": "No text detected in the image",
                "recommendations": ["Unable to analyze menu - image may be unclear or contain non-text elements"],
                "allergen_warnings": [],
                "etiquette": ["Consider asking staff for assistance with menu interpretation"],
                "translation": "OCR failed - please provide a clearer image"
            })
        
        # Simple menu item detection (look for patterns)
        lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
        menu_items = []
        prices = []
        
        for line in lines:
            # Look for price patterns (numbers with currency symbols)
            if any(char in line for char in ['$', '€', '£', '¥', '₹', '₽', '₩', '₪', '₨', '₡', '₵']):
                prices.append(line)
            elif len(line) > 3 and not line.isdigit():  # Likely a menu item
                menu_items.append(line)
        
        # Generate recommendations based on extracted text and preferences
        recommendations = []
        for item in menu_items[:5]:  # Top 5 items
            item_lower = item.lower()
            if preferences:
                for pref in preferences:
                    if pref.lower() in item_lower:
                        recommendations.append(f"✓ {item} (matches {pref} preference)")
                        break
                else:
                    recommendations.append(f"• {item}")
            else:
                recommendations.append(f"• {item}")
        
        if not recommendations:
            recommendations = ["Popular items based on menu analysis", "Ask server for today's specials"]
        
        # Check for allergen warnings
        allergen_warnings = []
        if allergies:
            text_lower = extracted_text.lower()
            for allergen in allergies:
                allergen_keywords = {
                    'nuts': ['nut', 'almond', 'walnut', 'peanut', 'cashew', 'pistachio'],
                    'dairy': ['milk', 'cheese', 'cream', 'butter', 'yogurt', 'lactose'],
                    'gluten': ['wheat', 'bread', 'pasta', 'flour', 'gluten'],
                    'soy': ['soy', 'soybean', 'tofu'],
                    'eggs': ['egg', 'mayo', 'mayonnaise'],
                    'fish': ['fish', 'salmon', 'tuna', 'cod', 'seafood'],
                    'shellfish': ['shrimp', 'crab', 'lobster', 'shellfish', 'prawns']
                }
                
                keywords = allergen_keywords.get(allergen.lower(), [allergen.lower()])
                if any(keyword in text_lower for keyword in keywords):
                    allergen_warnings.append(f"⚠️ {allergen.title()} detected in menu items")
        
        # Basic etiquette suggestions
        etiquette = [
            "Point to menu items if language barrier exists",
            "Ask 'What do you recommend?' in local language",
            "Check if service charge is included before tipping"
        ]
        
        # Simple translation attempt (placeholder - in production you'd use a translation API)
        translation_note = f"Extracted {len(lines)} lines of text from menu image"
        if language != "en":
            translation_note += f" (translation to {language} requires external API)"
        
        return ok({
            "language": language,
            "extracted_text": extracted_text,
            "menu_items": menu_items,
            "detected_prices": prices,
            "recommendations": recommendations,
            "allergen_warnings": allergen_warnings,
            "etiquette": etiquette,
            "translation": translation_note,
            "ocr_confidence": "OCR processing completed successfully"
        })
        
    except Exception as e:
        return ok({
            "language": language,
            "error": f"OCR processing failed: {str(e)}",
            "recommendations": ["Please provide a clearer image of the menu"],
            "allergen_warnings": ["Unable to detect allergens - please ask staff"],
            "etiquette": ["Consider using translation app as backup"],
            "translation": "OCR failed - manual translation needed"
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

ACCENT_TRAIN_DESCRIPTION = RichToolDescription(
    description="Accent and Dialect Adaptation Training: pronounce phrases with local accent and feedback.",
    use_when="User practicing phrases.",
)

@mcp.tool(description=ACCENT_TRAIN_DESCRIPTION.model_dump_json())
async def accent_and_dialect_training(
    target_phrase: Annotated[str, Field(description="Phrase to practice")],
    language: Annotated[str, Field(description="Target language/dialect")],
    user_audio_base64: Annotated[str | None, Field(description="Optional recorded audio for feedback")]=None,
) -> dict:
    # Placeholder scoring
    feedback = "Good pace, soften the 'r' sound, emphasize vowel length"
    score = 0.78
    return ok({"language": language, "feedback": feedback, "score": score})

LIVE_VOICE_INTERPRETER_DESCRIPTION = RichToolDescription(
    description="Two-Way Live Voice Interpreter: real-time voice translations.",
    use_when="Back-and-forth live interpreting.",
)

@mcp.tool(description=LIVE_VOICE_INTERPRETER_DESCRIPTION.model_dump_json())
async def two_way_live_voice_interpreter(
    source_audio_base64: Annotated[str, Field(description="Base64 audio in source language")],
    source_lang: Annotated[str, Field(description="Source language code")],
    target_lang: Annotated[str, Field(description="Target language code")],
) -> dict:
    # Placeholder: STT -> translate -> TTS
    return ok({"transcript": "Hello, where is the station?", "translation": "Hola, ¿dónde está la estación?", "audio_base64": None})

MESSAGE_RELAY_AUDIO_DESCRIPTION = RichToolDescription(
    description="Message Relay with Local Audio Translation: typed text -> local-language audio and back.",
    use_when="User wants to play audio in local language for another person.",
)

@mcp.tool(description=MESSAGE_RELAY_AUDIO_DESCRIPTION.model_dump_json())
async def message_relay_audio_translation(
    text: Annotated[str, Field(description="Text to translate and synthesize")],
    target_lang: Annotated[str, Field(description="Target language code")],
) -> dict:
    # Placeholder TTS audio
    return ok({"translation": text, "audio_base64": None})

EXPENSE_CONTEXT_DESCRIPTION = RichToolDescription(
    description="Real-time Expense Cultural Context: detect overpricing and negotiation tips.",
    use_when="User wants to know if a price is fair.",
)

@mcp.tool(description=EXPENSE_CONTEXT_DESCRIPTION.model_dump_json())
async def expense_cultural_context(
    item: Annotated[str, Field(description="Item/service name")],
    quoted_price: Annotated[float, Field(description="Quoted price in local currency")],
    typical_price_range: Annotated[list[float] | None, Field(description="Typical min/max price in local currency")]=None,
    negotiation_style: Annotated[str | None, Field(description="polite/friendly/firm")]="friendly",
) -> dict:
    min_p, max_p = (typical_price_range or [quoted_price * 0.6, quoted_price * 0.9])
    overpriced = quoted_price > max_p
    tips = ["Start with a smile and counter at 60-70%", "Be ready to walk away politely"]
    return ok({
        "item": item,
        "quoted_price": quoted_price,
        "typical_price_range": [min_p, max_p],
        "overpriced": overpriced,
        "negotiation_tips": tips,
    })

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

# Retain existing job and image tools for compatibility

JOBFINDER_DESC = RichToolDescription(
    description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
    use_when="Evaluate job descriptions or search for jobs using freeform goals.",
    side_effects="Returns insights, fetched job descriptions, or relevant links.",
)

@mcp.tool(description=JOBFINDER_DESC.model_dump_json())
async def job_finder(
    user_goal: Annotated[str, Field(description="The user's goal (description, intent, or freeform query)")],
    job_description: Annotated[str | None, Field(description="Full job description text")] = None,
    job_url: Annotated[AnyUrl | None, Field(description="URL to fetch a job description")] = None,
    raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
) -> str:
    if job_description:
        return (
            f"📝 **Job Description Analysis**\n\n"
            f"---\n{job_description.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**\n\n"
            f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
        )

    if job_url:
        content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
        return (
            f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n"
            f"---\n{content.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**"
        )

    if "look for" in user_goal.lower() or "find" in user_goal.lower():
        links = await Fetch.google_search_links(user_goal)
        return (
            f"🔍 **Search Results for**: _{user_goal}_\n\n" +
            "\n".join(f"- {link}" for link in links)
        )

    raise McpError(ErrorData(code=INVALID_PARAMS, message="Provide either a job description, a job URL, or a search query in user_goal."))

MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert an image to black and white and save it.",
    use_when="Use when the user provides an image to convert to black and white.",
    side_effects="Processes image in-memory and returns PNG base64.",
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
) -> list[TextContent | ImageContent]:
    import base64
    import io

    from PIL import Image

    try:
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))

        bw_image = image.convert("L")

        buf = io.BytesIO()
        bw_image.save(buf, format="PNG")
        bw_bytes = buf.getvalue()
        bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

        return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
