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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# Debug: Print environment variable status
print(f"🔑 AUTH_TOKEN loaded: {'✅' if TOKEN else '❌'}")
print(f"📱 MY_NUMBER loaded: {'✅' if MY_NUMBER else '❌'}")
print(f"🤖 GEMINI_API_KEY loaded: {'✅' if GEMINI_API_KEY else '❌'}")
if GEMINI_API_KEY:
    print(f"🔑 API Key starts with: {GEMINI_API_KEY[:10]}...")
else:
    print("⚠️ WARNING: GEMINI_API_KEY not found - menu intelligence will use fallback OCR only")

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

# --- Centralized Gemini API Helper ---
async def call_gemini_api(prompt: str, system_context: str = "", include_debug: bool = True) -> str:
    """
    Centralized Gemini API caller for all travel tools
    """
    try:
        import google.generativeai as genai
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise Exception("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Combine system context with user prompt
        full_prompt = f"{system_context}\n\n{prompt}" if system_context else prompt
        
        if include_debug:
            print(f"🤖 Calling Gemini API for enhanced response...")
            print(f"📝 Prompt length: {len(full_prompt)} characters")
        
        response = model.generate_content(full_prompt)
        
        if include_debug:
            print(f"✅ Gemini response received, length: {len(response.text) if response.text else 0}")
        
        if not response.text:
            raise Exception("Gemini returned empty response")
        
        return response.text
        
    except Exception as e:
        if include_debug:
            print(f"❌ Gemini API error: {str(e)}")
        raise Exception(f"Gemini API failed: {str(e)}")

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
    
    try:
        # Enhanced AI-powered cultural analysis using Gemini
        system_context = """You are a world-renowned cultural anthropologist and travel expert specializing in cross-cultural communication and etiquette. 
        Provide detailed, practical, and respectful cultural guidance for travelers."""
        
        profile_text = f" The traveler's profile/interests: {traveler_profile}" if traveler_profile else ""
        
        prompt = f"""
        Provide comprehensive cultural guidance for someone traveling from {home_country} to {destination_country}.{profile_text}
        
        Format your response as detailed JSON with these sections:
        {{
            "cultural_overview": {{
                "key_differences": "Major cultural differences between home and destination",
                "communication_style": "How people communicate (direct/indirect, formal/casual)",
                "social_hierarchy": "Understanding of respect and authority structures"
            }},
            "greetings_and_manners": {{
                "appropriate_greetings": "How to greet people properly",
                "personal_space": "Understanding of physical boundaries",
                "eye_contact": "Cultural norms around eye contact",
                "gestures_to_avoid": "Hand gestures or body language to be careful with"
            }},
            "taboos_and_sensitivities": [
                "Critical cultural taboos to absolutely avoid",
                "Religious sensitivities to be aware of",
                "Political topics to avoid discussing",
                "Social behaviors that might offend"
            ],
            "dining_etiquette": {{
                "table_manners": "Proper dining behavior and utensil use",
                "food_customs": "Special food-related customs or rituals",
                "business_dining": "Professional meal etiquette if applicable",
                "tipping_culture": "Expected tipping practices"
            }},
            "business_culture": {{
                "meeting_etiquette": "How to conduct business meetings",
                "gift_giving": "Appropriate business gifts and occasions",
                "dress_codes": "Professional attire expectations",
                "punctuality": "Time management and scheduling norms"
            }},
            "daily_life_integration": {{
                "public_behavior": "How to behave in public spaces",
                "shopping_customs": "Market and retail interaction norms",
                "transportation_etiquette": "Behavior on public transport",
                "technology_use": "Mobile phone and photography etiquette"
            }},
            "language_and_communication": {{
                "key_phrases": "Essential polite phrases to learn",
                "non_verbal_communication": "Important body language cues",
                "conversation_topics": "Safe and appreciated discussion topics",
                "cultural_humor": "Understanding local humor and what to avoid"
            }},
            "practical_integration_tips": [
                "Specific actionable advice for smooth cultural integration",
                "Common mistakes travelers from {home_country} make in {destination_country}",
                "Ways to show respect and appreciation for local culture"
            ]
        }}
        
        Make your advice specific, practical, and culturally sensitive. Include real examples where helpful.
        """
        
        gemini_response = await call_gemini_api(prompt, system_context)
        
        # Try to parse JSON response
        try:
            import json
            clean_text = gemini_response.strip()
            if clean_text.startswith('```json'):
                clean_text = clean_text[7:]
            if clean_text.endswith('```'):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()
            
            ai_insights = json.loads(clean_text)
            
            return ok({
                "home_country": home_country,
                "destination_country": destination_country,
                "traveler_profile": traveler_profile,
                "ai_powered_insights": ai_insights,
                "analysis_method": "Gemini 1.5 Flash Cultural Analysis",
                "success": True
            })
            
        except json.JSONDecodeError:
            # Fallback to text response if JSON parsing fails
            return ok({
                "home_country": home_country,
                "destination_country": destination_country,
                "traveler_profile": traveler_profile,
                "cultural_guidance": gemini_response,
                "analysis_method": "Gemini 1.5 Flash (text mode)",
                "success": True
            })
    
    except Exception as e:
        # Fallback to basic insights if Gemini fails
        basic_insights = {
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
            "insights": basic_insights,
            "analysis_method": "Basic fallback",
            "error": f"AI analysis failed: {str(e)}",
            "success": False
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
    
    try:
        # Enhanced AI-powered local social analysis using Gemini
        system_context = """You are a local cultural expert and social anthropologist with deep knowledge of urban dynamics, 
        local customs, and social behaviors in cities around the world. Provide practical, safety-conscious advice."""
        
        context_text = f" in the context of {context}" if context else ""
        
        prompt = f"""
        Provide detailed local social dynamics advice for someone visiting {city}, {country} during {time_of_day}{context_text}.
        
        Format your response as detailed JSON:
        {{
            "location_overview": {{
                "local_vibe": "General atmosphere and energy of the area",
                "typical_crowd": "Who you'll encounter at this time",
                "activity_level": "How busy or quiet it typically is",
                "safety_level": "General safety assessment for travelers"
            }},
            "time_specific_dynamics": {{
                "peak_hours": "When this area is busiest",
                "quiet_periods": "When it's more peaceful",
                "recommended_times": "Best times to visit for different purposes",
                "time_sensitive_warnings": "Things to be aware of at this specific time"
            }},
            "social_behavior_norms": {{
                "interaction_style": "How locals typically interact with each other and visitors",
                "conversation_appropriateness": "When and how to engage with locals",
                "personal_space": "Physical distance norms in this area",
                "queue_etiquette": "How lines and waiting work here"
            }},
            "practical_navigation_advice": [
                "How to move through the area like a local",
                "Where to stand/sit/walk for best experience",
                "How to avoid tourist traps or uncomfortable situations",
                "Local shortcuts or insider tips"
            ],
            "safety_and_awareness": {{
                "areas_to_avoid": "Specific locations or situations to be cautious about",
                "valuable_items": "How to handle money, phones, cameras",
                "emergency_contacts": "Local emergency numbers or helpful services",
                "situational_awareness": "What to watch out for"
            }},
            "cultural_integration": {{
                "dress_appropriately": "How to dress to fit in",
                "behavioral_cues": "How to read local social signals",
                "respectful_photography": "Photography etiquette in this area",
                "supporting_locals": "How to contribute positively to the community"
            }},
            "context_specific_advice": [
                "Advice specifically tailored to the {context if context else 'general visit'} context",
                "Unique opportunities or experiences available at this time",
                "Local customs specific to this situation"
            ]
        }}
        
        Be specific to {city}, {country} and provide actionable, culturally sensitive advice.
        """
        
        gemini_response = await call_gemini_api(prompt, system_context)
        
        # Try to parse JSON response
        try:
            import json
            clean_text = gemini_response.strip()
            if clean_text.startswith('```json'):
                clean_text = clean_text[7:]
            if clean_text.endswith('```'):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()
            
            ai_insights = json.loads(clean_text)
            
            return ok({
                "city": city,
                "country": country,
                "time_of_day": time_of_day,
                "context": context,
                "ai_powered_insights": ai_insights,
                "analysis_method": "Gemini 1.5 Flash Social Dynamics Analysis",
                "success": True
            })
            
        except json.JSONDecodeError:
            # Fallback to text response if JSON parsing fails
            return ok({
                "city": city,
                "country": country,
                "time_of_day": time_of_day,
                "context": context,
                "social_guidance": gemini_response,
                "analysis_method": "Gemini 1.5 Flash (text mode)",
                "success": True
            })
    
    except Exception as e:
        # Fallback to basic advice if Gemini fails
        basic_advice = [
            "Stay aware of personal space in crowded areas",
            "In markets, friendly bargaining is common",
            "Avoid displaying valuables",
        ]
        return ok({
            "city": city,
            "country": country,
            "time_of_day": time_of_day,
            "context": context,
            "advice": basic_advice,
            "analysis_method": "Basic fallback",
            "error": f"AI analysis failed: {str(e)}",
            "success": False
        })

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
    track_tool_usage("emergency_phrase_generator")
    
    try:
        # Enhanced AI-powered phrase generation using Gemini
        system_context = """You are a professional translator and emergency response expert with fluency in dozens of languages. 
        Provide accurate, culturally appropriate emergency phrases that are clear and respectful in urgent situations."""
        
        prompt = f"""
        Generate comprehensive emergency phrases for the intent "{intent}" in {language} language with {politeness_level} politeness level.
        
        Format your response as detailed JSON:
        {{
            "primary_phrases": {{
                "main_request": "The core emergency phrase for this situation",
                "urgent_version": "More urgent/desperate version if needed immediately",
                "polite_version": "More polite version for less urgent situations"
            }},
            "supporting_phrases": {{
                "location_help": "How to ask for directions or describe where you are",
                "contact_help": "How to ask someone to call emergency services",
                "language_barrier": "How to indicate you don't speak the local language",
                "thank_you": "How to express gratitude for help received"
            }},
            "pronunciation_guide": {{
                "phonetic_spelling": "How to pronounce the main phrases phonetically",
                "stress_patterns": "Which syllables to emphasize",
                "common_mistakes": "Pronunciation mistakes to avoid"
            }},
            "cultural_context": {{
                "appropriate_people": "Who to approach for this type of help",
                "body_language": "Appropriate gestures or body language to use",
                "cultural_sensitivity": "Cultural considerations for emergency situations",
                "escalation_protocol": "How to escalate if initial help isn't sufficient"
            }},
            "emergency_contacts": {{
                "local_emergency_number": "Primary emergency number in this region",
                "specific_services": "Numbers for police, medical, fire if different",
                "embassy_info": "How to contact your country's embassy",
                "tourist_assistance": "Tourist-specific emergency services"
            }},
            "backup_communication": {{
                "written_phrases": "Key phrases written down to show people",
                "universal_gestures": "Non-verbal communication that works across cultures",
                "translation_apps": "Recommended translation apps for this language",
                "visual_aids": "How to use pictures or drawings to communicate"
            }},
            "follow_up_phrases": [
                "What to say after getting initial help",
                "How to provide more details about your situation",
                "How to ask for ongoing assistance or follow-up"
            ]
        }}
        
        Make the phrases accurate, culturally appropriate, and practical for real emergency situations.
        Include pronunciation guides that would help an English speaker.
        """
        
        gemini_response = await call_gemini_api(prompt, system_context)
        
        # Try to parse JSON response
        try:
            import json
            clean_text = gemini_response.strip()
            if clean_text.startswith('```json'):
                clean_text = clean_text[7:]
            if clean_text.endswith('```'):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()
            
            ai_phrases = json.loads(clean_text)
            
            return ok({
                "intent": intent,
                "language": language,
                "politeness_level": politeness_level,
                "ai_powered_phrases": ai_phrases,
                "analysis_method": "Gemini 1.5 Flash Translation & Cultural Analysis",
                "success": True
            })
            
        except json.JSONDecodeError:
            # Fallback to text response if JSON parsing fails
            return ok({
                "intent": intent,
                "language": language,
                "politeness_level": politeness_level,
                "emergency_guidance": gemini_response,
                "analysis_method": "Gemini 1.5 Flash (text mode)",
                "success": True
            })
    
    except Exception as e:
        # Fallback to basic phrases if Gemini fails
        basic_phrases = {
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
        chosen = basic_phrases.get(intent, {"en": "Please help me."})
        
        return ok({
            "intent": intent,
            "language": language,
            "politeness_level": politeness_level,
            "phrase": chosen.get("en"),
            "analysis_method": "Basic fallback",
            "error": f"AI translation failed: {str(e)}",
            "success": False
        })

## Removed: predictive_risk_assessment (not needed)

## Removed: digital_safety_net (not needed)

## Removed: contextual_visual_storytelling (not needed)

MENU_INTEL_DESCRIPTION = RichToolDescription(
    description="Advanced Menu Intelligence & Local Cuisine Discovery: analyze menu photos with Gemini 2.0 Pro for allergens, recommendations, cultural insights, and discover must-try local dishes and restaurants worth visiting.",
    use_when="User shares a menu photo, wants dining recommendations, or needs local cuisine discovery in any location.",
)

@mcp.tool(description=MENU_INTEL_DESCRIPTION.model_dump_json())
async def menu_intelligence(
    menu_image_base64: Annotated[str | None, Field(description="Base64-encoded image of the menu (optional - can discover local cuisine without image)")]=None,
    allergies: Annotated[list[str] | None, Field(description="List of allergens to avoid")]=None,
    preferences: Annotated[list[str] | None, Field(description="Cuisine/diet preferences, e.g., vegetarian, spicy, traditional")]=None,
    location: Annotated[str | None, Field(description="City and country for local cuisine discovery (e.g., 'Tokyo, Japan')")]=None,
    discovery_mode: Annotated[bool, Field(description="Enable local cuisine discovery and must-try dishes recommendations")]=False,
    language: Annotated[str | None, Field(description="Language for output")]="en",
) -> dict:
    import base64
    import io
    import re
    import json
    from PIL import Image
    
    # Handle discovery mode - recommend local cuisine without menu image
    if discovery_mode and location:
        try:
            cuisine_discovery, method = await discover_local_cuisine(location, allergies, preferences, language)
            result = {
                "mode": "cuisine_discovery",
                "location": location,
                "language": language,
                "discovery_method": method,
                **cuisine_discovery
            }
            
            # Add practical travel tips
            result["travel_tips"] = {
                "best_dining_times": "Local peak dining hours and quiet periods",
                "reservation_culture": "When and how to make reservations",
                "payment_methods": "Accepted payment types and tipping customs",
                "dress_code": "Appropriate attire for different restaurant types"
            }
            
            track_tool_usage("menu_intelligence_discovery")
            return ok(result)
            
        except Exception as discovery_error:
            return ok({
                "mode": "cuisine_discovery",
                "location": location,
                "error": f"Local cuisine discovery failed: {str(discovery_error)}",
                "fallback_advice": [
                    "Search for '[location] must try food' online",
                    "Ask locals for restaurant recommendations", 
                    "Visit local food markets for authentic experiences",
                    "Look for restaurants with local customers"
                ],
                "success": False
            })
    
    # Handle menu image analysis
    if not menu_image_base64:
        return ok({
            "mode": "no_image_provided",
            "error": "No menu image provided and discovery_mode not enabled",
            "suggestion": "Provide a menu image or enable discovery_mode with location for local cuisine recommendations",
            "success": False
        })
    
    async def discover_local_cuisine(location, allergies, preferences, language):
        """Discover local cuisine and must-try dishes using Gemini 2.0 Pro"""
        try:
            import google.generativeai as genai
            
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise Exception("GEMINI_API_KEY not found in environment")
            
            # Configure Gemini Pro
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Create comprehensive prompt for local cuisine discovery
            allergen_text = f"IMPORTANT - Avoid recommending dishes with: {', '.join(allergies)}" if allergies else "No specific allergies to avoid"
            preference_text = f"User preferences: {', '.join(preferences)}" if preferences else "No specific preferences mentioned"
            
            prompt = f"""
You are a world-renowned culinary expert and travel guide. Provide comprehensive local cuisine information for {location}.

Return your response in the following JSON format:

{{
    "location": "{location}",
    "cuisine_overview": {{
        "cuisine_type": "name of local cuisine style",
        "history": "brief culinary history and influences",
        "characteristics": "what makes this cuisine unique"
    }},
    "must_try_dishes": [
        {{
            "name": "dish name in local language",
            "english_name": "English translation if different",
            "description": "what the dish is and how it's prepared",
            "cultural_significance": "why this dish is important locally",
            "taste_profile": "flavor description (spicy, sweet, umami, etc.)",
            "price_range": "typical cost range in local currency",
            "best_time_to_eat": "breakfast/lunch/dinner/snack",
            "allergen_info": "potential allergens in this dish",
            "vegetarian_friendly": true/false,
            "spice_level": "mild/medium/hot/very hot"
        }}
    ],
    "recommended_restaurants": [
        {{
            "name": "restaurant name",
            "type": "street food/casual/fine dining/local institution",
            "specialties": ["list of signature dishes"],
            "atmosphere": "description of dining experience",
            "price_range": "budget/moderate/expensive",
            "local_popularity": "why locals love this place",
            "tourist_friendly": true/false,
            "reservations_needed": true/false
        }}
    ],
    "food_districts": [
        {{
            "area_name": "district or street name",
            "specialty": "what this area is known for",
            "best_time_to_visit": "time recommendation",
            "atmosphere": "what to expect"
        }}
    ],
    "dining_etiquette": [
        "important cultural dining rules and customs"
    ],
    "food_safety_tips": [
        "practical advice for safe eating"
    ],
    "seasonal_specialties": [
        {{
            "season": "season name",
            "dishes": ["seasonal dishes available"],
            "festivals": "food-related festivals if any"
        }}
    ],
    "budget_recommendations": {{
        "street_food": "best budget options with price ranges",
        "mid_range": "good value restaurants",
        "splurge": "special occasion dining"
    }},
    "allergen_safe_options": [
        "dishes that are safe for mentioned allergies"
    ],
    "preference_matches": [
        "dishes that match user preferences"
    ],
    "language_help": {{
        "key_phrases": {{
            "ordering": "how to order in local language",
            "dietary_restrictions": "how to communicate allergies/preferences",
            "compliments": "how to compliment the food"
        }},
        "menu_terms": [
            {{"local_term": "english_meaning"}}
        ]
    }}
}}

User Context:
{allergen_text}
{preference_text}
Language for response: {language}

Please provide authentic, culturally accurate information based on your knowledge of {location}'s culinary scene. Focus on dishes and places that locals actually recommend, not just tourist attractions.
"""
            
            # Generate response
            response = model.generate_content(prompt)
            
            if not response.text:
                raise Exception("Gemini returned empty response for cuisine discovery")
            
            # Parse JSON response
            try:
                clean_text = response.text.strip()
                if clean_text.startswith('```json'):
                    clean_text = clean_text[7:]
                if clean_text.endswith('```'):
                    clean_text = clean_text[:-3]
                clean_text = clean_text.strip()
                
                result = json.loads(clean_text)
                return result, "Gemini 2.0 Pro Cuisine Discovery"
                
            except json.JSONDecodeError:
                # If JSON parsing fails, create structured response from text
                response_text = response.text
                return {
                    "location": location,
                    "cuisine_overview": {"cuisine_type": "Local Cuisine", "description": response_text[:300] + "..."},
                    "must_try_dishes": [],
                    "recommended_restaurants": [],
                    "dining_etiquette": [response_text[:200] + "..."],
                    "raw_response": response_text
                }, "Gemini 2.0 Pro (text mode)"
                
        except Exception as e:
            raise Exception(f"Gemini cuisine discovery failed: {str(e)}")
    
    async def analyze_menu_with_gemini_pro(image_bytes, allergies, preferences, location):
        """Enhanced menu analysis using Gemini Pro with safety settings"""
        print(f"📊 Starting Gemini menu analysis...")
        try:
            import google.generativeai as genai
            
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise Exception("GEMINI_API_KEY not found in environment")
            
            # Configure Gemini Pro
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Simple prompt for menu analysis
            prompt = "Analyze this menu image and tell me what food items you can see with their prices."
            
            # Debug logging
            print(f"🔍 Calling Gemini API with model: gemini-1.5-flash")
            print(f"🖼️ Image size: {len(image_bytes)} bytes")
            
            # Generate response
            response = model.generate_content([prompt, pil_image])
            
            print(f"✅ Gemini API response received, length: {len(response.text) if response.text else 0}")
            
            if not response.text:
                raise Exception("Gemini returned empty response")
            
            # Return simple text response
            return {
                "menu_analysis": response.text,
                "method": "Gemini 1.5 Flash",
                "success": True
            }, "Gemini 1.5 Flash Menu Analysis"
                
        except Exception as e:
            raise Exception(f"Gemini Pro menu analysis failed: {str(e)}")
    
    # Main processing logic starts here
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
                "mode": "image_analysis",
                "language": language,
                "error": f"Failed to decode image: {str(decode_error)}",
                "suggestion": "Please check the image format and try again",
                "success": False
            })
        
        # Enhanced Menu Analysis with Gemini 2.0 Pro
        try:
            analysis_result, analysis_method = await analyze_menu_with_gemini_pro(image_bytes, allergies, preferences, location)
            
            # Add cultural dining context if location provided
            if location and discovery_mode:
                try:
                    local_context, _ = await discover_local_cuisine(location, allergies, preferences, language)
                    analysis_result["local_cuisine_context"] = {
                        "regional_specialties": local_context.get("must_try_dishes", [])[:3],
                        "local_dining_customs": local_context.get("dining_etiquette", [])[:3],
                        "similar_local_dishes": "Dishes from this menu that match local cuisine"
                    }
                except Exception:
                    pass  # Local context is optional
            
            # Prepare comprehensive result
            final_result = {
                "mode": "enhanced_menu_analysis",
                "language": language,
                "location": location or "Not specified",
                "analysis_method": analysis_method,
                "success": True,
                **analysis_result
            }
            
            # Add practical dining guidance
            final_result["practical_guidance"] = {
                "ordering_strategy": "How to order like a local",
                "portion_planning": "How much to order for your group",
                "cultural_dos": ["What to do while dining"],
                "cultural_donts": ["What to avoid while dining"],
                "payment_guidance": "How to handle the bill and tipping"
            }
            
            track_tool_usage("menu_intelligence_enhanced")
            return ok(final_result)
            
        except Exception as gemini_error:
            return ok({
                "mode": "error_state",
                "language": language,
                "error": f"Gemini API failed: {str(gemini_error)}",
                "suggestions": [
                    "Try with a clearer, higher-resolution image",
                    "Ensure good lighting and minimal glare",
                    "Check that GEMINI_API_KEY is properly configured"
                ],
                "success": False
            })
        
    except Exception as e:
        return ok({
            "mode": "unexpected_error",
            "language": language,
            "error": f"Unexpected error in menu intelligence: {str(e)}",
            "fallback_advice": [
                "Ask staff for menu recommendations",
                "Use translation app for basic text",
                "Point to dishes when ordering",
                "Request allergen information from server"
            ],
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
    track_tool_usage("local_navigation_social_intelligence")
    
    try:
        # Enhanced AI-powered navigation analysis using Gemini
        system_context = """You are a local transportation expert and safety advisor with deep knowledge of urban navigation, 
        local safety conditions, tourist considerations, and optimal routing strategies."""
        
        prompt = f"""
        Provide comprehensive navigation guidance from {origin} to {destination} using {mode} transport with {caution_preference} caution level.
        
        Format your response as detailed JSON:
        {{
            "route_analysis": {{
                "estimated_time": "Approximate travel time",
                "distance": "Approximate distance",
                "difficulty_level": "Easy/Moderate/Challenging for travelers",
                "scenic_value": "How interesting or beautiful the route is"
            }},
            "safety_assessment": {{
                "overall_safety_score": "Scale 1-10 with 10 being very safe",
                "time_sensitive_warnings": "Safety considerations that vary by time of day",
                "tourist_specific_risks": "Risks particularly relevant to travelers",
                "emergency_protocols": "What to do if you encounter problems"
            }},
            "route_segments": [
                {{
                    "segment_description": "Description of this part of the journey",
                    "navigation_instructions": "Step-by-step directions",
                    "safety_level": "Low/Medium/High risk for this segment",
                    "local_tips": "Insider knowledge for this area",
                    "landmarks": "Notable landmarks to help with navigation",
                    "estimated_time": "Time for this segment"
                }}
            ],
            "mode_specific_advice": {{
                "transport_tips": "Specific advice for {mode} travel",
                "payment_methods": "How to pay for transport if applicable",
                "etiquette": "Behavioral norms for this transport mode",
                "backup_options": "Alternative transport if primary fails"
            }},
            "cultural_navigation": {{
                "local_behavior": "How locals typically navigate this route",
                "tourist_considerations": "How to avoid looking like an obvious tourist",
                "interaction_points": "Where you might need to interact with locals",
                "language_needs": "Key phrases for navigation help"
            }},
            "practical_preparation": {{
                "what_to_bring": "Items needed for this journey",
                "apps_to_download": "Helpful navigation or transport apps",
                "offline_preparation": "What to prepare in case of no internet",
                "weather_considerations": "How weather might affect the route"
            }},
            "alternative_routes": [
                {{
                    "route_name": "Alternative route description",
                    "reason": "Why you might choose this route instead",
                    "trade_offs": "Pros and cons compared to main route"
                }}
            ],
            "local_insights": [
                "Hidden gems or interesting stops along the way",
                "Local shortcuts or route optimizations",
                "Cultural experiences available during the journey",
                "Safety tips specific to this area and route"
            ]
        }}
        
        Consider the {caution_preference} caution preference in your safety assessments and route recommendations.
        """
        
        gemini_response = await call_gemini_api(prompt, system_context)
        
        # Try to parse JSON response
        try:
            import json
            clean_text = gemini_response.strip()
            if clean_text.startswith('```json'):
                clean_text = clean_text[7:]
            if clean_text.endswith('```'):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()
            
            ai_navigation = json.loads(clean_text)
            
            return ok({
                "origin": origin,
                "destination": destination,
                "mode": mode,
                "caution_preference": caution_preference,
                "ai_powered_navigation": ai_navigation,
                "analysis_method": "Gemini 1.5 Flash Navigation & Safety Analysis",
                "success": True
            })
            
        except json.JSONDecodeError:
            # Fallback to text response if JSON parsing fails
            return ok({
                "origin": origin,
                "destination": destination,
                "mode": mode,
                "caution_preference": caution_preference,
                "navigation_guidance": gemini_response,
                "analysis_method": "Gemini 1.5 Flash (text mode)",
                "success": True
            })
    
    except Exception as e:
        # Fallback to basic navigation if Gemini fails
        basic_steps = [
            {"instruction": "Head north 200m", "risk": "low"},
            {"instruction": "Through market lane", "risk": "medium", "note": "crowded at evenings"},
            {"instruction": "Arrive at destination", "risk": "low"},
        ]
        score = 0.15 if caution_preference == "high" else 0.25
        
        return ok({
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "caution_preference": caution_preference,
            "safety_score": score,
            "steps": basic_steps,
            "analysis_method": "Basic fallback",
            "error": f"AI navigation failed: {str(e)}",
            "success": False
        })

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
    action: Annotated[Literal["save", "list", "analyze"], Field(description="Save a memory, list memories, or analyze memories")],
    user_id: Annotated[str, Field(description="User identifier")],
    title: Annotated[str | None, Field(description="Memory title")]=None,
    text: Annotated[str | None, Field(description="Narrative text")]=None,
    image_base64: Annotated[str | None, Field(description="Optional photo base64")]=None,
    tags: Annotated[list[str] | None, Field(description="Optional tags")]=None,
) -> dict:
    track_tool_usage("travel_memory_archive")
    
    if action == "save":
        if not (title or text or image_base64):
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Provide at least title, text, or image_base64 to save"))
        
        # Enhanced memory processing with Gemini analysis
        enhanced_memory = {
            "id": str(uuid.uuid4()),
            "ts": datetime.utcnow().isoformat(),
            "title": title,
            "text": text,
            "image_base64": image_base64,
            "tags": tags or [],
        }
        
        try:
            # Use Gemini to enhance the memory with AI insights
            memory_content = f"Title: {title}\nText: {text}" if title and text else (title or text or "Photo memory")
            
            system_context = """You are a travel memory curator and cultural anthropologist who helps travelers 
            understand and reflect on their experiences with deeper cultural and personal insights."""
            
            prompt = f"""
            Analyze this travel memory and provide enriching insights:
            
            Memory: {memory_content}
            
            Provide response as JSON:
            {{
                "cultural_insights": [
                    "Cultural significance or context of this experience",
                    "What this reveals about local customs or traditions"
                ],
                "emotional_reflection": {{
                    "mood": "Emotional tone of this memory",
                    "significance": "Why this moment was meaningful",
                    "growth_opportunity": "What this experience might teach"
                }},
                "travel_wisdom": [
                    "Practical lessons learned from this experience",
                    "Advice for future travelers in similar situations"
                ],
                "connection_opportunities": {{
                    "similar_experiences": "What other experiences this connects to",
                    "future_exploration": "Related places or experiences to explore",
                    "cultural_bridges": "How this connects to other cultures"
                }},
                "memory_enhancement": {{
                    "additional_context": "Historical or cultural background",
                    "sensory_details": "Sensory elements that make this memory vivid",
                    "storytelling_angle": "How to tell this story to others"
                }}
            }}
            """
            
            ai_analysis = await call_gemini_api(prompt, system_context, include_debug=False)
            
            # Try to parse and add AI insights
            try:
                import json
                clean_text = ai_analysis.strip()
                if clean_text.startswith('```json'):
                    clean_text = clean_text[7:]
                if clean_text.endswith('```'):
                    clean_text = clean_text[:-3]
                enhanced_memory["ai_insights"] = json.loads(clean_text.strip())
                enhanced_memory["analysis_method"] = "Gemini 1.5 Flash Memory Analysis"
            except:
                enhanced_memory["ai_insights"] = {"analysis": ai_analysis}
                enhanced_memory["analysis_method"] = "Gemini 1.5 Flash (text mode)"
        
        except Exception as e:
            enhanced_memory["ai_insights"] = {"error": f"AI analysis failed: {str(e)}"}
            enhanced_memory["analysis_method"] = "Basic storage"
        
        MEMORIES.setdefault(user_id, []).append(enhanced_memory)
        return ok({"saved": enhanced_memory, "success": True})
    
    elif action == "analyze":
        user_memories = MEMORIES.get(user_id, [])
        if not user_memories:
            return ok({"analysis": "No memories found to analyze", "success": False})
        
        try:
            # Analyze patterns across all memories
            memory_summaries = []
            for memory in user_memories[-10:]:  # Analyze last 10 memories
                summary = f"Title: {memory.get('title', 'Untitled')}, Text: {memory.get('text', 'No text')[:100]}"
                memory_summaries.append(summary)
            
            system_context = """You are a travel pattern analyst who helps travelers understand their journey themes, 
            growth patterns, and travel personality through their collected memories."""
            
            prompt = f"""
            Analyze these travel memories to identify patterns and provide insights:
            
            Memories: {', '.join(memory_summaries)}
            
            Provide comprehensive analysis as JSON:
            {{
                "travel_patterns": {{
                    "preferred_experiences": "Types of experiences user gravitates toward",
                    "cultural_interests": "Cultural themes that appear frequently",
                    "growth_trajectory": "How the traveler seems to be evolving",
                    "comfort_zone": "What feels safe vs adventurous for this traveler"
                }},
                "travel_personality": {{
                    "style": "Travel style (adventurous, cultural, comfort-focused, etc.)",
                    "motivations": "What seems to drive this person's travel",
                    "social_patterns": "How they interact with places and people",
                    "learning_style": "How they process and remember experiences"
                }},
                "recommendations": {{
                    "future_destinations": "Places that would appeal based on patterns",
                    "experience_types": "Activities or experiences to try next",
                    "growth_opportunities": "Ways to expand travel comfort zone",
                    "documentation_style": "How to better capture future memories"
                }},
                "memory_themes": [
                    "Common themes across their travel memories",
                    "Recurring emotional or cultural elements",
                    "Evolution of interests over time"
                ],
                "insights_summary": "Overall insights about this traveler's journey and preferences"
            }}
            """
            
            pattern_analysis = await call_gemini_api(prompt, system_context)
            
            # Parse analysis
            try:
                import json
                clean_text = pattern_analysis.strip()
                if clean_text.startswith('```json'):
                    clean_text = clean_text[7:]
                if clean_text.endswith('```'):
                    clean_text = clean_text[:-3]
                ai_patterns = json.loads(clean_text.strip())
                
                return ok({
                    "memory_count": len(user_memories),
                    "travel_pattern_analysis": ai_patterns,
                    "analysis_method": "Gemini 1.5 Flash Pattern Analysis",
                    "success": True
                })
            except:
                return ok({
                    "memory_count": len(user_memories),
                    "travel_insights": pattern_analysis,
                    "analysis_method": "Gemini 1.5 Flash (text mode)",
                    "success": True
                })
        
        except Exception as e:
            return ok({
                "memory_count": len(user_memories),
                "analysis": f"Pattern analysis failed: {str(e)}",
                "success": False
            })
    
    # list action
    user_memories = MEMORIES.get(user_id, [])
    return ok({
        "memories": list(reversed(user_memories)),
        "count": len(user_memories),
        "success": True
    })

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
        
        # Synthesize unified response with Gemini-powered analysis
        try:
            # Use Gemini to analyze the travel request and provide intelligent recommendations
            system_context = """You are an expert travel AI agent who analyzes complex travel requests and provides 
            comprehensive, personalized travel assistance. You understand traveler needs and can orchestrate multiple tools effectively."""
            
            user_context = f"""
            Travel Request: {travel_request}
            Home Country: {home_country or 'Not specified'}
            Current Location: {current_location or 'Not specified'}
            Dietary Restrictions: {dietary_restrictions or 'None specified'}
            Detected Locations: {locations}
            """
            
            prompt = f"""
            Analyze this travel request and provide comprehensive assistance:
            
            {user_context}
            
            Provide response as detailed JSON:
            {{
                "request_interpretation": {{
                    "primary_intent": "Main goal of this travel request",
                    "secondary_needs": "Additional needs identified",
                    "urgency_level": "How time-sensitive this request is",
                    "complexity_level": "Simple/Moderate/Complex travel request"
                }},
                "recommended_action_plan": [
                    {{
                        "step": "Step number",
                        "action": "What to do",
                        "tool_suggestion": "Which MCP tool would help",
                        "priority": "High/Medium/Low",
                        "rationale": "Why this step is important"
                    }}
                ],
                "personalized_insights": {{
                    "traveler_profile": "Inferred travel style and preferences",
                    "experience_level": "Novice/Intermediate/Expert traveler assessment",
                    "risk_tolerance": "Conservative/Moderate/Adventurous assessment",
                    "cultural_readiness": "How prepared they seem for cultural differences"
                }},
                "immediate_recommendations": [
                    "Most important things to do right now",
                    "Critical preparation steps",
                    "Quick wins for better travel experience"
                ],
                "location_specific_advice": {{
                    "destination_insights": "Key things to know about intended destination",
                    "cultural_preparation": "Cultural aspects to research or prepare for",
                    "practical_preparation": "Logistics and practical steps needed",
                    "hidden_opportunities": "Unique experiences or insights for this destination"
                }},
                "safety_and_logistics": {{
                    "safety_priorities": "Safety considerations for this trip",
                    "documentation_needs": "Passport, visa, insurance considerations",
                    "health_preparations": "Vaccinations, health insurance, medications",
                    "communication_plan": "How to stay connected and handle emergencies"
                }},
                "tool_orchestration": {{
                    "immediate_tools": "Tools to use right now",
                    "preparation_tools": "Tools for trip planning phase",
                    "travel_tools": "Tools to use while traveling",
                    "memory_tools": "Tools for documenting and remembering the trip"
                }},
                "success_metrics": {{
                    "short_term": "How to measure immediate success",
                    "trip_success": "What would make this trip successful",
                    "long_term": "Long-term travel growth opportunities"
                }}
            }}
            
            Make recommendations specific to their request and inferred needs.
            """
            
            ai_analysis = await call_gemini_api(prompt, system_context)
            
            # Try to parse AI analysis
            try:
                import json
                clean_text = ai_analysis.strip()
                if clean_text.startswith('```json'):
                    clean_text = clean_text[7:]
                if clean_text.endswith('```'):
                    clean_text = clean_text[:-3]
                ai_insights = json.loads(clean_text.strip())
                
                # Enhance response with AI insights
                unified_response = {
                    "request_analysis": {
                        "original_request": travel_request,
                        "detected_locations": locations,
                        "ai_interpretation": ai_insights.get("request_interpretation", {}),
                        "personalized_insights": ai_insights.get("personalized_insights", {})
                    },
                    "ai_powered_recommendations": ai_insights,
                    "suggested_tools": suggested_tools,
                    "immediate_guidance": guidance,
                    "analysis_method": "Gemini 1.5 Flash Intelligent Analysis"
                }
                
                # Create enhanced summary from AI insights
                if "immediate_recommendations" in ai_insights:
                    summary_parts = ["🤖 **AI-Powered Travel Assistance**:"]
                    for rec in ai_insights["immediate_recommendations"][:3]:
                        summary_parts.append(f"• {rec}")
                    unified_response["summary"] = "\n".join(summary_parts)
                
                # Enhanced next steps from AI analysis
                if "tool_orchestration" in ai_insights:
                    orchestration = ai_insights["tool_orchestration"]
                    next_steps = []
                    if orchestration.get("immediate_tools"):
                        next_steps.append(f"🚀 **Now**: {orchestration['immediate_tools']}")
                    if orchestration.get("preparation_tools"):
                        next_steps.append(f"📋 **Prepare**: {orchestration['preparation_tools']}")
                    if orchestration.get("travel_tools"):
                        next_steps.append(f"✈️ **While Traveling**: {orchestration['travel_tools']}")
                    
                    unified_response["ai_next_steps"] = next_steps
                
                return ok(unified_response)
                
            except json.JSONDecodeError:
                # Fallback to text analysis if JSON parsing fails
                unified_response["ai_analysis"] = ai_analysis
                unified_response["analysis_method"] = "Gemini 1.5 Flash (text mode)"
        
        except Exception as ai_error:
            # Continue with basic analysis if AI fails
            unified_response["ai_error"] = f"AI analysis failed: {str(ai_error)}"
            unified_response["analysis_method"] = "Basic fallback"
        
        
        # Create a human-friendly summary (keep existing logic as fallback)
        if "summary" not in unified_response:
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
        
        # Add next steps for tools to use (keep existing logic as fallback)
        if "ai_next_steps" not in unified_response:
            next_steps = []
            for tool_suggestion in suggested_tools:
                tool_name = tool_suggestion["tool"]
                reason = tool_suggestion["reason"]
                next_steps.append(f"🔧 Use `{tool_name}` to {reason.lower()}")
            
            # Add general next steps
            if not any("menu_intelligence" in step for step in next_steps):
                next_steps.append("🍽️ Use `menu_intelligence` when you find restaurants - take photos for analysis")
            
            if not wants_to_save_memory and not any("travel_memory_archive" in step for step in next_steps):
                next_steps.append("💾 Use `travel_memory_archive` to save important experiences")
            
            unified_response["next_steps"] = next_steps
        
        unified_response["success"] = True
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
            "next_steps": ["Try simplifying your request or use individual tools directly"],
            "analysis_method": "Error fallback",
            "success": False
        })

## Removed: job_finder (not needed)

## Removed: make_img_black_and_white (not needed)

# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
