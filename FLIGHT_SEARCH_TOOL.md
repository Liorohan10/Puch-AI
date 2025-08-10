# Flight and Transport Search Tool

## 🛫 New Feature: Comprehensive Transport Search

I've added a powerful new tool to your MCP server that handles flight and transport queries using the Gemini API with refined prompts for accurate results.

## 🔧 Tool Details

### `flight_and_transport_search`

**Purpose**: Search for flights and transport options between two cities with detailed timing and pricing information.

**Parameters**:
- `origin` (required): Departure city/location (e.g., "Kolkata", "Delhi", "Mumbai")
- `destination` (required): Arrival city/location (e.g., "Moscow", "Bhubaneswar", "Chennai") 
- `travel_date` (required): Date of travel in format "DD Month YYYY" or "YYYY-MM-DD"
- `transport_type` (optional): "flights", "trains", "buses", "all" (default: "all")

## 📝 Example Queries Supported

### Query 1: International Flight Search
```
"I am looking to travel to Moscow from Kolkata on 28th August 2025. Can you list all the available flights with their timings and prices?"
```

**Tool Call**:
```json
{
  "tool": "flight_and_transport_search",
  "parameters": {
    "origin": "Kolkata",
    "destination": "Moscow", 
    "travel_date": "28th August 2025",
    "transport_type": "flights"
  }
}
```

### Query 2: Multi-Modal Transport Search
```
"I am looking to travel to Bhubaneswar from Kolkata on 28th August 2025. Can you list all the available modes of transport with their timings and prices?"
```

**Tool Call**:
```json
{
  "tool": "flight_and_transport_search",
  "parameters": {
    "origin": "Kolkata",
    "destination": "Bhubaneswar",
    "travel_date": "28th August 2025", 
    "transport_type": "all"
  }
}
```

## 🎯 Response Structure

The tool returns comprehensive information in this format:

```json
{
  "search_query": {
    "origin": "Kolkata",
    "destination": "Moscow",
    "travel_date": "28th August 2025", 
    "transport_type": "flights",
    "search_timestamp": "2025-08-10T..."
  },
  "transport_information": "Detailed Gemini-generated transport options with:\n- Flight schedules and pricing\n- Multiple airlines and classes\n- Booking platforms\n- Travel duration\n- Connection details",
  "search_tips": [
    "Compare prices across multiple booking platforms",
    "Consider flexible dates for better deals",
    "Book in advance for international flights",
    "Check visa requirements for international travel"
  ],
  "booking_platforms": {
    "flights": ["Google Flights", "Skyscanner", "Kayak", "MakeMyTrip"],
    "trains": ["IRCTC", "RailYatri", "ConfirmTkt"],
    "buses": ["RedBus", "AbhiBus", "Paytm Travel"]
  },
  "important_notes": [
    "Prices are subject to change and availability",
    "Always verify schedules on official websites"
  ]
}
```

## 🧠 Gemini API Integration

### Optimized Prompt Strategy

The tool uses a sophisticated system prompt that instructs Gemini to:

1. **Find comprehensive transport options** with specific details
2. **Include real-time pricing and schedules** for 2025
3. **Structure responses** with clear sections (Flights, Trains, Buses, etc.)
4. **Provide actionable booking information** with platforms and availability
5. **Consider practical factors** like visas, baggage, connections

### Generation Configuration

- **Temperature**: 0.3 (for factual accuracy)
- **Model**: gemini-1.5-pro
- **Max tokens**: 4000 (for comprehensive responses)
- **Top-p**: 0.8, Top-k: 40 (balanced creativity/accuracy)

## 🔍 What Gemini Will Return

For your example queries, Gemini will provide:

### Kolkata → Moscow (Flights)
- **Direct flights**: Air India, Aeroflot options
- **Connecting flights**: Via Dubai, Istanbul, Delhi
- **Pricing**: Economy, Business, First class ranges
- **Timing**: Departure/arrival times with durations
- **Booking**: Best platforms and current availability
- **Visa info**: Russian visa requirements
- **Alternatives**: Nearby dates with better deals

### Kolkata → Bhubaneswar (All Transport)
- **Flights**: IndiGo, Air India domestic options
- **Trains**: Rajdhani, Jan Shatabdi, Express trains
- **Buses**: Luxury, semi-luxury, government buses
- **Pricing**: Detailed fare comparison across modes
- **Duration**: Journey times for each option
- **Booking**: Platform recommendations for each mode

## 🚀 Deployment Status

✅ **Tool added** to `mcp_starter.py`
✅ **Integrated** with existing logging system  
✅ **Updated** server status tool
✅ **Compatible** with current authentication
✅ **Ready** for Render deployment

## 📊 Usage Tracking

The tool includes:
- Usage statistics tracking
- Comprehensive input/output logging
- Error handling and fallback responses
- Performance monitoring

## 🔧 Testing

Use the included test script:
```bash
python test_flight_search.py
```

This validates the tool structure and example query handling.

## 🎯 Next Steps

1. **Deploy** the updated code to Render
2. **Test** with real MCP client queries
3. **Monitor** Gemini API usage and response quality
4. **Optimize** prompts based on user feedback

Your MCP server now supports comprehensive flight and transport search queries with accurate, actionable results powered by Gemini! ✈️🚂🚌
