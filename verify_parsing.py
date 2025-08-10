#!/usr/bin/env python3
"""
Quick test to verify the correct origin/destination parsing
"""

def test_specific_queries():
    """Test the exact user queries to verify correct parsing"""
    
    import re
    
    def parse_travel_query_fixed(query: str) -> dict:
        """Fixed version of the parse function"""
        parsed = {
            "origin": "",
            "destination": "", 
            "travel_date": "",
            "transport_type": "all"
        }
        
        query_lower = query.lower()
        
        # Extract transport type preferences
        if any(word in query_lower for word in ["flight", "flights", "fly", "air"]):
            parsed["transport_type"] = "flights"
        elif "all" in query_lower or "modes" in query_lower:
            parsed["transport_type"] = "all"
        
        # Extract origin and destination with correct logic
        # "travel to X from Y" means: going FROM Y TO X (Y=origin, X=destination)
        travel_to_from_pattern = r"travel\s+to\s+([a-zA-Z\s]+?)\s+from\s+([a-zA-Z\s]+?)(?:\s+on|\s+for|\s*$)"
        travel_match = re.search(travel_to_from_pattern, query, re.IGNORECASE)
        
        if travel_match:
            parsed["destination"] = travel_match.group(1).strip().title()  # X (where going TO)
            parsed["origin"] = travel_match.group(2).strip().title()       # Y (where coming FROM)
        else:
            # Try "from X to Y" pattern
            from_to_pattern = r"from\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?:\s+on|\s+for|\s*$)"
            from_to_match = re.search(from_to_pattern, query, re.IGNORECASE)
            
            if from_to_match:
                parsed["origin"] = from_to_match.group(1).strip().title()      # X (FROM)
                parsed["destination"] = from_to_match.group(2).strip().title() # Y (TO)
        
        # Extract date
        date_patterns = [
            r"on\s+(\d{1,2}(?:st|nd|rd|th)?\s+[a-zA-Z]+\s+\d{4})",
            r"(\d{1,2}(?:st|nd|rd|th)?\s+[a-zA-Z]+\s+\d{4})"
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, query, re.IGNORECASE)
            if date_match:
                parsed["travel_date"] = date_match.group(1).strip()
                break
        
        return parsed
    
    test_queries = [
        {
            "query": "I am looking to travel to Moscow from Kolkata on 28th August 2025. Can you list all the available flights with their timings and prices?",
            "expected": {
                "origin": "Kolkata",
                "destination": "Moscow",
                "travel_date": "28th August 2025",
                "transport_type": "flights"
            }
        },
        {
            "query": "I am looking to travel to Bhubaneswar from Kolkata on 28th August 2025. Can you list all the available modes of transport with their timings and prices?",
            "expected": {
                "origin": "Kolkata",
                "destination": "Bhubaneswar", 
                "travel_date": "28th August 2025",
                "transport_type": "all"
            }
        }
    ]
    
    print("🔍 Testing User Example Queries with Fixed Parsing")
    print("=" * 60)
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}:")
        print(f'"{test["query"]}"')
        
        parsed = parse_travel_query_fixed(test["query"])
        expected = test["expected"]
        
        print(f"\n🎯 Expected:")
        print(f"   Origin: {expected['origin']}")
        print(f"   Destination: {expected['destination']}")
        print(f"   Date: {expected['travel_date']}")
        print(f"   Transport: {expected['transport_type']}")
        
        print(f"\n📊 Parsed:")
        print(f"   Origin: {parsed['origin']}")
        print(f"   Destination: {parsed['destination']}")
        print(f"   Date: {parsed['travel_date']}")
        print(f"   Transport: {parsed['transport_type']}")
        
        # Check correctness
        correct = (
            parsed["origin"] == expected["origin"] and
            parsed["destination"] == expected["destination"] and
            parsed["travel_date"] == expected["travel_date"] and
            parsed["transport_type"] == expected["transport_type"]
        )
        
        print(f"\n✅ Result: {'CORRECT ✅' if correct else 'INCORRECT ❌'}")
        print("=" * 60)
    
    print(f"\n🎉 The parsing logic should now correctly handle:")
    print(f"✅ 'travel to X from Y' → origin=Y, destination=X")
    print(f"✅ 'from X to Y' → origin=X, destination=Y")

if __name__ == "__main__":
    test_specific_queries()
