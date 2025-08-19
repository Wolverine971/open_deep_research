#!/usr/bin/env python3
"""Test script to verify TavilySearch is working correctly"""

import os
from langchain_tavily import TavilySearch

# Check if API key is set
api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    print("❌ TAVILY_API_KEY not set in environment")
    exit(1)

print("✅ TAVILY_API_KEY found")

# Test basic search
try:
    tavily = TavilySearch(max_results=2)
    response = tavily.invoke({"query": "Python programming"})
    
    print(f"\n📊 Response type: {type(response)}")
    print(f"📊 Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
    
    if 'results' in response:
        print(f"\n✅ Found {len(response['results'])} results")
        for i, result in enumerate(response['results'][:2], 1):
            print(f"\nResult {i}:")
            print(f"  Title: {result.get('title', 'N/A')}")
            print(f"  URL: {result.get('url', 'N/A')}")
            print(f"  Score: {result.get('score', 0):.2f}")
    else:
        print("❌ No 'results' key in response")
        
except Exception as e:
    print(f"❌ Error: {e}")