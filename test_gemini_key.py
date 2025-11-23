#!/usr/bin/env python3
"""Test Gemini API key with different models."""

import requests

API_KEY = "AIzaSyC7JjdAl0wf1mJ9cPSVTATb2NhY7FiLXUo"

# Models to test
models = [
    "gemini-pro",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-exp-1206"
]

for model in models:
    print(f"\nTesting model: {model}")
    print("="*50)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
    payload = {
        "contents": [{
            "parts": [{"text": "Say 'working'"}]
        }]
    }

    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result:
                text = result['candidates'][0]['content']['parts'][0]['text']
                print(f"✅ SUCCESS! Response: {text[:100]}")
                print(f"\n🎯 Use this model: --model {model}")
                break
        else:
            print(f"❌ Error: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Exception: {e}")
else:
    print("\n\n⚠️ No working models found!")
    print("Please check:")
    print("1. Visit https://aistudio.google.com/app/apikey")
    print("2. Make sure your API key is activated")
    print("3. Try generating a new API key")
