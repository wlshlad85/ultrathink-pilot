import os
import json
try:
    import openai
except Exception:
    openai = None
from dotenv import load_dotenv

load_dotenv()  # loads .env if present

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
# Configure openai if key present
if OPENAI_KEY and openai:
    openai.api_key = OPENAI_KEY

# Helper: call OpenAI ChatCompletion (simple wrapper)
def generate_with_openai(prompt, model="gpt-4o", max_tokens=400, temperature=0.0):
    """
    Generate a response using OpenAI ChatCompletion API.
    - temperature=0.0 for deterministic-ish outputs (note: absolute determinism is not guaranteed).
    - model default can be changed; use a model you have access to.
    """
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    if not openai:
        raise RuntimeError("openai package not available in environment")

    # Build messages
    messages = [
        {"role": "system", "content": "You are an assistant that returns a JSON object matching a provided schema. Be concise."},
        {"role": "user", "content": prompt}
    ]

    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1
    )
    # Extract text
    text = resp.choices[0].message["content"]
    return text

# Mock generator (deterministic fallback)
def generate_mock(prompt):
    """
    Deterministic mock generator: parse prompt heuristics and return JSON string.
    Used for development and CI when no API key is available.
    """
    # Minimal heuristic: if prompt mentions BTC -> BUY Momentum else HOLD
    obj = {
        "asset": "BTC-USD" if "BTC" in prompt.upper() else "ASSET-UNKNOWN",
        "market_state": {"sentiment": "neutral", "confidence": 0.75, "risk": "medium", "opportunity": 0.5},
        "strategy": {"name": "Momentum", "rationale": "mocked: default momentum signal"},
        "action": {"recommendation": "BUY" if "BTC" in prompt.upper() else "HOLD"},
        "citations": []
    }
    return json.dumps(obj, indent=2)
