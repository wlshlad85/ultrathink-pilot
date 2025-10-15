#!/usr/bin/env python3
"""
MR-SR with optional OpenAI integration (safe fallback).
- Calls agents.model_backends.generate_with_openai if OPENAI_API_KEY present.
- Validates minimal schema via agents.schema.validate_basic_structure.
- Falls back to deterministic output on error.
"""
import argparse, yaml, json, os, sys, re
from datetime import datetime
from agents import model_backends
from agents import schema as schema_mod

def load_yaml(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)

def deterministic_output(fixture, asset="ASSET-UNKNOWN"):
    expected = fixture.get("expected", {}) or {}
    input_mr = fixture.get("input_mr_sr", {}) or {}
    action = expected.get("action", input_mr.get("action", "HOLD"))
    strategy = expected.get("strategy", input_mr.get("strategy", "Momentum"))
    out = {
        "generated_at": datetime.utcnow().isoformat()+"Z",
        "asset": asset,
        "market_state": {
            "sentiment": expected.get("sentiment", "neutral"),
            "confidence": expected.get("confidence_min", 0.75),
            "risk": expected.get("risk", "medium"),
            "opportunity": expected.get("opportunity", 0.5)
        },
        "reasoning_levels": [{"level":"surface","insight":"RSI neutral","confidence":0.6}],
        "strategy": {"name": strategy, "rationale": expected.get("rationale","auto-generated")},
        "action": {"recommendation": action},
        "citations": expected.get("citations", [])
    }
    if input_mr:
        out["input_mr_sr"] = input_mr
    return out

def build_prompt_from_fixture(fx, asset):
    t = f"""
Produce a concise JSON recommendation for asset {asset}.
Return ONLY a single JSON object with keys:
  - asset
  - market_state: {{ sentiment, confidence (0-1), risk, opportunity(optional) }}
  - strategy: {{ name, rationale }}
  - action: {{ recommendation, optional risk_percent, optional stop_loss }}
  - citations: [ ... ]

Context:
{json.dumps(fx, indent=2)}

Important: Return valid JSON only (no surrounding commentary).
"""
    return t

def extract_json_from_text(text):
    # Try to extract the first JSON object in text
    m = re.search(r'\{[\s\S]*\}', text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def call_model(prompt):
    """
    Returns (obj, err). If obj is None, err explains reason.
    """
    # Prefer real OpenAI if key present
    if os.getenv("OPENAI_API_KEY"):
        try:
            raw = model_backends.generate_with_openai(prompt,
                                                     model=os.getenv("OPENAI_MODEL","gpt-4o"),
                                                     max_tokens=int(os.getenv("OPENAI_MAX_TOKENS","400")),
                                                     temperature=float(os.getenv("OPENAI_TEMPERATURE","0.0")))
            parsed = extract_json_from_text(raw)
            if parsed is None:
                return None, f"could not parse JSON from model output (raw first 400 chars): {raw[:400]}"
            return parsed, None
        except Exception as e:
            return None, f"openai call error: {e}"
    else:
        # mock path
        try:
            raw = model_backends.generate_mock(prompt)
            parsed = json.loads(raw)
            return parsed, None
        except Exception as e:
            return None, f"mock generation error: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, help="Path to YAML test fixture")
    parser.add_argument("--asset", required=False, default="BTC-USD", help="Asset identifier (optional)")
    parser.add_argument("--out", required=False, help="Output JSON file (if omitted, prints to stdout)")
    args = parser.parse_args()

    fx = load_yaml(args.fixture)
    prompt = build_prompt_from_fixture(fx, args.asset)

    model_obj, err = call_model(prompt)
    if model_obj:
        ok, reason = schema_mod.validate_basic_structure(model_obj)
        if not ok:
            out = deterministic_output(fx, asset=args.asset)
            out["_fallback_reason"] = f"schema validation failed: {reason}"
        else:
            out = model_obj
    else:
        out = deterministic_output(fx, asset=args.asset)
        out["_fallback_reason"] = f"model failure: {err}"

    out.setdefault("generated_at", datetime.utcnow().isoformat()+"Z")
    out.setdefault("metadata", {})["source_fixture"] = os.path.basename(args.fixture)
    out.setdefault("citations", out.get("citations", []))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote MR-SR output to {args.out}")
    else:
        print(json.dumps(out, indent=2))
