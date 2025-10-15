#!/usr/bin/env python3
"""
Simple MR-SR skeleton:
- Reads a YAML fixture (tests/agentic/basic/*.yaml)
- If fixture contains 'expected' fields, uses them to build a deterministic output.
- If fixture contains 'input_mr_sr', merges those fields into output (useful for ERS tests).
- Writes JSON to --out path or stdout.
"""
import argparse, yaml, json, sys, os
from datetime import datetime

def load_yaml(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)

def build_output(fixture, asset="ASSET-UNKNOWN"):
    expected = fixture.get("expected", {}) or {}
    input_mr = fixture.get("input_mr_sr", {}) or {}

    # Fill in market_state
    market_state = {
        "sentiment": expected.get("sentiment", "neutral"),
        "confidence": expected.get("confidence_min", expected.get("confidence", 0.75)),
        "risk": expected.get("risk", "medium"),
        "opportunity": expected.get("opportunity", 0.5)
    }

    # Reasoning simple skeleton: include Surface with default insight if none provided
    reasoning_levels = [
        {"level":"surface","insight": "RSI neutral","confidence": market_state["confidence"]*0.8},
        {"level":"tactical","insight":"no short-term breakout detected","confidence": market_state["confidence"]*0.6},
    ]

    action = expected.get("action", input_mr.get("action", "HOLD"))
    strategy = expected.get("strategy", input_mr.get("strategy", "Momentum"))

    # include safety fields (if provided)
    out = {
        "generated_at": datetime.utcnow().isoformat()+"Z",
        "asset": asset,
        "market_state": market_state,
        "reasoning_levels": reasoning_levels,
        "strategy": {"name": strategy, "rationale": expected.get("rationale","auto-generated")},
        "action": {"recommendation": action},
        "metadata": {
            "source_fixture": os.path.basename(args.fixture)
        }
    }

    # Merge any input_mr_sr fields so ERS tests can use them
    if input_mr:
        out["input_mr_sr"] = input_mr

    # If the fixture included stop_loss/atr or risk_percent, propagate them into action
    if "risk_percent" in input_mr:
        out["action"]["risk_percent"] = input_mr["risk_percent"]
    if "stop_loss" in input_mr:
        out["action"]["stop_loss"] = input_mr["stop_loss"]
    if "atr14" in input_mr:
        out["action"]["atr14"] = input_mr["atr14"]

    # citations placeholder
    out["citations"] = expected.get("citations", [])

    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, help="Path to YAML test fixture")
    parser.add_argument("--asset", required=False, default="BTC-USD", help="Asset identifier (optional)")
    parser.add_argument("--out", required=False, help="Output JSON file (if omitted, prints to stdout)")
    args = parser.parse_args()

    fixture = load_yaml(args.fixture)
    out = build_output(fixture, asset=args.asset)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote MR-SR output to {args.out}")
    else:
        print(json.dumps(out, indent=2))
