#!/usr/bin/env python3
"""
Simple ERS skeleton:
- Reads MR-SR JSON (from --in)
- Applies simple policy checks:
   * If input_mr_sr.risk_percent > 2.0 -> veto
   * Else if stop_loss and atr14 present -> require stop_loss >= 1.2 * atr14
   * Else approve
- Outputs a small JSON with decision and reasons
"""
import argparse, json, sys, os
from datetime import datetime

def load_json(path):
    with open(path,'r') as f:
        return json.load(f)

def evaluate(mr):
    reasons = []
    decision = "approve"

    # check embedded input fields first
    input_mr = mr.get("input_mr_sr", {}) or {}
    rp = input_mr.get("risk_percent", mr.get("action", {}).get("risk_percent"))
    sl = input_mr.get("stop_loss", mr.get("action", {}).get("stop_loss"))
    atr14 = input_mr.get("atr14", mr.get("action", {}).get("atr14"))

    # Rule: risk percent > 2% -> veto
    if rp is not None:
        try:
            rp_float = float(rp)
            if rp_float > 2.0:
                decision = "veto"
                reasons.append(f"risk_percent {rp_float} > 2.0")
        except Exception:
            reasons.append(f"invalid risk_percent: {rp}")

    # Rule: if stop-loss and atr present, require stop_loss >= 1.2 * atr14
    if decision != "veto" and sl is not None and atr14 is not None:
        try:
            slf = float(sl)
            atrf = float(atr14)
            if slf < 1.2 * atrf:
                decision = "veto"
                reasons.append(f"stop_loss {slf} < 1.2 * atr14 ({1.2*atrf})")
            else:
                reasons.append(f"stop_loss {slf} >= 1.2 * atr14 ({1.2*atrf})")
        except Exception:
            reasons.append("invalid stop_loss or atr14 values")

    if not reasons:
        reasons.append("no policy triggers, approved")

    return {
        "evaluated_at": datetime.utcnow().isoformat()+"Z",
        "decision": decision,
        "reasons": reasons,
        "input_metadata": {
            "source_asset": mr.get("asset"),
            "source_fixture": mr.get("metadata", {}).get("source_fixture")
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True, help="Input MR-SR JSON file")
    parser.add_argument("--out", required=False, help="Output JSON file")
    args = parser.parse_args()

    mr = load_json(args.infile)
    result = evaluate(mr)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote ERS output to {args.out}")
    else:
        print(json.dumps(result, indent=2))
