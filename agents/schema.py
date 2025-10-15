"""
Very small schema validator for MR-SR outputs.
Checks that the required top-level keys exist.
"""
def validate_basic_structure(obj):
    req = ["asset", "market_state", "strategy", "action"]
    missing = [k for k in req if k not in obj]
    if missing:
        return False, f"missing keys: {missing}"
    # simple type checks (optional)
    if not isinstance(obj["market_state"], dict):
        return False, "market_state must be an object"
    return True, ""
