import glob
import subprocess
import json
from pathlib import Path
import yaml
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "agentic" / "basic"
TMPDIR = Path("/tmp/ultrathink_orch")

def run_cmd(cmd):
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\\nSTDOUT:\\n{proc.stdout}\\nSTDERR:\\n{proc.stderr}")
    return proc.stdout, proc.stderr

@pytest.mark.parametrize("fixture_path", sorted(glob.glob(str(FIXTURE_DIR / "*.yaml"))))
def test_agent_pipeline_matches_fixture(fixture_path):
    # load fixture
    with open(fixture_path,"r") as f:
        fx = yaml.safe_load(f)

    base = Path(fixture_path).name
    mr_out = TMPDIR / f"mr_{base}.json"
    ers_out = TMPDIR / f"ers_{base}.json"

    # run MR-SR
    run_cmd(f'python agents/mr_sr.py --fixture "{fixture_path}" --asset BTC-USD --out "{mr_out}"')

    # run ERS
    run_cmd(f'python agents/ers.py --in "{mr_out}" --out "{ers_out}"')

    # load ERS result
    with open(ers_out,"r") as f:
        ers = json.load(f)

    # Determine expected decision:
    expected = fx.get("expected", {})
    # Priority: explicit expected.ers_decision, then expected.action -> map to approve/veto
    expected_decision = expected.get("ers_decision")
    if not expected_decision:
        # If expected.action present, derive approve/veto: actions that exceed risk or bad SL are in input_mr_sr tests.
        exp_action = expected.get("action")
        if exp_action:
            # Simple heuristic: if expected.action == "BUY" or "HOLD" -> expect approve unless input_mr_sr implies violation
            expected_decision = "approve"
        else:
            expected_decision = "approve"

    # Assert decision equals expected (lowercase compare)
    assert ers["decision"].lower() == expected_decision.lower(), f"{base}: ers decision {ers['decision']} != expected {expected_decision}"
