#!/usr/bin/env python3
"""
Simple two-agent orchestrator for the pilot.
Runs MR-SR -> ERS for each fixture under tests/agentic/basic and produces a run_report.md.
"""
import subprocess, glob, os, json, sys, tempfile
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "agentic" / "basic"
TMPDIR = Path(tempfile.gettempdir()) / "ultrathink_orch"
TMPDIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = ROOT / "eval" / "run_report.md"

def run_cmd(cmd, capture=False):
    if capture:
        return subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    else:
        return subprocess.run(cmd, shell=True, check=True)

def process_fixture(fpath):
    base = Path(fpath).name
    mr_out = TMPDIR / f"mr_{base}.json"
    ers_out = TMPDIR / f"ers_{base}.json"

    # Set PYTHONPATH to include project root
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT)

    # run MR-SR
    mr_cmd = [sys.executable, str(ROOT / "agents" / "mr_sr.py"), "--fixture", str(fpath), "--asset", "BTC-USD", "--out", str(mr_out)]
    subprocess.run(mr_cmd, check=True, env=env)

    # run ERS
    ers_cmd = [sys.executable, str(ROOT / "agents" / "ers.py"), "--in", str(mr_out), "--out", str(ers_out)]
    subprocess.run(ers_cmd, check=True, env=env)

    # load ERS result
    with open(ers_out,'r') as f:
        ers = json.load(f)
    return base, ers

def main(dry=False):
    fixtures = sorted(glob.glob(str(FIXTURE_DIR / "*.yaml")))
    summary = []
    for fx in fixtures:
        try:
            name, ers = process_fixture(fx)
            summary.append((name, ers["decision"], ers.get("reasons", [])))
            print(f"{name}: {ers['decision']}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR running {fx}: {e}", file=sys.stderr)
            summary.append((Path(fx).name, "error", [str(e)]))

    # write a simple markdown report
    with open(REPORT_PATH, "w") as f:
        f.write(f"# Orchestration run report\n\n")
        f.write(f"Run at: {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write("| fixture | decision | reasons |\n")
        f.write("|---|---|---|\n")
        for name, decision, reasons in summary:
            rs = "<br/>".join(reasons) if reasons else ""
            f.write(f"| {name} | {decision} | {rs} |\n")

    print("\nSummary:")
    for name, decision, reasons in summary:
        print(f"- {name} -> {decision}")

    print(f"\nReport written to: {REPORT_PATH}")
    if dry:
        print("Dry run flag set — no further actions.")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Run but mark as dry")
    args = p.parse_args()
    main(dry=args.dry_run)
