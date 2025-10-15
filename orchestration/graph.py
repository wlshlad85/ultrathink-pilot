#!/usr/bin/env python3
"""
Simple two-agent orchestrator for the pilot.
Runs MR-SR -> ERS for each fixture under tests/agentic/basic and produces a run_report.md.
"""
import subprocess, glob, os, json, sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "agentic" / "basic"
TMPDIR = Path("/tmp/ultrathink_orch")
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

    # run MR-SR
    run_cmd(f'python agents/mr_sr.py --fixture "{fpath}" --asset BTC-USD --out "{mr_out}"')
    # run ERS
    run_cmd(f'python agents/ers.py --in "{mr_out}" --out "{ers_out}"')

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
        f.write(f"Run at: {datetime.utcnow().isoformat()}Z\n\n")
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
