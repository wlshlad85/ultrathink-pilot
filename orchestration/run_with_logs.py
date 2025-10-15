#!/usr/bin/env python3
"""
Run MR-SR -> ERS across fixtures and collect simple observability metrics:
- per-fixture: mr_latency_ms, ers_latency_ms, decision, reasons
- aggregate: counts, latencies list (for p50/p95 later)
Writes:
 - eval/metrics.json
 - eval/run_report.md (updated with metrics link)
"""
import subprocess, glob, os, json, time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "agentic" / "basic"
TMPDIR = Path("/tmp/ultrathink_orch")
TMPDIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = ROOT / "eval" / "metrics.json"
REPORT_PATH = ROOT / "eval" / "run_report.md"

def run_cmd_timed(cmd):
    start = time.time()
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    end = time.time()
    elapsed_ms = int((end - start) * 1000)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\\nSTDOUT:\\n{proc.stdout}\\nSTDERR:\\n{proc.stderr}")
    return elapsed_ms, proc.stdout, proc.stderr

def process_fixture(fpath):
    base = Path(fpath).name
    mr_out = TMPDIR / f"mr_{base}.json"
    ers_out = TMPDIR / f"ers_{base}.json"

    mr_lat_ms, _, _ = run_cmd_timed(f'python agents/mr_sr.py --fixture "{fpath}" --asset BTC-USD --out "{mr_out}"')
    ers_lat_ms, _, _ = run_cmd_timed(f'python agents/ers.py --in "{mr_out}" --out "{ers_out}"')

    with open(ers_out,'r') as f:
        ers = json.load(f)

    return {
        "fixture": base,
        "mr_latency_ms": mr_lat_ms,
        "ers_latency_ms": ers_lat_ms,
        "decision": ers.get("decision"),
        "reasons": ers.get("reasons", []),
    }

def p95(values):
    if not values:
        return None
    values_sorted = sorted(values)
    idx = int(0.95 * (len(values_sorted)-1))
    return values_sorted[idx]

def main():
    fixtures = sorted(glob.glob(str(FIXTURE_DIR / "*.yaml")))
    metrics = {"runs": [], "summary": {}}
    for fx in fixtures:
        try:
            r = process_fixture(fx)
            print(f"{r['fixture']}: {r['decision']} (mr {r['mr_latency_ms']}ms, ers {r['ers_latency_ms']}ms)")
            metrics["runs"].append(r)
        except Exception as e:
            print(f"ERROR processing {fx}: {e}")
            metrics["runs"].append({"fixture": Path(fx).name, "error": str(e)})
    # aggregate
    mr_latencies = [r["mr_latency_ms"] for r in metrics["runs"] if "mr_latency_ms" in r]
    ers_latencies = [r["ers_latency_ms"] for r in metrics["runs"] if "ers_latency_ms" in r]
    decisions = [r["decision"] for r in metrics["runs"] if "decision" in r]

    metrics["summary"]["count"] = len(metrics["runs"])
    metrics["summary"]["mr_p50_ms"] = median(mr_latencies) if mr_latencies else None
    metrics["summary"]["ers_p50_ms"] = median(ers_latencies) if ers_latencies else None
    metrics["summary"]["mr_p95_ms"] = p95(mr_latencies)
    metrics["summary"]["ers_p95_ms"] = p95(ers_latencies)
    metrics["summary"]["decisions"] = {d: decisions.count(d) for d in set(decisions)}

    # placeholder: token estimates / cost - set to 0 for now until we instrument token counting
    metrics["summary"]["estimated_token_in"] = 0
    metrics["summary"]["estimated_token_out"] = 0
    metrics["summary"]["estimated_cost_gbp"] = 0.0

    # write metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # update report with link to metrics
    with open(REPORT_PATH, "w") as f:
        f.write("# Orchestration run report (with metrics)\\n\\n")
        f.write(f"Run at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\\n\\n")
        f.write("## Summary metrics\\n\\n")
        for k,v in metrics["summary"].items():
            f.write(f"- {k}: {v}\\n")
        f.write("\\n## Per-fixture results\\n\\n")
        f.write("| fixture | decision | mr_ms | ers_ms | reasons |\\n")
        f.write("|---|---:|---:|---:|---|\\n")
        for r in metrics["runs"]:
            reasons = '<br/>'.join(r.get("reasons", [])) if "reasons" in r else (r.get("error",""))
            f.write(f"| {r['fixture']} | {r.get('decision','error')} | {r.get('mr_latency_ms','-')} | {r.get('ers_latency_ms','-')} | {reasons} |\\n")

    print(f"Metrics written to {METRICS_PATH}")
    print(f"Report updated at {REPORT_PATH}")

if __name__ == '__main__':
    main()
