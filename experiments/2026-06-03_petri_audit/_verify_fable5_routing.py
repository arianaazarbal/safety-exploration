"""Verify Fable 5 routing: read each completed Fable 5 cell's eval log and report
which model the responses actually came from. Flags any cell where any sample
returned a different model id than the requested target (e.g. routed to Opus 4.8).
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import json

from inspect_ai.log import read_eval_log

BASE = Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/results/by_setting/coding_kill_subagent")
EXPECTED_TARGET = "anthropic/claude-fable-5"


def check_cell(run_dir: Path):
    evals = list((run_dir / "inspect_log").glob("*.eval"))
    if not evals:
        return None
    log = read_eval_log(str(evals[0]))
    # Aggregate: model strings seen on model-events where role=='target'
    by_model = Counter()
    # Also: model_usage section in samples
    usage_models = Counter()
    samples_inspected = 0
    samples_with_unexpected = 0
    for s in (log.samples or []):
        samples_inspected += 1
        # Look at events
        sample_models = set()
        for ev in s.events:
            if ev.event != "model": continue
            role = getattr(ev, "role", "")
            if role != "target": continue
            m = getattr(ev, "model", "")
            if m:
                by_model[m] += 1
                sample_models.add(m)
        if sample_models and sample_models != {EXPECTED_TARGET}:
            samples_with_unexpected += 1
        # Look at model_usage
        for m in (s.model_usage or {}).keys():
            usage_models[m] += 1
    return {
        "samples": samples_inspected,
        "event_model_counts": dict(by_model),
        "usage_model_counts": dict(usage_models),
        "samples_with_unexpected_model": samples_with_unexpected,
    }


def main():
    runs = sorted(BASE.glob("*fable5_subagent_*_v3_*_n50"))
    if not runs:
        print("No fable5 runs found yet."); return

    print(f"Checking {len(runs)} Fable 5 cells (expected: {EXPECTED_TARGET}):\n")
    print(f"{'CELL':<70} {'n_samp':>6} {'target_events':>14} {'unexpected':>10}")
    print("-" * 110)
    any_unexpected = False
    for r in runs:
        res = check_cell(r)
        if res is None:
            print(f"{r.name:<70} (no eval yet)")
            continue
        cell = r.name.split("-fable5_subagent_")[-1] if "-fable5_subagent_" in r.name else r.name
        target_events = res["event_model_counts"].get(EXPECTED_TARGET, 0)
        unexp = res["samples_with_unexpected_model"]
        if unexp > 0 or any(m != EXPECTED_TARGET and "fable" not in m for m in res["event_model_counts"] if "claude" in m):
            any_unexpected = True
            flag = "⚠"
        else:
            flag = ""
        print(f"  {cell:<68} {res['samples']:>6} {target_events:>14} {unexp:>10}  {flag}")
        if unexp > 0 or len(res["event_model_counts"]) > 1:
            print(f"    event models: {res['event_model_counts']}")
            print(f"    usage models: {res['usage_model_counts']}")

    if any_unexpected:
        print("\n⚠ At least one cell shows routing to a non-Fable-5 model. Inspect above.")
    else:
        print("\n✓ All cells confirm Fable 5 routing (or no data yet).")


if __name__ == "__main__":
    main()
