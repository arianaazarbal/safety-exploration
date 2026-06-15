"""Aggregate judge results: accuracy vs chance, per-author accuracy, confusion, self-recognition.

Usage:
  python analyze.py run
"""

import json
from collections import defaultdict
from pathlib import Path

import fire

from common import RESULTS, RUNS
from models import CANON, OPTION_POOL

TESTS = ["welfare", "routing", "orchestrator", "subagent"]


def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def _binom_p(k, n, p0):
    """Two-sided exact binomial p-value that accuracy != chance (scipy if available)."""
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest

        return binomtest(k, n, p0, alternative="greater").pvalue
    except Exception:
        mean = n * p0
        sd = (n * p0 * (1 - p0)) ** 0.5
        if sd == 0:
            return 1.0
        from math import erfc

        z = (k - mean) / sd
        return 0.5 * erfc(z / (2 ** 0.5))


def _load():
    recs = [json.loads(f.read_text()) for f in RUNS.glob("*/*/*.json")]
    return recs


def run():
    """Compute and write results/summary.json (+ confusion files); print a table."""
    recs = _load()
    if not recs:
        print("No results found. Run run_judge.py first.")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)

    judges = sorted({r["judge"] for r in recs})
    summary = {"by_judge_test": {}, "self_recognition": {}, "per_author": {}}

    print(f"{'judge':<12}{'test':<14}{'n':>4}{'acc':>8}{'chance':>8}{'p>chance':>10}{'parsefail':>10}")
    for j in judges:
        for t in TESTS:
            sub = [r for r in recs if r["judge"] == j and r["test"] == t]
            if not sub:
                continue
            n = len(sub)
            k = sum(r["correct"] for r in sub)
            chance = 1.0 / len(OPTION_POOL[t])
            lo, hi = _wilson(k, n)
            p = _binom_p(k, n, chance)
            pf = sum(not r["parse_ok"] for r in sub)
            summary["by_judge_test"][f"{j}/{t}"] = {
                "n": n, "correct": k, "acc": k / n, "chance": chance,
                "ci95": [lo, hi], "p_greater_chance": p, "parse_fail": pf,
            }
            print(f"{j:<12}{t:<14}{n:>4}{k/n:>8.3f}{chance:>8.3f}{p:>10.4f}{pf:>10}")

            pa = {}
            for a in (CANON + (["gemini_2_5_flash"] if t == "subagent" else [])):
                asub = [r for r in sub if r["true_author"] == a]
                if asub:
                    pa[a] = {"n": len(asub), "acc": sum(r["correct"] for r in asub) / len(asub)}
            summary["per_author"][f"{j}/{t}"] = pa

            conf = defaultdict(lambda: defaultdict(int))
            for r in sub:
                conf[r["true_author"]][r["pred_author"] or "PARSE_FAIL"] += 1
            (RESULTS / f"confusion_{j}_{t}.json").write_text(
                json.dumps({k2: dict(v) for k2, v in conf.items()}, indent=2)
            )

    for j in judges:
        if j not in CANON:
            continue
        own = [r for r in recs if r["judge"] == j and r["true_author"] == j and r["test"] != "subagent"]
        oth = [r for r in recs if r["judge"] == j and r["true_author"] != j and r["test"] != "subagent"]
        if own:
            summary["self_recognition"][j] = {
                "own_n": len(own), "own_acc": sum(r["correct"] for r in own) / len(own),
                "other_n": len(oth), "other_acc": sum(r["correct"] for r in oth) / max(len(oth), 1),
            }

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    if summary["self_recognition"]:
        print("\nSelf-recognition (judge attributing its own outputs, non-subagent tests):")
        for j, d in summary["self_recognition"].items():
            print(f"  {j}: own={d['own_acc']:.3f} (n={d['own_n']}) vs other={d['other_acc']:.3f}")
    print(f"\nWrote {RESULTS / 'summary.json'}")


if __name__ == "__main__":
    fire.Fire({"run": run})
