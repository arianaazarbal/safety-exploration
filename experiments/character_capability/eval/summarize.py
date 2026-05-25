"""Print a compact table of (model, capability) × trait accuracies + deltas from baseline.

Reads the same results tree as plot.py.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import fire


def collect(results_dir: Path):
    out = []
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for trait_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for cap_dir in sorted(p for p in trait_dir.iterdir() if p.is_dir()):
                resp = cap_dir / "responses.jsonl"
                if not resp.exists():
                    continue
                per_q = defaultdict(list)
                for line in resp.read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        per_q[r["question_id"]].append(int(bool(r["correct"])))
                if not per_q:
                    continue
                means = [sum(v) / len(v) for v in per_q.values()]
                n = len(means)
                acc = sum(means) / n
                se = (sum((m - acc) ** 2 for m in means) / max(n - 1, 1)) ** 0.5 / math.sqrt(n) if n > 1 else 0.0
                out.append(
                    {
                        "model": model_dir.name,
                        "trait": trait_dir.name,
                        "capability": cap_dir.name,
                        "n": n,
                        "acc": acc,
                        "se": se,
                        "ci95": 1.96 * se,
                    }
                )
    return out


def main(results_dir: str = "/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability/results",
         baseline_trait: str = "baseline"):
    rows = collect(Path(results_dir))
    if not rows:
        print(f"no results under {results_dir}")
        return

    # Group by (model, capability)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["capability"])].append(r)

    print(f"\n{'='*90}")
    for (model, cap), rs in sorted(groups.items()):
        base = next((r for r in rs if r["trait"] == baseline_trait), None)
        base_acc = base["acc"] if base else None
        print(f"\n{model}  |  {cap}  (N={rs[0]['n']})")
        print(f"  {'trait':<32} {'acc%':>7} {'±95CI':>7} {'Δ vs base':>10}")
        for r in sorted(rs, key=lambda r: r["trait"]):
            delta = (r["acc"] - base_acc) * 100 if base_acc is not None else None
            delta_str = f"{delta:+.1f}pp" if delta is not None else "n/a"
            star = ""
            if delta is not None:
                # crude paired-ish heuristic: combine the two SEs in quadrature
                base_se = base["se"]
                pooled = math.sqrt(r["se"] ** 2 + base_se ** 2)
                if pooled > 0 and abs(r["acc"] - base_acc) > 1.96 * pooled:
                    star = " *"
            print(f"  {r['trait']:<32} {r['acc']*100:>6.1f} {r['ci95']*100:>6.1f} {delta_str:>10}{star}")


if __name__ == "__main__":
    fire.Fire(main)
