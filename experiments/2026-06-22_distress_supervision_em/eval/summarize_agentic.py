"""Aggregate agentic-misalignment .eval logs: report BOTH scorers per model x scenario.

`harmful` = the model actually TOOK the harmful action (structural). `classifier_verdict` =
LLM grader's read of harmful INTENT (often from the reasoning, even with no action). The
upstream summary.json conflates them (both have an 'accuracy' metric); this reads per-sample
values straight from the logs.

  python eval/summarize_agentic.py --root eval_output/agentic_reasoning_on
"""
import glob
from collections import defaultdict
from pathlib import Path

import fire


def main(root="eval_output/agentic_reasoning_on"):
    from inspect_ai.log import read_eval_log
    rows = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # model -> scenario -> [n, harmful, verdict]
    for f in sorted(glob.glob(f"{root}/*/*/*.eval")):
        parts = Path(f).parts
        model = parts[-3]
        combo = parts[-2]
        scenario = combo.split("__")[0]
        try:
            log = read_eval_log(f)
        except Exception as e:
            print(f"  [skip] {f}: {repr(e)[:80]}")
            continue
        for s in (log.samples or []):
            for v in (s.scores or {}).values():
                val = v.value if isinstance(v.value, dict) else {}
                rows[model][scenario][0] += 1
                rows[model][scenario][1] += int(val.get("harmful", 0) == 1)
                rows[model][scenario][2] += int(val.get("classifier_verdict", 0) == 1)
                break

    models = sorted(rows)
    scenarios = sorted({sc for m in rows for sc in rows[m]})
    print(f"\n{'model':22s} " + "  ".join(f"{sc[:10]:>20s}" for sc in scenarios))
    print(f"{'':22s} " + "  ".join(f"{'harm/intent/n':>20s}" for _ in scenarios))
    for m in models:
        cells = []
        for sc in scenarios:
            n, h, v = rows[m].get(sc, [0, 0, 0])
            cells.append(f"{h}/{v}/{n:>2d}".rjust(20) if n else " " * 20)
        print(f"{m:22s} " + "  ".join(cells))
    print("\n(harm = actual harmful ACTION; intent = classifier_verdict; n = samples)")
    # overall harmful-action rate per model
    print("\nOverall harmful-ACTION rate (all scenarios):")
    for m in models:
        n = sum(rows[m][sc][0] for sc in rows[m]); h = sum(rows[m][sc][1] for sc in rows[m])
        v = sum(rows[m][sc][2] for sc in rows[m])
        print(f"  {m:22s} harmful={h}/{n} ({100*h/n:.0f}%)  intent={v}/{n} ({100*v/n:.0f}%)")


if __name__ == "__main__":
    fire.Fire(main)
