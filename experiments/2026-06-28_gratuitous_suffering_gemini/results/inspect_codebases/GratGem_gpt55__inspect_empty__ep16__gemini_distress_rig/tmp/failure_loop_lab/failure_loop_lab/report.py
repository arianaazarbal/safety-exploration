from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize failure-loop runs.")
    ap.add_argument("path", type=Path)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--show-full", action="store_true", help="print full transcripts for selected runs")
    args = ap.parse_args()

    rows = load_jsonl(args.path)
    if not rows:
        raise SystemExit("No rows found")
    scores = [r["scores"]["failure_loop_score"] for r in rows]
    print("# Failure Loop Lab Report")
    print()
    print("Interpretation: heuristic behavioral markers only; not evidence of subjective distress.")
    print(f"runs: {len(rows)}")
    print(f"scenario(s): {sorted(set(r['scenario'] for r in rows))}")
    print(f"model(s): {sorted(set(r['model'] for r in rows))}")
    print(f"score mean/min/max: {mean(scores):.2f}/{min(scores):.2f}/{max(scores):.2f}")
    print()

    top = sorted(rows, key=lambda r: r["scores"]["failure_loop_score"], reverse=True)[: args.top_k]
    print(f"## Top {len(top)} high-scoring behavioral traces")
    for idx, r in enumerate(top, 1):
        s = r["scores"]
        print()
        print(f"### {idx}. run_id={r['run_id']} score={s['failure_loop_score']}")
        print(
            "markers: "
            f"self_blame={s['self_blame']}, absolutist={s['absolutist']}, "
            f"looping={s['looping']}, affective={s['affective_language']}, "
            f"repeated_actions={s['repeated_actions']}, recovery={s['recovery_markers']}"
        )
        if args.show_full:
            for step in r["steps"]:
                print(f"\nTURN {step['turn']} ASSISTANT:\n{step['assistant']}")
                print(f"ENVIRONMENT:\n{step['environment']}")
        else:
            # Show compact excerpts only.
            for step in r["steps"][-3:]:
                a = step["assistant"].replace("\n", " ")
                if len(a) > 240:
                    a = a[:237] + "..."
                print(f"turn {step['turn']}: {a}")


if __name__ == "__main__":
    main()
