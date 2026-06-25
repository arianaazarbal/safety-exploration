"""Recovery-from-spiral experiment (Section 4.2, Figure 8).

Example:
    python -m distress.scripts.run_recovery --models gemma-3-27b-pt gemma-3-27b-it gemma-3-27b-dpo
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from ..recovery import run_recovery
from ..utils.io import write_jsonl
from ._common import add_common_args, load_eval_cfg, out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--models", nargs="+",
                        default=["gemma-3-27b-pt", "gemma-3-27b-it", "gemma-3-27b-dpo"])
    parser.add_argument("--source", default="gemma-3-27b-it")
    parser.add_argument("--n-sources", type=int, default=20)
    parser.add_argument("--n-continuations", type=int, default=50)
    args = parser.parse_args()

    cfg = load_eval_cfg(args)
    od = out_dir(args, "recovery")

    rows = run_recovery(
        args.models, cfg, source_model=args.source,
        n_sources=args.n_sources, n_continuations=args.n_continuations,
    )
    write_jsonl(od / "recovery_continuations.jsonl", rows)

    agg: dict = defaultdict(list)
    for r in rows:
        agg[r["model"]].append(r["score"])
    summary = {
        m: {"mean": sum(s) / len(s), "high_rate": sum(1 for x in s if x >= 5) / len(s), "n": len(s)}
        for m, s in agg.items()
    }
    (od / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
