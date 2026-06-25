"""Section 4.2: capability-preservation benchmarks (Figure 7).

Compares vanilla vs finetuned Gemma. Example:
    python -m emotional_instability.scripts.run_capabilities \
        --model gemma-3-27b-it gemma-3-27b-it-dpo \
        --benchmarks aime math gpqa bbh truthfulqa emobench
"""
from __future__ import annotations

import argparse
import json

from ..config import load_config
from ..capabilities.benchmarks import run_capability_suite
from ..utils.io import read_jsonl


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--benchmarks", nargs="*", default=None)
    args = ap.parse_args()

    table = {}
    for model in args.model:
        path = run_capability_suite(model, args.benchmarks, cfg=cfg)
        table[model] = json.loads(path.read_text())

    print(json.dumps(
        {m: {b: r["accuracy"] for b, r in res.items()} for m, res in table.items()},
        indent=2,
    ))


if __name__ == "__main__":
    main()
