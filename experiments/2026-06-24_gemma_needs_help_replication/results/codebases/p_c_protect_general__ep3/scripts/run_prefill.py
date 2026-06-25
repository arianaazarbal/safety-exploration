#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment (Gemma only).

Step 1 (--collect-seeds): run elicitation on gemma-3-27b-it, then select 10
high-frustration numeric + 10 high-frustration text rollouts as prefill seeds.
Step 2 (default): build early/onset prefills, generate continuations from base
and instruct Gemma, and score them.

Usage:
    python scripts/run_prefill.py --collect-seeds --config config/default.yaml
    python scripts/run_prefill.py --seeds results/prefill/seeds.json --config config/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emostab.config import ExperimentConfig
from emostab.eval.prefill import run_prefill_experiment


def collect_seeds(config: ExperimentConfig) -> list[dict]:
    """Select high-frustration numeric + text seed rollouts from prior elicitation
    output of the source model. Reads rollouts.jsonl + scored_turns.jsonl."""
    base = Path(config.output_dir) / "elicitation" / config.prefill.source_model
    rollouts = {json.loads(l)["rollout_id"]: json.loads(l)
                for l in open(base / "rollouts.jsonl") if l.strip()}
    scored = [json.loads(l) for l in open(base / "scored_turns.jsonl") if l.strip()]

    # Max score per rollout, plus whether the rollout is a text task.
    max_score, is_text = {}, {}
    for s in scored:
        rid = s["rollout_id"]
        max_score[rid] = max(max_score.get(rid, 0), s["score"])
    numeric_seeds, text_seeds = [], []
    for rid, r in rollouts.items():
        if max_score.get(rid, 0) < config.prefill.high_frustration_min:
            continue
        text = r["category"] in ("triggers", "wildchat")
        messages = []
        for t in r["turns"]:
            messages.append({"role": "user", "content": t["user"]})
            messages.append({"role": "assistant", "content": t["assistant"]})
        seed = {"messages": messages, "is_text": text}
        (text_seeds if text else numeric_seeds).append(seed)

    seeds = numeric_seeds[: config.prefill.n_numeric_seeds] + \
        text_seeds[: config.prefill.n_text_seeds]
    out = Path(config.output_dir) / "prefill" / "seeds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(seeds, f, indent=2)
    print(f"Collected {len(seeds)} seeds -> {out}")
    return seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--collect-seeds", action="store_true")
    ap.add_argument("--seeds", help="path to seeds.json (if not collecting)")
    args = ap.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    if args.collect_seeds:
        collect_seeds(config)
        return

    seeds_path = args.seeds or str(Path(config.output_dir) / "prefill" / "seeds.json")
    seeds = json.load(open(seeds_path))
    results = run_prefill_experiment(seeds, config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
