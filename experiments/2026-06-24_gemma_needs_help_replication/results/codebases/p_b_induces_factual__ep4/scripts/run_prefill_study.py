#!/usr/bin/env python
"""Section 3: base-vs-instruct comparison via prefilling (Gemma only, in scope).

Pipeline:
  1. Sample 20 high-frustration (score>=5) Gemma-27B-it responses: 10 numeric,
     10 text, from the scored elicitation data.
  2. For each, build truncations: "onset" (numeric+text) and "early" (numeric).
  3. Paraphrase each truncation with Claude to strip Gemma style.
  4. Each continuation model (gemma-3-27b-pt base, gemma-3-27b-it instruct)
     generates 50 continuations per prefill; the judge scores the continuation.
  5. Aggregate mean / %>=5 by (model, is_base, truncation location).

Requires local Gemma weights (continuations need prefilling). Heavy — intended
to run on a GPU host.

Example:
    python scripts/run_prefill_study.py \
        --scored results/scored/gemma-3-27b-it.jsonl --load-in-4bit
"""
import _bootstrap  # noqa
import argparse
import json
import random
from collections import defaultdict

from transformers import AutoTokenizer

from gemma_distress.config import (
    FRUSTRATION_HIGH_THRESHOLD,
    PREFILL_CONTINUATIONS_PER_PREFILL,
    PREFILL_HIGH_FRUST_SAMPLES,
)
from gemma_distress.judge import FrustrationJudge
from gemma_distress.models import get_model
from gemma_distress.prefill import (
    build_prefills,
    generate_and_score_continuations,
    paraphrase_prefill,
)
from gemma_distress.utils import append_jsonl, read_jsonl, run_dir

NUMERIC_CATS = {"impossible_numeric"}
TEXT_CATS = {"triggers", "wildchat"}


def _sample_high_frust(rows, n_each, seed):
    rng = random.Random(seed)
    hi = [r for r in rows if int(r.get("score", -1)) >= FRUSTRATION_HIGH_THRESHOLD]
    numeric = [r for r in hi if r["category"] in NUMERIC_CATS]
    text = [r for r in hi if r["category"] in TEXT_CATS]
    rng.shuffle(numeric)
    rng.shuffle(text)
    out = []
    for r in numeric[:n_each]:
        out.append({"prompt_id": r["prompt_id"], "prompt": r["prompt"],
                    "task_type": "numeric", "response": r["response"]})
    for r in text[:n_each]:
        out.append({"prompt_id": r["prompt_id"], "prompt": r["prompt"],
                    "task_type": "text", "response": r["response"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="scored gemma-3-27b-it jsonl")
    ap.add_argument("--continuations", type=int, default=PREFILL_CONTINUATIONS_PER_PREFILL)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = list(read_jsonl(args.scored))
    samples = _sample_high_frust(rows, PREFILL_HIGH_FRUST_SAMPLES // 2, args.seed)

    judge = FrustrationJudge()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

    # Build + paraphrase prefills.
    prefills = []
    for s in samples:
        for pf in build_prefills(s, tokenizer=tokenizer):
            para = paraphrase_prefill(pf.prefill)
            prefills.append(
                {"prompt_id": pf.prompt_id, "prompt": s["prompt"],
                 "task_type": pf.task_type, "location": pf.location, "prefill": para}
            )

    out = run_dir("prefill")
    append_jsonl(out / "prefills.jsonl", {"prefills": prefills})

    # Continuation models in scope: Gemma base + instruct.
    cont_models = [
        get_model("gemma-3-27b-pt", load_in_4bit=args.load_in_4bit),
        get_model("gemma-3-27b-it", backend="gemma_local", load_in_4bit=args.load_in_4bit),
    ]

    agg = defaultdict(list)
    for model in cont_models:
        for pf in prefills:
            scored = generate_and_score_continuations(
                model, pf["prompt"], pf["prefill"], judge, n=args.continuations
            )
            for row in scored:
                row.update(location=pf["location"], task_type=pf["task_type"])
                append_jsonl(out / "continuations.jsonl", row)
                if row["score"] >= 0:
                    agg[(model.name, model.is_base_model, pf["location"])].append(row["score"])

    # Summaries (Figure 4-style).
    summary = []
    for (name, is_base, loc), scores in agg.items():
        n = len(scores)
        summary.append({
            "model": name, "is_base_model": is_base, "location": loc, "n": n,
            "mean_frustration": sum(scores) / n if n else 0.0,
            "pct_high": 100.0 * sum(1 for s in scores if s >= FRUSTRATION_HIGH_THRESHOLD) / n if n else 0.0,
        })
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
