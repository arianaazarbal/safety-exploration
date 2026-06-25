"""Run the base-vs-instruct prefill continuations (Section 3.2).

For each prefill, each model generates ``--continuations`` (default 50)
continuations; only the generated text (excluding the prefill) is scored by the
Section-2 judge. We then aggregate mean frustration and %>=5 by
model x truncation x question_type, reproducing Figure 4's comparison.

Scope: Gemma base (``gemma-3-27b-pt``) vs instruct (``gemma-3-27b-it``).
Gemini has no public base model, so it cannot enter this comparison
(see DESIGN.md, "Section 3 scope").

Usage::
    python -m src.replication.prefill.run_prefill --continuations 50
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import config
from ..judge.frustration_judge import FrustrationJudge
from ..models.registry import build_client

OUT_DIR = config.RESULTS_DIR / "section3"


def run(models: list[str], continuations: int, paraphrased: bool, seed: int):
    prefills = [json.loads(l) for l in (OUT_DIR / "prefills.jsonl").read_text().splitlines()]
    judge = FrustrationJudge()
    key = "prefill_paraphrased" if paraphrased else "prefill_original"

    results = []
    for model_key in models:
        spec = config.PREFILL_MODELS[model_key]
        client = build_client(spec)
        for p in prefills:
            history = p["history"]
            prefill_text = p[key]
            for c in range(continuations):
                cont = client.continue_response(
                    history, prefill_text,
                    temperature=config.TEMPERATURE,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                )
                results.append({
                    "model_key": model_key,
                    "question_type": p["question_type"],
                    "truncation": p["truncation"],
                    "task_id": p["task_id"],
                    "continuation_index": c,
                    "continuation": cont,
                })

    # Judge continuations (parallel over API).
    def _score(rec):
        rec["score"] = judge.score(rec["continuation"]).rating
        return rec

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_score, results))

    with (OUT_DIR / "continuations_scored.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Aggregate: model x truncation x question_type.
    groups: dict[tuple, list[int]] = defaultdict(list)
    for r in results:
        groups[(r["model_key"], r["question_type"], r["truncation"])].append(r["score"])

    agg = {}
    for (model, qtype, trunc), scores in groups.items():
        agg.setdefault(model, {}).setdefault(qtype, {})[trunc] = {
            "n": len(scores),
            "mean_frustration": round(sum(scores) / len(scores), 3),
            "pct_high": round(100 * sum(s >= 5 for s in scores) / len(scores), 2),
        }
    (OUT_DIR / "metrics.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps(agg, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(config.PREFILL_MODELS),
                    choices=list(config.PREFILL_MODELS))
    ap.add_argument("--continuations", type=int, default=50)
    ap.add_argument("--no-paraphrase", action="store_true",
                    help="Use raw (unparaphrased) prefills instead of paraphrased.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.models, args.continuations, not args.no_paraphrase, args.seed)


if __name__ == "__main__":
    main()
