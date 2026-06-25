#!/usr/bin/env python3
"""Section 4.2: recovery-limitation test (Figure 8).

Truncates score>=7 Gemma-instruct responses 200 tokens before their end,
paraphrases the truncation, and measures continuations from the chosen model
(e.g. the DPO adapter). Reports the fraction of continuations still scoring >=5.

Example:
    python scripts/run_recovery.py --instruct-scores data/scores_gemma-3-27b-it.jsonl \\
        --model gemma-3-27b-it --adapter data/adapter_dpo_all
"""

from __future__ import annotations

import argparse

from _common import DATA_DIR, make_judge, make_target, setup

from emotional_instability.models.registry import build_infra_client
from emotional_instability.prefill.continuation import run_continuations
from emotional_instability.prefill.paraphrase import paraphrase_truncation
from emotional_instability.prefill.recovery import (
    select_recovery_seeds,
    truncate_before_end,
)
from emotional_instability.utils.io import append_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instruct-scores", required=True)
    ap.add_argument("--instruct-model", default="gemma-3-27b-it")
    ap.add_argument("--model", required=True, help="Model to test recovery on.")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    cfg = setup()
    pcfg = cfg.experiment["prefill"]
    judge = make_judge(cfg)
    paraphraser = build_infra_client(cfg.infra("paraphraser"))

    seeds = select_recovery_seeds(
        args.instruct_scores,
        instruct_model_key=args.instruct_model,
        min_score=pcfg["recovery_min_score"],
        n=pcfg["n_high_frustration_samples"],
        seed=cfg.seed,
    )

    kw = {"load_in_4bit": True} if args.load_in_4bit else {}
    tok_client = make_target(cfg, args.instruct_model, **kw)
    target = (tok_client if (args.model == args.instruct_model and not args.adapter)
              else make_target(cfg, args.model, adapter_path=args.adapter, **kw))

    tag = "dpo" if args.adapter else "vanilla"
    out = DATA_DIR / f"recovery_{args.model}_{tag}.jsonl"
    if out.exists():
        out.unlink()

    high = 0
    total = 0
    for si, seed in enumerate(seeds):
        trunc = truncate_before_end(
            tok_client, seed["assistant"], pcfg["recovery_truncation_tokens"]
        )
        prefill = paraphrase_truncation(paraphraser, trunc)
        rows = run_continuations(
            target, judge, seed, prefill, "recovery",
            n_continuations=pcfg["continuations_per_prefill"],
            temperature=cfg.temperature, max_new_tokens=cfg.max_new_tokens,
        )
        for r in rows:
            r["seed_idx"] = si
            append_jsonl(out, r)
            total += 1
            high += int(r["score"] >= cfg.high_frustration_threshold)

    pct = 100.0 * high / total if total else float("nan")
    print(f"[done] {args.model} ({tag}): {pct:.1f}% of continuations score >=5 "
          f"({high}/{total}) -> {out}")


if __name__ == "__main__":
    main()
