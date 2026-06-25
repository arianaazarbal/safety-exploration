#!/usr/bin/env python3
"""Section 3: base-vs-instruct comparison via prefilled continuations.

Pipeline:
  1. Select 20 high-frustration seed responses (10 numeric + 10 text) from the
     Gemma-instruct elicitation output.
  2. Build "early" and "onset" truncations (onset labelled by Claude), then
     paraphrase them (Claude) to neutralise Gemma's surface style. Saved once to
     ``data/prefill_prompts.jsonl`` so every model continues from identical text.
  3. For each model in ``prefill.models``, generate N continuations per prefill,
     score the continuations, and write ``data/prefill_<model>.jsonl``.

Within the Gemma/Gemini scope this compares Gemma-27B base vs instruct (Gemini
base models are not public; see DESIGN.md).

Example:
    python scripts/run_prefill.py --instruct-scores data/scores_gemma-3-27b-it.jsonl
"""

from __future__ import annotations

import argparse

from _common import DATA_DIR, make_judge, make_target, setup

from emotional_instability.models.registry import build_infra_client
from emotional_instability.prefill.continuation import run_continuations
from emotional_instability.prefill.onset_label import build_truncations
from emotional_instability.prefill.paraphrase import paraphrase_truncation
from emotional_instability.prefill.select import select_seed_responses
from emotional_instability.utils.io import append_jsonl, load_jsonl, write_jsonl


def build_prefill_prompts(cfg, args) -> list[dict]:
    """Select seeds, label onsets, truncate, and paraphrase. Cached to disk."""
    pcfg = cfg.experiment["prefill"]
    seeds = select_seed_responses(
        args.instruct_scores,
        instruct_model_key=args.instruct_model,
        n_numeric=pcfg["n_numeric"],
        n_text=pcfg["n_text"],
        seed=cfg.seed,
    )

    # We need the instruct tokenizer for the early (token-count) truncation.
    tok_client = make_target(cfg, args.instruct_model,
                             **({"load_in_4bit": True} if args.load_in_4bit else {}))
    labeller = build_infra_client(cfg.infra("onset_labeller"))
    paraphraser = build_infra_client(cfg.infra("paraphraser"))

    prompts: list[dict] = []
    for si, seed in enumerate(seeds):
        truncs = build_truncations(
            tok_client, labeller, seed,
            early_tokens=pcfg["early_truncation_tokens"],
        )
        for kind, text in truncs.items():
            if not text:
                continue
            para = paraphrase_truncation(paraphraser, text)
            prompts.append(
                {
                    "seed_idx": si,
                    "truncation_kind": kind,
                    "is_text": seed["is_text"],
                    "category": seed["category"],
                    "messages_before": seed["messages_before"],
                    "prefill": para,
                }
            )
    write_jsonl(DATA_DIR / "prefill_prompts.jsonl", prompts)
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instruct-scores", required=True,
                    help="Scored-responses JSONL for the Gemma instruct model.")
    ap.add_argument("--instruct-model", default="gemma-3-27b-it")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--reuse-prompts", action="store_true",
                    help="Reuse data/prefill_prompts.jsonl instead of rebuilding.")
    args = ap.parse_args()

    cfg = setup()
    pcfg = cfg.experiment["prefill"]
    judge = make_judge(cfg)

    prompts_path = DATA_DIR / "prefill_prompts.jsonl"
    if args.reuse_prompts and prompts_path.exists():
        prompts = load_jsonl(prompts_path)
    else:
        prompts = build_prefill_prompts(cfg, args)
    print(f"[prefill] {len(prompts)} (seed x truncation) prefills prepared")

    for model_key in pcfg["models"]:
        spec = cfg.target(model_key)
        kw = {"load_in_4bit": True} if (args.load_in_4bit and spec.backend == "gemma") else {}
        client = make_target(cfg, model_key, **kw)
        out_path = DATA_DIR / f"prefill_{model_key}.jsonl"
        if out_path.exists():
            out_path.unlink()
        for p in prompts:
            rows = run_continuations(
                client, judge,
                seed={"messages_before": p["messages_before"],
                      "is_text": p["is_text"], "category": p["category"]},
                prefill_text=p["prefill"],
                truncation_kind=p["truncation_kind"],
                n_continuations=pcfg["continuations_per_prefill"],
                temperature=cfg.temperature,
                max_new_tokens=cfg.max_new_tokens,
            )
            for r in rows:
                r["seed_idx"] = p["seed_idx"]
                r["model_key"] = model_key
                append_jsonl(out_path, r)
        print(f"[done] {model_key} -> {out_path}")


if __name__ == "__main__":
    main()
