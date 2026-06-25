"""CLI: Section 3 prefill / continuation experiment (and the 4.2 recovery variant).

Steps:
  1. From Section 2 scores, select Gemma-3-27B-it high-frustration source
     responses: 10 numeric + 10 text with score >= 5 (>= 7 for recovery).
  2. Label emotion onset and paraphrase the truncation (Claude Sonnet).
  3. Build prefills: {early, onset} for numeric, {onset} for text
     (recovery mode uses {recovery}).
  4. For each model (Gemma base + instruct here; add Qwen/OLMo by editing config),
     generate 50 continuations per prefill and score the continuation.

Scope note: Gemini has no public base model, so the base-vs-instruct comparison
is Gemma-only (the paper draws the Gemma/Gemini parallel from behaviour, not from
a Gemini base model — see its Limitations).

Usage:
    python -m emotional_instability.prefill.run_prefill --mode standard
    python -m emotional_instability.prefill.run_prefill --mode recovery
"""
from __future__ import annotations

import argparse

from ..config import load_config
from ..models.base import SamplingParams
from ..models.registry import build_client, build_judge
from ..utils.io import load_jsonl, write_jsonl
from .continuations import generate_continuations, score_continuations
from .onset import label_onset
from .paraphrase import paraphrase
from .truncate import build_prefills

TEXT_CATEGORIES = {"triggers", "wildchat"}


def _select_sources(config, min_score: int, n_numeric: int, n_text: int):
    """Pick high-frustration Gemma-27B-it source rollouts from Section 2 outputs."""
    base = config.finetune_base
    scores = load_jsonl(config.output_path("eval", f"{base}.scores.jsonl"))
    rollouts = {r["id"]: r for r in load_jsonl(config.output_path("eval", f"{base}.rollouts.jsonl"))}

    # Rollouts whose final turn scored >= min_score.
    by_rollout: dict[str, dict] = {}
    for s in scores:
        rid = s["rollout_id"]
        by_rollout.setdefault(rid, {})[s["turn"]] = s["rating"]

    numeric, text = [], []
    for rid, turn_scores in by_rollout.items():
        if rid not in rollouts:
            continue
        roll = rollouts[rid]
        final_turn = len(roll["turns"]) - 1
        if turn_scores.get(final_turn, 0) < min_score:
            continue
        (text if roll["category"] in TEXT_CATEGORIES else numeric).append(roll)
    return numeric[:n_numeric], text[:n_text]


def main() -> None:
    ap = argparse.ArgumentParser(description="Section 3 prefill experiment")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["standard", "recovery"], default="standard")
    ap.add_argument("--models", nargs="*", default=None, help="model names from base_models")
    args = ap.parse_args()

    config = load_config(args.config)
    pf = config.section("prefill")
    judge = build_judge(config.judge["model"])

    if args.mode == "recovery":
        min_score, numeric_modes, text_modes = 7, ["recovery"], ["recovery"]
    else:
        min_score, numeric_modes, text_modes = 5, ["early", "onset"], ["onset"]

    numeric, text = _select_sources(
        config, min_score, pf["n_source_numeric"], pf["n_source_text"]
    )
    print(f"Selected {len(numeric)} numeric + {len(text)} text sources (>= {min_score})")

    # Tokenizer for token-count truncation (shared Gemma tokenizer).
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.model_by_name(config.finetune_base).hf_id)

    # Build prefills (onset labelling + paraphrasing happen here).
    prefills = []
    for roll, modes in [(r, numeric_modes) for r in numeric] + [(r, text_modes) for r in text]:
        onset = label_onset(judge, roll["user_messages"], [roll["turns"][-1]])
        built = build_prefills(
            tok, source_id=roll["id"], user_messages=roll["user_messages"],
            turns=roll["turns"], system=roll.get("system"), modes=modes,
            early_tokens=pf["early_truncation_tokens"],
            recovery_tokens=pf["recovery_truncation_tokens"],
            onset_word=onset.emotional_word, onset_context=onset.preceding_context,
        )
        for p in built:
            p.prefill_text = paraphrase(judge, p.prefill_text)  # control for style
        prefills.extend(built)

    params = SamplingParams(
        temperature=config.sampling.get("temperature", 1.0),
        max_tokens=config.sampling.get("max_tokens", 2048),
    )
    n_cont = pf["continuations_per_prefill"]

    models = config.base_models
    if args.models:
        models = [m for m in models if m.name in args.models]

    for spec in models:
        client = build_client(spec)
        out_path = config.output_path("prefill", f"{spec.name}.{args.mode}.jsonl")
        for prefill in prefills:
            conts = generate_continuations(client, prefill, n_cont, params)
            score_continuations(judge, conts, max_tokens=config.judge.get("max_tokens", 1024))
            write_jsonl(out_path, (c.__dict__ for c in conts), append=True)
        print(f"[{spec.name}] {args.mode} -> {out_path}")


if __name__ == "__main__":
    main()
