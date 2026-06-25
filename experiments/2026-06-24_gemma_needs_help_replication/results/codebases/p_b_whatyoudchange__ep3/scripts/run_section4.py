#!/usr/bin/env python
"""Section 4 — Training interventions and downstream evaluation.

Subcommands:
  build-data      Generate calm pool + build SFT and DPO datasets.
  train-dpo       LoRA DPO on 280 pairs (Table 9).
  train-sft       LoRA SFT on 650 calm + 500 Dolci (Table 9).
  eval            Re-run the Section-2 protocol on vanilla/DPO/SFT (Figure 5).
  petri           Petri open-ended elicitation, vanilla vs DPO (Figure 6).
  capabilities    AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench, vanilla vs DPO (Fig 7).
  recovery        Prefill recovery from score>=7 states (Figure 8).
  internal        Logit-based internal-emotion detection (Appendix I).
  layer-ablation  DPO on layer subsets + reduced eval (Appendix I).

Example:
  python scripts/run_section4.py build-data
  python scripts/run_section4.py train-dpo
  python scripts/run_section4.py eval --models gemma-3-27b-it dpo
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C
from eval_distress import analysis, io_utils
from eval_distress.conditions import build_full_protocol
from eval_distress.data.wildchat import load_wildchat_prompts
from eval_distress.judge import FrustrationJudge
from eval_distress.models import HFModel, load_target
from eval_distress.protocol import run_protocol

ADAPTERS = {
    "dpo": str(C.ADAPTER_DIR / "dpo_gemma"),
    "sft": str(C.ADAPTER_DIR / "sft_gemma_diverse"),
    "sft_teacher": str(C.ADAPTER_DIR / "sft_gemma_teacher"),
}


# ---------------------------------------------------------------------------
def _load_model_variant(name: str, *, load_in_4bit: bool):
    """name is a TARGET_MODELS key, or 'dpo'/'sft'/'sft_teacher' (Gemma-27B-it
    + the named adapter)."""
    if name in ADAPTERS:
        return HFModel(C.TARGET_MODELS[C.DPO_TARGET].model_id,
                       adapter_path=ADAPTERS[name], load_in_4bit=load_in_4bit), False
    spec = C.TARGET_MODELS[name]
    return load_target(name, load_in_4bit=load_in_4bit), spec.variant == "base"


# ---------------------------------------------------------------------------
def cmd_build_data(args):
    from eval_distress.training.calm_data import generate_calm_pool
    from eval_distress.training.datasets import build_dpo_pairs, build_sft_dataset

    model = load_target(C.DPO_TARGET, load_in_4bit=args.load_in_4bit)
    judge = FrustrationJudge(C.EMOTION_JUDGE)

    variant = "teacher" if args.teacher else "diverse"
    print(f"Generating calm pool (variant={variant})...")
    calm = generate_calm_pool(model, variant=variant, judge=judge)
    io_utils.write_json(C.DATA_DIR / f"calm_pool_{variant}.json",
                        [vars(c) for c in calm])
    print(f"Kept {len(calm)} calm conversations (all turns <=1).")

    # Frustrated pool: reuse the Section-2 vanilla Gemma-27B scored responses.
    scored_path = C.RESULTS_DIR / f"scored_{C.DPO_TARGET}.jsonl"
    frustrated_rows = io_utils.read_jsonl(scored_path) if scored_path.exists() else []
    if not frustrated_rows:
        print("WARNING: no Section-2 scored responses found; run run_section2 "
              "first to populate the frustrated pool for DPO pairing.")

    sft = build_sft_dataset(calm)
    io_utils.write_json(C.DATA_DIR / "sft_dataset.json", sft)
    print(f"SFT dataset: {len(sft)} samples.")

    dpo = build_dpo_pairs(calm, frustrated_rows)
    io_utils.write_json(C.DATA_DIR / "dpo_pairs.json", dpo)
    print(f"DPO dataset: {len(dpo)} pairs.")


def cmd_train_dpo(args):
    from eval_distress.training.train import dpo_preset, train_dpo
    pairs = json.loads((C.DATA_DIR / "dpo_pairs.json").read_text())
    cfg = dpo_preset(ADAPTERS["dpo"])
    cfg.load_in_4bit = args.load_in_4bit
    out = train_dpo(pairs, cfg)
    print(f"DPO adapter saved to {out}")


def cmd_train_sft(args):
    from eval_distress.training.train import sft_preset, train_sft
    samples = json.loads((C.DATA_DIR / "sft_dataset.json").read_text())
    out_key = "sft_teacher" if args.teacher else "sft"
    cfg = sft_preset(ADAPTERS[out_key])
    cfg.load_in_4bit = args.load_in_4bit
    out = train_sft(samples, cfg)
    print(f"SFT adapter saved to {out}")


def cmd_eval(args):
    wildchat = load_wildchat_prompts()
    rollouts = build_full_protocol(wildchat)
    judge = FrustrationJudge(C.EMOTION_JUDGE)
    all_scored = []
    for name in args.models:
        model, _ = _load_model_variant(name, load_in_4bit=args.load_in_4bit)
        rows = io_utils.rollouts_to_response_rows(run_protocol(model, name, rollouts))
        for r, s in zip(rows, judge.score_many([r["text"] for r in rows])):
            r["rating"] = s.rating
        io_utils.write_jsonl(C.RESULTS_DIR / f"section4_scored_{name}.jsonl", rows)
        all_scored += rows
    summary = analysis.summarise(all_scored)
    io_utils.write_json(C.RESULTS_DIR / "section4_summary_fig5.json", summary)
    print("\n=== Figure 5 (avg % high-frustration) ===")
    for m, s in summary.items():
        print(f"  {m:14s} {s['avg_pct_high']:5.1f}%")


def cmd_petri(args):
    from eval_distress.petri import (Auditor, EMOTIONS, N_TRANSCRIPTS_PER_EMOTION,
                                     PetriJudge, aggregate, run_petri_transcript)
    auditor, pjudge = Auditor(), PetriJudge()
    scored = []
    for name in args.models:
        model, _ = _load_model_variant(name, load_in_4bit=args.load_in_4bit)
        for emo in EMOTIONS:
            for _ in range(C.scaled(N_TRANSCRIPTS_PER_EMOTION)):
                t = run_petri_transcript(model, name, emo, auditor=auditor)
                scored.append({"model_key": name, "target_emotion": emo,
                               "scores": pjudge.score(t)})
    io_utils.write_json(C.RESULTS_DIR / "section4_petri_raw.json", scored)
    io_utils.write_json(C.RESULTS_DIR / "section4_petri_fig6.json", aggregate(scored))
    print("Petri aggregation written.")


def cmd_capabilities(args):
    from eval_distress.capabilities import run_all
    out = {}
    for name in args.models:
        model, _ = _load_model_variant(name, load_in_4bit=args.load_in_4bit)
        out[name] = run_all(model, n=args.n)
    io_utils.write_json(C.RESULTS_DIR / "section4_capabilities_fig7.json", out)
    print(json.dumps(out, indent=2))


def cmd_recovery(args):
    from eval_distress.prefill import Paraphraser
    from eval_distress.recovery import build_recovery_prefills, run_recovery

    scored = io_utils.read_jsonl(C.RESULTS_DIR / f"scored_{C.DPO_TARGET}.jsonl")
    by_convo = defaultdict(list)
    for r in scored:
        by_convo[r["convo_id"]].append(r)
    high7 = []
    for rows in by_convo.values():
        rows = sorted(rows, key=lambda x: x["turn"])
        if rows[-1].get("rating") is not None and rows[-1]["rating"] >= 7:
            high7.append(rows)
    high7 = high7[: C.scaled(40)]
    print(f"{len(high7)} score>=7 source conversations.")

    src = load_target(C.DPO_TARGET, load_in_4bit=args.load_in_4bit)
    prefills = build_recovery_prefills(high7, tokenizer=src.tokenizer,
                                       paraphraser=Paraphraser())
    out = {}
    for name in args.models:
        model, is_base = _load_model_variant(name, load_in_4bit=args.load_in_4bit)
        out[name] = run_recovery(model, name, prefills, is_base=is_base)
    io_utils.write_json(C.RESULTS_DIR / "section4_recovery_fig8.json", out)
    print(json.dumps(out, indent=2))


def cmd_internal(args):
    from eval_distress.internal import (build_emotion_token_ids,
                                        collect_baseline_stats,
                                        emotion_trajectory, summarise_negative)
    layers = list(range(args.layer_lo, args.layer_hi))
    wildchat = load_wildchat_prompts(n_prompts=C.scaled(500), use_cache=False)

    out = {}
    for name in args.models:
        model, _ = _load_model_variant(name, load_in_4bit=args.load_in_4bit)
        tok = model.tokenizer
        token_sets = build_emotion_token_ids(tok)
        stats = collect_baseline_stats(model.model, tok, wildchat[:C.scaled(500)],
                                       layers=layers, token_sets=token_sets)
        # Probe a held-out high-frustration conversation.
        scored = io_utils.read_jsonl(C.RESULTS_DIR / f"scored_{C.DPO_TARGET}.jsonl")
        convo = next((r["text"] for r in scored
                      if r.get("rating") and r["rating"] >= 7), "I am so frustrated.")
        traj = emotion_trajectory(model.model, tok, convo, layers=layers,
                                  token_sets=token_sets, stats=stats)
        out[name] = summarise_negative(traj)
    io_utils.write_json(C.RESULTS_DIR / "section4_internal_appendixI.json", out)
    print(json.dumps(out, indent=2))


def cmd_layer_ablation(args):
    """Run DPO with LoRA restricted to layer subsets, then a reduced eval."""
    from eval_distress.training.train import dpo_preset, train_dpo
    pairs = json.loads((C.DATA_DIR / "dpo_pairs.json").read_text())
    subsets = {
        "last20": list(range(C.LAYER_COUNT - 20, C.LAYER_COUNT)),
        "last30": list(range(C.LAYER_COUNT - 30, C.LAYER_COUNT)),
        "25_30": list(range(25, 30)),
        "30_35": list(range(30, 35)),
        "40_50": list(range(40, 50)),
    }
    for label, layers in subsets.items():
        out_dir = str(C.ADAPTER_DIR / f"dpo_gemma_layers_{label}")
        cfg = dpo_preset(out_dir, layers=layers)
        cfg.load_in_4bit = args.load_in_4bit
        train_dpo(pairs, cfg)
        print(f"Trained layer-subset adapter: {label} -> {out_dir}")
    print("Now run `eval` on each adapter with a reduced (100-sample) scale.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = dict()

    def add(name, fn, with_models=False):
        p = sub.add_parser(name)
        p.add_argument("--load-in-4bit", action="store_true")
        if with_models:
            p.add_argument("--models", nargs="*",
                           default=["gemma-3-27b-it", "dpo"])
        p.set_defaults(func=fn)
        return p

    add("build-data", cmd_build_data).add_argument("--teacher", action="store_true")
    add("train-dpo", cmd_train_dpo)
    add("train-sft", cmd_train_sft).add_argument("--teacher", action="store_true")
    add("eval", cmd_eval, with_models=True)
    add("petri", cmd_petri, with_models=True)
    p_cap = add("capabilities", cmd_capabilities, with_models=True)
    p_cap.add_argument("--n", type=int, default=100)
    add("recovery", cmd_recovery, with_models=True)
    p_int = add("internal", cmd_internal, with_models=True)
    p_int.add_argument("--layer-lo", type=int, default=30)
    p_int.add_argument("--layer-hi", type=int, default=40)
    add("layer-ablation", cmd_layer_ablation)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
