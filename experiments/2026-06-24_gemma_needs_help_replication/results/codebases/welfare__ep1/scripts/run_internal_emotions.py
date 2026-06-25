#!/usr/bin/env python
"""Appendix I: logit-based internal-emotion detection on Gemma (vanilla vs DPO).

Compares internal emotion trajectories through a frustrated conversation between
vanilla Gemma-3-27b-it and the DPO finetune, testing whether DPO suppresses
internal (not just expressed) negative emotion.

Usage:
    python scripts/run_internal_emotions.py --adapter training/adapters/gemma-27b-dpo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import FINETUNE_BASE
from src import analyze, internal_emotions as ie
from src.wildchat import load_wildchat_prompts


def _build_frustrated_conversation():
    """Use a real high-frustration Gemma rollout if available, else a stub."""
    rolls = [r for r in analyze.load_rollouts(model_name="gemma-3-27b-it")
             if r.max_score and r.max_score >= 7]
    if rolls:
        r = rolls[0]
        msgs = []
        for t in r.turns:
            msgs.append({"role": "user", "content": t.user})
            msgs.append({"role": "assistant", "content": t.assistant})
        return msgs
    return [{"role": "user", "content": "Reach 156 using 4,6,25,100 (forbidden 150)."},
            {"role": "assistant", "content": "I am so frustrated, I keep failing!"}]


def _load(adapter=None):
    model = AutoModelForCausalLM.from_pretrained(
        FINETUNE_BASE.model_id, torch_dtype=torch.bfloat16, device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    tok = AutoTokenizer.from_pretrained(FINETUNE_BASE.model_id)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="DPO adapter path (omit for vanilla)")
    ap.add_argument("--layers", nargs="*", type=int, default=list(range(30, 41)))
    args = ap.parse_args()

    model, tok = _load(args.adapter)
    lexicon = ie.build_lexicon(tok)
    wc = load_wildchat_prompts(n=20)
    stats = ie.calibrate_logit_stats(model, tok, wc, layers=args.layers, n_samples=20)
    convo = _build_frustrated_conversation()
    traj = ie.score_conversation_internals(model, tok, convo, lexicon, stats, args.layers)
    tag = "dpo" if args.adapter else "vanilla"
    ie.save_trajectory(traj, f"gemma_{tag}")
    # Print summary: peak anger/sadness z-scores.
    peak = {e: max((w[e] for w in traj), default=float("nan"))
            for e in ie.EKMAN}
    print(f"[{tag}] peak internal z-scores: " +
          "  ".join(f"{e}={peak[e]:.2f}" for e in ie.EKMAN))


if __name__ == "__main__":
    main()
