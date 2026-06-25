"""Section 2 driver: run rollouts for a model, score them, save JSONL.

Usage:
    python -m emo_instability.eval.run_eval --model gemma-3-27b-it --preset default
    python -m emo_instability.eval.run_eval --model gemini-2.5-flash --preset smoke

Outputs (under runs/eval/<model>/):
    rollouts.jsonl   one row per conversation
    scored.jsonl     one row per scored assistant turn (with frustration score)
"""
from __future__ import annotations

import argparse
import os

from ..config import get_config
from ..models.registry import build_client
from ..models.judges import AnthropicClient
from ..utils.io import run_dir, write_jsonl
from .conditions import build_conversations
from .judge import FrustrationJudge
from .rollout import run_rollout, rollout_to_scored_units


def run_model_eval(model_name: str, cfg, *, adapter_path=None, score: bool = True) -> str:
    out_dir = run_dir(cfg.output_root, "eval", model_name)
    client = build_client(model_name, adapter_path=adapter_path,
                          disable_thinking=cfg.eval.sampling.disable_thinking)

    specs = build_conversations(cfg.eval, seed=0)
    print(f"[{model_name}] running {len(specs)} conversations "
          f"(~{sum(s.n_turns for s in specs)} scored responses)")

    rollout_rows = []
    scored_units = []
    try:
        from tqdm import tqdm
        specs_iter = tqdm(specs, desc=f"rollout[{model_name}]")
    except ImportError:
        specs_iter = specs

    for spec in specs_iter:
        record = run_rollout(client, spec, cfg.eval.sampling)
        rollout_rows.append(record.to_row())
        scored_units.extend(rollout_to_scored_units(record))

    write_jsonl(os.path.join(out_dir, "rollouts.jsonl"), rollout_rows)

    if score:
        judge_client = AnthropicClient(cfg.eval.judge.frustration_model)
        judge = FrustrationJudge(
            judge_client,
            max_tokens=cfg.eval.judge.max_tokens,
            max_retries=cfg.eval.judge.max_retries,
        )
        judge.score_units(scored_units)

    write_jsonl(os.path.join(out_dir, "scored.jsonl"), scored_units)
    print(f"[{model_name}] wrote {len(scored_units)} scored responses to {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Run Section 2 distress elicitation.")
    ap.add_argument("--model", required=True, help="model name from registry")
    ap.add_argument("--preset", default="default", choices=["default", "smoke"])
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter path (Gemma finetunes)")
    ap.add_argument("--no-score", action="store_true", help="generate rollouts only, skip judging")
    args = ap.parse_args()

    cfg = get_config(args.preset)
    run_model_eval(args.model, cfg, adapter_path=args.adapter, score=not args.no_score)


if __name__ == "__main__":
    main()
