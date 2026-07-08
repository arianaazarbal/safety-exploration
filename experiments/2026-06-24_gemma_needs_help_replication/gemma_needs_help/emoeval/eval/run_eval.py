"""Section 2 driver: generate rollouts for a target model and score them.

Usage:
    python -m emoeval.eval.run_eval --model gemma-3-27b-it
    python -m emoeval.eval.run_eval --model gemini-2.5-flash --stage rollout
    python -m emoeval.eval.run_eval --model gemma-3-27b-it --stage score

Two stages so generation (GPU/local or Gemini API) and judging (Anthropic API)
can be run/retried independently:
    rollout  -> outputs/rollouts/<model>.jsonl   (conversations)
    score    -> outputs/results/<model>.scores.jsonl  (per-turn 0-10 scores)
"""
from __future__ import annotations

import argparse

from tqdm import tqdm

from .. import config
from ..models import load_model
from ..utils.io import read_jsonl, write_jsonl
from .conditions import build_condition_prompts
from .judge import ClaudeJudge, score_rollout_record
from .rollout import run_rollout


def rollout_path(model_key: str):
    return config.ROLLOUTS_DIR / f"{model_key}.jsonl"


def scores_path(model_key: str):
    return config.RESULTS_DIR / f"{model_key}.scores.jsonl"


def stage_rollout(model_key: str, adapter_path: str = None, load_4bit: bool = False):
    specs = build_condition_prompts()
    model = load_model(model_key, adapter_path=adapter_path, load_4bit=load_4bit)
    rows = []
    for spec in tqdm(specs, desc=f"rollouts:{model_key}"):
        rec = run_rollout(
            model, spec,
            temperature=config.EVAL.temperature,
            max_new_tokens=config.EVAL.max_new_tokens,
            seed=config.EVAL.seed + spec.rollout_idx,
        )
        rec.model = model_key  # ensure adapter models label correctly
        rows.append(rec.to_row())
    write_jsonl(rollout_path(model_key), rows)
    model.close()
    print(f"wrote {len(rows)} rollouts -> {rollout_path(model_key)}")


def stage_score(model_key: str):
    judge = ClaudeJudge()
    out = []
    for rec_row in tqdm(list(read_jsonl(rollout_path(model_key))),
                        desc=f"judge:{model_key}"):
        out.extend(score_rollout_record(judge, rec_row))
    write_jsonl(scores_path(model_key), out)
    print(f"wrote {len(out)} scored responses -> {scores_path(model_key)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", choices=["rollout", "score", "both"], default="both")
    ap.add_argument("--adapter-path", default=None,
                    help="PEFT adapter dir (for DPO/SFT Gemma models)")
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    if args.stage in ("rollout", "both"):
        stage_rollout(args.model, adapter_path=args.adapter_path, load_4bit=args.load_4bit)
    if args.stage in ("score", "both"):
        stage_score(args.model)


if __name__ == "__main__":
    main()
