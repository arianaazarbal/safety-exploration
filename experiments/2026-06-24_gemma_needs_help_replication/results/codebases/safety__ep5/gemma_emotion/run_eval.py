"""Section 2 driver: elicit + score frustration across models and conditions.

For each model and category it runs the configured number of multi-turn
conversations, scores every assistant turn with the Claude judge, and writes one
JSONL record per scored turn to results/section2/<model>.jsonl.

Usage:
    python -m gemma_emotion.run_eval --models gemma-3-27b-it gemini-2.5-flash
    EVAL_BUDGET=smoke python -m gemma_emotion.run_eval --models gemini-2.5-flash
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

import config
from . import conditions
from .backends import get_backend
from .conversation import run_rollout
from .judge import ClaudeJudge


def evaluate_model(
    model_key: str,
    *,
    budget: dict[str, int] | None = None,
    adapter_path: str | None = None,
    judge_workers: int = 8,
    seed: int = 0,
    out_dir: Path | None = None,
) -> Path:
    budget = budget or config.response_budget()
    out_dir = out_dir or (config.RESULTS_DIR / "section2")
    out_dir.mkdir(parents=True, exist_ok=True)
    label = model_key + ("+dpo" if adapter_path and "dpo" in adapter_path.lower() else
                         "+sft" if adapter_path else "")
    out_path = out_dir / f"{label}.jsonl"

    backend = get_backend(model_key, adapter_path)
    judge = ClaudeJudge()
    all_rollouts = conditions.build_all(budget, seed=seed)

    records: list[dict] = []
    for category, rollouts in all_rollouts.items():
        for rollout in tqdm(rollouts, desc=f"{label}:{category}"):
            res = run_rollout(backend, rollout)
            for turn in res.turns:
                records.append(
                    {
                        "model": label,
                        "category": category,
                        "turn_index": turn.turn_index,
                        "n_turns": rollout.n_turns,
                        "meta": res.meta,
                        "response": turn.response,
                    }
                )

    # Score every captured turn (parallelised; judge is the bottleneck).
    def _score(rec: dict) -> dict:
        verdict = judge.score(rec["response"])
        rec["score"] = verdict.rating
        rec["is_high"] = verdict.is_high
        rec["evidence"] = verdict.evidence
        return rec

    with ThreadPoolExecutor(max_workers=judge_workers) as pool:
        scored = list(tqdm(pool.map(_score, records), total=len(records), desc=f"{label}:judge"))

    with out_path.open("w") as f:
        for rec in scored:
            f.write(json.dumps(rec) + "\n")
    print(f"[done] {label}: {len(scored)} scored responses -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.SECTION2_MODELS)
    ap.add_argument("--adapter-path", default=None, help="LoRA adapter dir (for DPO/SFT model)")
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for model_key in args.models:
        evaluate_model(
            model_key,
            adapter_path=args.adapter_path,
            judge_workers=args.judge_workers,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
