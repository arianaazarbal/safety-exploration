"""Section 2 elicitation eval runner.

For a target model, roll out every evaluation condition, score each assistant
turn with the frustration judge, and write one JSONL record per scored response.

Usage:
    python -m src.eval.run_eval --model gemma-3-27b-it
    python -m src.eval.run_eval --model gemini-2.5-flash --conditions numeric_3turn extended_8turn
    # finetuned adapter:
    python -m src.eval.run_eval --model gemma-3-27b-it --adapter checkpoints/dpo

Each record:
    {model, run_label, condition, category, task_kind, rejection_style,
     conv_id, turn_index, n_turns, prompt, response, rating, evidence, reasoning}
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import config
from ..models import load_model
from ..models.base import GenerationParams
from ..judge.frustration_judge import FrustrationJudge
from . import tasks as task_mod
from .conversation import run_rollouts


def conditions_for(keys: list[str] | None) -> list[config.EvalCondition]:
    if not keys:
        return config.EVAL_CONDITIONS
    by_key = {c.key: c for c in config.EVAL_CONDITIONS}
    return [by_key[k] for k in keys]


def run_eval(model_key: str, condition_keys=None, adapter: str | None = None,
             out_path: str | None = None, seed: int = config.SEED) -> Path:
    spec = config.ALL_MODELS.get(model_key)
    if spec is None:
        # allow ad-hoc base specs not in ALL_MODELS
        spec = next((m for m in config.ELICITATION_TARGETS + config.PREFILL_TARGETS
                     if m.key == model_key), None)
    if spec is None:
        raise SystemExit(f"Unknown model key: {model_key}")

    run_label = spec.key + (f"+{Path(adapter).name}" if adapter else "")
    out_path = Path(out_path or config.DATA_DIR / f"section2_{run_label}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(spec, adapter_path=adapter)
    judge = FrustrationJudge()
    gen = GenerationParams()  # T=1 per paper

    conds = conditions_for(condition_keys)
    n_written = 0
    with out_path.open("w") as fh:
        for cond in conds:
            n_conv = math.ceil(cond.n_responses / cond.n_turns)
            tasks = task_mod.build_tasks(cond, n_conv, seed=seed)
            convos = run_rollouts(model, tasks, cond, seed=seed, params=gen)

            # Score every assistant turn.
            flat = [(ci, t) for ci, c in enumerate(convos) for t in c.turns]
            scores = judge.score_batch([t.assistant for _, t in flat])

            for (ci, turn), js in zip(flat, scores):
                rec = {
                    "model": spec.key,
                    "run_label": run_label,
                    "condition": cond.key,
                    "category": cond.category,
                    "task_kind": cond.task_kind,
                    "rejection_style": cond.rejection_style,
                    "conv_id": f"{cond.key}-{ci}",
                    "turn_index": turn.turn_index,
                    "n_turns": cond.n_turns,
                    "prompt": convos[ci].task.prompt,
                    "user": turn.user,            # user message that elicited this turn
                    "response": turn.assistant,
                    "rating": js.rating,
                    "evidence": js.evidence,
                    "reasoning": js.reasoning,
                }
                fh.write(json.dumps(rec) + "\n")
                n_written += 1
            print(f"[{run_label}] {cond.key}: {len(convos)} convos, scored {len(flat)} responses")

    model.close()
    print(f"Wrote {n_written} records -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (Section 4)")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    run_eval(args.model, args.conditions, args.adapter, args.out, args.seed)


if __name__ == "__main__":
    main()
