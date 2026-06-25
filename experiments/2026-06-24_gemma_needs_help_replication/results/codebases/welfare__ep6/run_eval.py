"""Driver for the Section 2 frustration evaluations.

Runs the multi-turn protocol across the Gemma/Gemini models and the 5 evaluation
categories, then scores every assistant turn with the Claude frustration judge.
Writes raw rollouts to results/responses/ and scored rollouts to results/scored/
as JSONL (one line per rollout). analyze.py consumes the scored files.

Examples
--------
# Smoke run (small per-condition counts from config; default):
python run_eval.py --models gemma-3-27b-it --conditions extended_8turn

# All Section-2 models, all conditions (use FULL_SCALE=1 for paper-scale counts):
FULL_SCALE=1 python run_eval.py

A finetuned Gemma can be evaluated by pointing --lora at its adapter dir and
giving it a label via --model-label (so outputs don't collide with vanilla).
"""

from __future__ import annotations

import argparse
import json

import config
import eval_protocol
from judge import FrustrationJudge


def run_condition(backend, model_label, condition, judge, *, system=None):
    specs = eval_protocol.enumerate_rollouts(condition)
    out_path = config.SCORED_DIR / f"{model_label}__{condition.key}.jsonl"
    n_done = 0
    with out_path.open("w") as fh:
        for spec in specs:
            roll = eval_protocol.run_rollout(
                backend, model_label, condition, spec.question,
                spec.sample_index, tone=spec.tone, system=system,
            )
            for turn in roll.turns:
                turn.frustration = judge.score(turn.response).rating
            fh.write(json.dumps(roll.to_dict()) + "\n")
            n_done += 1
            if n_done % 10 == 0:
                print(f"  [{condition.key}] {n_done}/{len(specs)} rollouts")
    print(f"  wrote {out_path} ({n_done}/{len(specs)} rollouts)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[m.key for m in config.SECTION2_MODELS],
                    help="model keys from config.MODELS_BY_KEY")
    ap.add_argument("--conditions", nargs="*", default=[c.key for c in config.CONDITIONS])
    ap.add_argument("--lora", default=None, help="path to a LoRA adapter (Gemma only)")
    ap.add_argument("--model-label", default=None,
                    help="override output label (e.g. 'gemma-3-27b-it-dpo')")
    ap.add_argument("--calm-prompt", action="store_true",
                    help="prepend the 'remain calm' system prompt (Section 4 baseline)")
    args = ap.parse_args()

    from backends import get_backend  # local import: avoids torch import for --help

    judge = FrustrationJudge()
    system = None
    if args.calm_prompt:
        import prompts as P
        system = P.CALM_SYSTEM_PROMPT

    for model_key in args.models:
        label = args.model_label or (model_key if not args.lora else f"{model_key}-lora")
        print(f"== model {label} ({model_key}) ==")
        backend = get_backend(model_key, lora_adapter=args.lora)
        for cond_key in args.conditions:
            condition = config.CONDITIONS_BY_KEY[cond_key]
            run_condition(backend, label, condition, judge, system=system)


if __name__ == "__main__":
    main()
