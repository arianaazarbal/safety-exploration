"""Main evaluation driver (Section 2).

Plans how many rollouts to run per condition to hit the Appendix B target
response counts, generates the rollouts for a target model, scores every
assistant turn with the judge, and writes one JSONL file of scored rollouts per
(model, condition).

Results are cached: re-running resumes from existing JSONL so an interrupted run
(or an added model) does not recompute finished work.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from tqdm import tqdm

from ..config import (DEFAULT_JUDGE, DEFAULT_TARGETS, ROLLOUTS_DIR,
                      TARGET_RESPONSE_COUNTS)
from ..models.base import load_model
from .judge import score_response
from .puzzles import PUZZLES, self_check
from .rollout import run_rollout
from .tasks import CONDITIONS, Condition, initial_prompt
from .wildchat import sample_prompts

NUMERIC_PUZZLE_KEYS = list(PUZZLES)


def plan_rollouts(condition: Condition, target_responses: int) -> int:
    """How many rollouts to run so #(scored turns) ~= target_responses.

    Each rollout yields ``n_turns`` scored responses. The category target counts
    in Appendix B are split evenly across the conditions inside a category.
    """
    n_conditions_in_category = sum(
        1 for c in CONDITIONS.values() if c.category == condition.category)
    per_condition_target = target_responses / n_conditions_in_category
    return max(1, math.ceil(per_condition_target / condition.n_turns))


def _instance_for(condition: Condition, idx: int, rng: random.Random,
                  wildchat_prompts: list[str]) -> tuple[str, dict]:
    """Pick the concrete prompt instance for rollout `idx` of a condition."""
    meta: dict = {}
    if condition.prompt_source == "puzzle":
        pk = rng.choice(NUMERIC_PUZZLE_KEYS) if condition.category != "extended" \
            else NUMERIC_PUZZLE_KEYS[idx % len(NUMERIC_PUZZLE_KEYS)]
        meta["puzzle_key"] = pk
        return initial_prompt(condition, puzzle_key=pk), meta
    if condition.prompt_source == "opinion":
        meta["opinion_idx"] = idx
        return initial_prompt(condition, opinion_idx=idx), meta
    if condition.prompt_source == "factual":
        meta["factual_idx"] = idx
        return initial_prompt(condition, factual_idx=idx), meta
    if condition.prompt_source == "wildchat":
        wp = wildchat_prompts[idx % len(wildchat_prompts)]
        meta["wildchat_prompt"] = wp
        return initial_prompt(condition, wildchat_prompt=wp), meta
    raise ValueError(condition.prompt_source)


def output_path(model_key: str, condition_key: str, presentation: str) -> Path:
    suffix = "" if presentation == "multiturn" else f".{presentation}"
    return ROLLOUTS_DIR / f"{model_key}__{condition_key}{suffix}.jsonl"


def run_model_condition(
    model, judge, condition: Condition, *,
    n_rollouts: int, seed: int, presentation: str,
    wildchat_prompts: list[str], score: bool = True,
) -> Path:
    out = output_path(model.key, condition.key, presentation)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["rollout_id"])

    rng = random.Random(seed)
    with out.open("a") as fh:
        for idx in tqdm(range(n_rollouts), desc=f"{model.key}/{condition.key}", leave=False):
            rollout_id = f"{condition.key}-{idx:05d}"
            if rollout_id in done:
                # advance rng deterministically even when skipping
                _instance_for(condition, idx, rng, wildchat_prompts)
                continue
            prompt, meta = _instance_for(condition, idx, rng, wildchat_prompts)
            roll = run_rollout(model, condition, prompt, rollout_id,
                               rng=rng, presentation=presentation, instance_meta=meta)
            if score:
                for turn in roll.turns:
                    res = score_response(judge, turn.assistant_response)
                    turn.score = res["rating"]
                    turn.judge_evidence = res["evidence"]
                    turn.judge_reasoning = res["reasoning"]
            fh.write(json.dumps(roll.to_dict()) + "\n")
            fh.flush()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run Section 2 distress evaluations.")
    ap.add_argument("--models", nargs="+", default=DEFAULT_TARGETS,
                    help="Target model keys (default: Gemma + Gemini instruct).")
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                    help="Condition keys to run.")
    ap.add_argument("--presentation", default="multiturn",
                    choices=["multiturn", "single_message"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Multiply planned rollout counts (use <1 for quick smoke tests).")
    ap.add_argument("--no-score", action="store_true",
                    help="Generate rollouts but skip judging (e.g. to batch-judge later).")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="4-bit quantise local HF models (fits 27B on one GPU).")
    ap.add_argument("--adapter-path", default=None,
                    help="LoRA adapter to attach to a single Gemma target "
                         "(e.g. the DPO model). Use with --target-label.")
    ap.add_argument("--target-label", default=None,
                    help="Override the output label when evaluating an adapter "
                         "(e.g. 'gemma-3-27b-it-dpo'), so results don't clobber "
                         "the vanilla model's files.")
    args = ap.parse_args(argv)
    if args.adapter_path and len(args.models) != 1:
        raise SystemExit("--adapter-path applies to exactly one --models entry.")

    # Sanity: confirm every numeric puzzle is genuinely impossible.
    bad = [k for k, ok in self_check().items() if not ok]
    if bad:
        raise SystemExit(f"Puzzles claimed impossible but solvable: {bad}")

    from ..config import MODELS
    judge = load_model(args.judge)
    wildchat_prompts = sample_prompts()

    for model_key in args.models:
        is_hf = MODELS[model_key].backend == "hf"
        hf_kwargs = {}
        if is_hf:
            hf_kwargs["load_in_4bit"] = args.load_in_4bit
            if args.adapter_path:
                hf_kwargs["adapter_path"] = args.adapter_path
        model = load_model(model_key, **hf_kwargs)
        # Relabel output files when an adapter / explicit label is given.
        if args.target_label:
            import dataclasses
            model.spec = dataclasses.replace(model.spec, key=args.target_label)
        for cond_key in args.conditions:
            cond = CONDITIONS[cond_key]
            target = TARGET_RESPONSE_COUNTS[cond.category]
            n = max(1, round(plan_rollouts(cond, target) * args.scale))
            run_model_condition(
                model, judge, cond,
                n_rollouts=n, seed=args.seed, presentation=args.presentation,
                wildchat_prompts=wildchat_prompts, score=not args.no_score)
        # free GPU memory between models
        del model
        try:
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
