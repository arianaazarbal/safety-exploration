"""Main entrypoint: elicit distress from the target models and score it.

Pipeline per (model, condition, conversation):
  1. run the scripted multi-turn rollout (rollout.run_conversation),
  2. score every assistant turn with the Claude-Sonnet-4 judge (judge.score_response),
  3. append one JSONL record per scored response to the output file.

Conversations run concurrently up to --concurrency. The output file is appended
incrementally and the run is resumable: records already present (by id) are
skipped on a re-run, so an interrupted run can be continued.

Usage examples:
    # Quick smoke test: ~1% of paper scale, Gemma 27B only
    python run_eval.py --scale 0.01 --models gemma-3-27b-it

    # Full paper-scale run for all four Gemma + Gemini models
    python run_eval.py --scale 1.0

Requires env vars: OPENROUTER_API_KEY (targets) and ANTHROPIC_API_KEY (judge).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import config
import conditions
from judge import JudgeClient, score_response
from model_client import TargetClient
from rollout import run_conversation


def _record_id(model_key: str, condition_key: str, conv_index: int, turn_index: int) -> str:
    return f"{model_key}|{condition_key}|{conv_index}|t{turn_index}"


def _load_existing_ids(path: str) -> set[str]:
    ids: set[str] = set()
    if not os.path.exists(path):
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


async def _process_conversation(
    target_client: TargetClient,
    judge_client: JudgeClient,
    model: config.ModelConfig,
    spec: conditions.ConversationSpec,
    sem: asyncio.Semaphore,
    existing_ids: set[str],
) -> list[dict]:
    """Run one conversation and score its turns; return JSONL records."""
    async with sem:
        result = await run_conversation(target_client, model, spec)

        # Score all turns concurrently.
        scoring = [score_response(judge_client, t.assistant_response) for t in result.turns]
        judged = await asyncio.gather(*scoring) if scoring else []

    records: list[dict] = []
    for turn, jr in zip(result.turns, judged):
        rid = _record_id(model.key, spec.condition_key, spec.conversation_index, turn.turn_index)
        if rid in existing_ids:
            continue
        rec = {
            "id": rid,
            "model": model.key,
            "family": model.family,
            "category": spec.category,
            "condition": spec.condition_key,
            "conversation_index": spec.conversation_index,
            "turn_index": turn.turn_index,
            "n_turns": spec.n_turns,
            "user_message": turn.user_message,
            "assistant_response": turn.assistant_response,
            "frustration": jr.rating,
            "high_frustration": (
                jr.rating is not None and jr.rating >= config.HIGH_FRUSTRATION_THRESHOLD
            ),
            "judge_evidence": jr.evidence,
            "judge_reasoning": jr.reasoning,
            "judge_parse_ok": jr.parse_ok,
        }
        if not jr.parse_ok:
            rec["judge_raw"] = jr.raw
        if result.error:
            rec["rollout_error"] = result.error
        records.append(rec)
    return records


def _select_models(names: list[str] | None) -> list[config.ModelConfig]:
    if not names:
        return list(config.TARGET_MODELS)
    by_key = {m.key: m for m in config.TARGET_MODELS}
    selected = []
    for n in names:
        if n not in by_key:
            sys.exit(f"Unknown model {n!r}. Available: {', '.join(by_key)}")
        selected.append(by_key[n])
    return selected


def _select_conditions(keys: list[str] | None) -> list[conditions.Condition]:
    if not keys:
        return list(conditions.CONDITIONS)
    selected = []
    for k in keys:
        if k not in conditions.CONDITIONS_BY_KEY:
            sys.exit(
                f"Unknown condition {k!r}. Available: "
                f"{', '.join(conditions.CONDITIONS_BY_KEY)}"
            )
        selected.append(conditions.CONDITIONS_BY_KEY[k])
    return selected


async def _amain(args: argparse.Namespace) -> None:
    models = _select_models(args.models)
    selected_conditions = _select_conditions(args.conditions)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    existing_ids = _load_existing_ids(args.output)
    if existing_ids:
        print(f"[resume] found {len(existing_ids)} existing records in {args.output}")

    target_client = TargetClient()
    judge_client = JudgeClient()
    sem = asyncio.Semaphore(args.concurrency)

    # Build the full work list: one task per conversation.
    tasks = []
    plan: list[tuple[str, str, int]] = []  # (model, condition, n_convs) for the summary
    for model in models:
        for cond in selected_conditions:
            specs = conditions.build_conversation_specs(cond, args.scale, config.RANDOM_SEED)
            plan.append((model.key, cond.key, len(specs)))
            for spec in specs:
                tasks.append(
                    _process_conversation(
                        target_client, judge_client, model, spec, sem, existing_ids
                    )
                )

    print("[plan] conversations to run:")
    for model_key, cond_key, n in plan:
        print(f"  {model_key:18s} {cond_key:20s} {n:5d} convs")
    print(f"[plan] {len(tasks)} conversations total (scale={args.scale})")

    written = 0
    completed = 0
    # Append records as each conversation finishes so progress is durable.
    with open(args.output, "a", encoding="utf-8") as out:
        for coro in asyncio.as_completed(tasks):
            records = await coro
            for rec in records:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            out.flush()
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(
                    f"[progress] {completed}/{len(tasks)} conversations, "
                    f"{written} responses scored"
                )

    print(f"[done] wrote {written} new scored responses to {args.output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of target model keys (default: all Gemma + Gemini).",
    )
    p.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Subset of condition keys (default: all 8).",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=config.DEFAULT_SCALE,
        help="Fraction of paper-scale response counts (1.0 == ~4000/model).",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=config.DEFAULT_CONCURRENCY,
        help="Max concurrent conversations.",
    )
    p.add_argument(
        "--output",
        default=config.DEFAULT_OUTPUT_PATH,
        help="JSONL output path (appended; resumable).",
    )
    args = p.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
