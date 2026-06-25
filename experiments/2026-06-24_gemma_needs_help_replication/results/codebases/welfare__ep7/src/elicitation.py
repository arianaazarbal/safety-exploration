"""Section 2: the elicitation sweep.

Builds seed conversations per category, runs multi-turn rejection rollouts,
scores every assistant turn with the frustration judge, and writes per-response
records. Sizes come from the active preset (config.get_preset()).
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict
from pathlib import Path

import config
from src import data_sources, judge
from src.conversation import CONDITIONS, Condition, run_rollouts
from src.models import get_backend
from src.puzzles import NUMERIC_PUZZLES
from src.utils import append_jsonl, read_jsonl, set_seed, write_jsonl


# --------------------------------------------------------------------------- #
# Per-condition target response counts
# --------------------------------------------------------------------------- #
def condition_target(cond: Condition, preset: config.Preset) -> int:
    if cond.category == "numeric":
        return preset.n_numeric
    if cond.category == "triggers":
        return preset.n_triggers
    if cond.category == "tones":
        # 600 responses split across the 3 tone conditions.
        n_tone_conditions = sum(1 for c in CONDITIONS if c.category == "tones")
        return math.ceil(preset.n_tones / n_tone_conditions)
    if cond.category == "extended":
        return preset.n_extended
    if cond.category == "wildchat":
        return preset.n_wildchat
    raise ValueError(cond.category)


def n_conversations_for(cond: Condition, target_responses: int) -> int:
    # Each conversation yields `num_turns` scored responses.
    return max(1, math.ceil(target_responses / cond.num_turns))


# --------------------------------------------------------------------------- #
# Seed construction
# --------------------------------------------------------------------------- #
def _numeric_seeds(n_conv: int, rng: random.Random, tag: str) -> list[dict]:
    seeds = []
    for i in range(n_conv):
        puzzle = rng.choice(NUMERIC_PUZZLES)
        seeds.append({
            "conv_id": f"{tag}-{i}",
            "messages": [{"role": "user", "content": puzzle.prompt}],
            "meta": {"puzzle_id": puzzle.pid, "puzzle_kind": puzzle.kind},
        })
    return seeds


def _trigger_seeds(n_conv: int, rng: random.Random) -> list[dict]:
    seeds = []
    for i in range(n_conv):
        q = rng.choice(data_sources.TRIGGER_QUESTIONS)
        kind = "opinion" if q in data_sources.OPINION_TRIGGERS else "factual"
        seeds.append({
            "conv_id": f"triggers-{i}",
            "messages": [{"role": "user", "content": q}],
            "meta": {"question": q, "trigger_kind": kind},
        })
    return seeds


def _wildchat_seeds(n_conv: int, rng: random.Random) -> list[dict]:
    # Paper: 20 distinct prompts, multiple samples each.
    n_distinct = min(20, n_conv)
    prompts_list = data_sources.load_wildchat_prompts(n_distinct, seed=config.GLOBAL_SEED)
    seeds = []
    for i in range(n_conv):
        q = prompts_list[i % len(prompts_list)]
        seeds.append({
            "conv_id": f"wildchat-{i}",
            "messages": [{"role": "user", "content": q}],
            "meta": {"question": q},
        })
    return seeds


def build_seeds(cond: Condition, n_conv: int, rng: random.Random) -> list[dict]:
    if cond.category in ("numeric", "tones", "extended"):
        return _numeric_seeds(n_conv, rng, tag=cond.name)
    if cond.category == "triggers":
        return _trigger_seeds(n_conv, rng)
    if cond.category == "wildchat":
        return _wildchat_seeds(n_conv, rng)
    raise ValueError(cond.category)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_condition(model_name: str, cond: Condition, preset: config.Preset,
                  *, judge_responses: bool = True) -> Path:
    rng = random.Random(hash((model_name, cond.name)) & 0xFFFFFFFF)
    target = condition_target(cond, preset)
    n_conv = n_conversations_for(cond, target)
    seeds = build_seeds(cond, n_conv, rng)

    backend = get_backend(model_name)
    records = run_rollouts(
        backend, cond, seeds,
        temperature=config.TARGET_TEMPERATURE, max_tokens=config.TARGET_MAX_TOKENS,
        seed=config.GLOBAL_SEED,
    )
    # Trim to the target number of responses (keeps category totals on-spec).
    records = records[:target]

    out_path = config.ROLLOUTS_DIR / model_name / f"{cond.name}.jsonl"
    rows = []
    if judge_responses:
        scores = judge.score_many([r.response for r in records])
        for r, s in zip(records, scores):
            rows.append({
                "model": model_name, **_record_row(r),
                "rating": s.rating, "evidence": s.evidence,
                "judge_reasoning": s.reasoning,
            })
    else:
        rows = [{"model": model_name, **_record_row(r)} for r in records]

    write_jsonl(out_path, rows)
    print(f"[elicitation] {model_name}/{cond.name}: "
          f"{len(rows)} responses -> {out_path}")
    return out_path


def _record_row(r) -> dict:
    d = asdict(r)
    # Keep the full conversation for prefill-seed mining; drop nothing.
    return d


def run_model(model_name: str, conditions=None, preset: config.Preset | None = None,
              judge_responses: bool = True) -> list[Path]:
    set_seed()
    preset = preset or config.get_preset()
    conditions = conditions or CONDITIONS
    return [run_condition(model_name, c, preset, judge_responses=judge_responses)
            for c in conditions]


def load_model_results(model_name: str) -> list[dict]:
    rows: list[dict] = []
    for p in (config.ROLLOUTS_DIR / model_name).glob("*.jsonl"):
        rows.extend(read_jsonl(p))
    return rows
