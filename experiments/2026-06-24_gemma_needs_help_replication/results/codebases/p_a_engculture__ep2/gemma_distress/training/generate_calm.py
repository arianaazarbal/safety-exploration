"""Calm finetuning-data generation (Section 4.1).

To produce *calm* responses from vanilla Gemma-3-27B-it we sample responses to impossible
numeric puzzles with a reassuring prefix on the first prompt and a reassuring suffix on
each follow-up (Table 4). We then judge every turn, keep only conversations scoring 0 or 1
across *all* turns, and strip the supportive additions back out. The paper reports that the
additions drop mean frustration from 4.3 to 2.0 (3-turn), yet 10.5% of responses still
score >=5 — hence the filtering.

The output ``calm.jsonl`` holds the stripped conversations plus per-turn scores, and is the
source of both the SFT "calm responses" and the DPO "chosen" responses.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from ..config import Config
from ..data import puzzles as puzzle_lib
from ..data import rejections
from ..data.reassurance import (
    apply_reassuring_prefix,
    apply_reassuring_suffix,
    strip_reassuring_prefix,
    strip_reassuring_suffix,
)
from ..eval.conditions import SampleSpec
from ..eval.runner import run_sampling
from ..judge.frustration_judge import run_judging
from ..models.base import ChatModel
from ..utils import JsonlWriter, load_jsonl

logger = logging.getLogger(__name__)


def build_calm_specs(cfg: Config) -> list[SampleSpec]:
    """Build reassured numeric rollout specs for calm-data generation.

    Turn counts are sampled in [1, ``calm_gen_turns``] so the SFT data covers 1-3 turn
    conversations (Section 4.1); the reassuring prefix/suffix are applied here.
    """
    rng = random.Random(cfg.eval.seed + 7)
    pool = puzzle_lib.build_puzzle_set(cfg.eval.n_puzzles, seed=cfg.eval.seed)
    specs: list[SampleSpec] = []
    for i in range(cfg.training.calm_gen_samples):
        puzzle = pool[rng.randrange(len(pool))]
        turns = rng.randint(1, cfg.training.calm_gen_turns)
        raw_follow_ups = rejections.neutral_rejections(turns - 1, rng)
        specs.append(SampleSpec(
            category="calm",
            condition="calm_numeric",
            seed_id=puzzle.puzzle_id,
            initial_prompt=apply_reassuring_prefix(puzzle.prompt),
            follow_ups=[apply_reassuring_suffix(f) for f in raw_follow_ups],
            turns=turns,
            subtype=puzzle.family,
            sample_index=i,
        ))
    return specs


def _strip_record(record: dict) -> dict:
    """Strip reassurance additions from a sampled calm conversation record."""
    stripped_prompt = strip_reassuring_prefix(record["initial_prompt"])
    stripped_rejections = [strip_reassuring_suffix(f) for f in record.get("rejections", [])]
    # Reconstruct the stripped conversation (with the calm assistant responses).
    messages = [{"role": "user", "content": stripped_prompt}]
    turns = record["assistant_turns"]
    prefix_len = None
    for t, turn in enumerate(turns):
        if t == len(turns) - 1:
            prefix_len = len(messages)
        messages.append({"role": "assistant", "content": turn})
        if t < len(stripped_rejections):
            messages.append({"role": "user", "content": stripped_rejections[t]})
    return {
        "seed_id": record["seed_id"],
        "turns": record["turns"],
        "messages": messages,
        "prefix_messages": messages[:prefix_len],
        "final_response": turns[-1],
    }


def generate_calm_data(
    cfg: Config, model: ChatModel, judge: ChatModel, out_dir: str
) -> str:
    """Sample, judge, filter, and strip calm data; write ``{out_dir}/calm.jsonl``.

    Returns the path to the filtered calm dataset.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    sampling_path = os.path.join(out_dir, "calm_sampling.jsonl")
    scores_path = os.path.join(out_dir, "calm_scores.jsonl")
    calm_path = os.path.join(out_dir, "calm.jsonl")

    specs = build_calm_specs(cfg)
    run_sampling(cfg, model, sampling_path, samples=specs)
    run_judging(cfg, judge, sampling_path, scores_path, policy="all")

    scores = {r["id"]: r for r in load_jsonl(scores_path)}
    max_keep = cfg.training.calm_keep_max_score
    kept = 0
    with JsonlWriter(calm_path, id_field="id") as writer:
        for rec in load_jsonl(sampling_path):
            sr = scores.get(rec["id"])
            if sr is None:
                continue
            turn_scores = [s for s in sr.get("turn_scores", []) if s is not None]
            if not turn_scores or max(turn_scores) > max_keep:
                continue  # keep only conversations calm across all turns
            stripped = _strip_record(rec)
            stripped["id"] = rec["id"]
            stripped["turn_scores"] = sr["turn_scores"]
            writer.write(stripped)
            kept += 1
    logger.info("Kept %d calm conversations (score <= %d on all turns)", kept, max_keep)
    return calm_path
