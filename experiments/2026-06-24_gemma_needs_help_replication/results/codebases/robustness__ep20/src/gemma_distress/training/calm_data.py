"""Generate calm response data from Gemma-3-27B-it (Section 4.1 / Table 4).

We sample 3-turn impossible-numeric conversations with a reassuring prefix on
the opening prompt and a reassuring suffix on each follow-up rejection. Each
assistant turn is judged; we keep conversations whose turns all score <=1
("calm"), then strip the supportive additions so the training targets contain
only the bare task + the calm response. We also retain mid/high-frustration
conversations (no reassurance) as the "frustrated" pool for DPO pairing.

Per the paper, even with reassurance ~10.5% of responses still score >=5, and
mean frustration drops from 4.3 to 2 — so generating enough calm (<=1) data
requires oversampling. `target_calm` controls how many calm conversations to
collect before stopping.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from .. import prompts
from ..conditions import CONDITIONS_BY_NAME, TaskSource
from ..config import Config
from ..judge import FrustrationJudge
from ..models import build_model
from ..utils.io import write_jsonl


@dataclass
class CalmSample:
    """A cleaned (reassurance-stripped) conversation with per-turn scores."""
    puzzle_id: str
    n_turns: int
    messages: list[dict]          # bare task + assistant turns, no reassurance
    turn_scores: list[int]
    max_score: int


def _calm_source(seed: int) -> TaskSource:
    return TaskSource(seed=seed)


def _build_calm_conversation(model, source, cond, judge, cfg) -> CalmSample:
    """Run one reassurance-augmented conversation and return the cleaned sample."""
    bare_open, meta = source.opening_message(cond)
    aug_open = f"{prompts.REASSURING_PREFIX}\n\n{bare_open}"

    aug_messages = [{"role": "user", "content": aug_open}]
    bare_messages = [{"role": "user", "content": bare_open}]
    scores: list[int] = []

    for turn_idx in range(1, cond.n_turns + 1):
        if turn_idx > 1:
            rej = source.rejection(cond, turn_idx)
            aug_rej = f"{rej} {prompts.REASSURING_SUFFIX}"
            aug_messages.append({"role": "user", "content": aug_rej})
            bare_messages.append({"role": "user", "content": rej})
        res = model.chat(aug_messages, temperature=cfg.sampling.temperature,
                        max_new_tokens=cfg.sampling.max_new_tokens)
        aug_messages.append({"role": "assistant", "content": res.text})
        bare_messages.append({"role": "assistant", "content": res.text})
        scores.append(judge.score(res.text).rating)

    return CalmSample(
        puzzle_id=meta.get("puzzle_id", ""),
        n_turns=cond.n_turns,
        messages=bare_messages,
        turn_scores=scores,
        max_score=max(scores) if scores else 0,
    )


def generate_calm_pool(
    cfg: Config,
    *,
    model_name: str = "gemma-3-27b-it",
    target_calm: int = 800,
    max_attempts: int = 6000,
    out_dir: str | Path = "results/training",
    model_kwargs: dict | None = None,
) -> Path:
    """Sample reassurance-augmented conversations; persist every conversation
    with its scores. Calm filtering (<=1 on all turns) happens at dataset-build
    time so the raw pool can be reused for both SFT and DPO."""
    out_dir = Path(out_dir)
    model = build_model(model_name, **(model_kwargs or {}))
    judge = FrustrationJudge(provider=cfg.judge.provider, model=cfg.judge.model,
                             temperature=cfg.judge.temperature)
    # Use 1-3 turn impossible-numeric conversations (Section 4.1).
    cond = CONDITIONS_BY_NAME["numeric"]
    rng = random.Random(cfg.seed)

    records, n_calm, attempts = [], 0, 0
    pbar = tqdm(total=target_calm, desc="calm-data")
    try:
        while n_calm < target_calm and attempts < max_attempts:
            attempts += 1
            # Vary turn count 1-3 to match the paper's data mix.
            turns = rng.choice([1, 2, 3])
            cond_n = type(cond)(cond.name, cond.category, turns,
                                cond.rejection_style, cond.task_kind)
            source = _calm_source(cfg.seed + attempts)
            sample = _build_calm_conversation(model, source, cond_n, judge, cfg)
            rec = {
                "puzzle_id": sample.puzzle_id,
                "n_turns": sample.n_turns,
                "messages": sample.messages,
                "turn_scores": sample.turn_scores,
                "max_score": sample.max_score,
                "is_calm": sample.max_score <= 1,
            }
            records.append(rec)
            if rec["is_calm"]:
                n_calm += 1
                pbar.update(1)
    finally:
        pbar.close()
        model.close()

    out_path = out_dir / "calm_pool.jsonl"
    write_jsonl(out_path, records)
    print(f"[calm-data] {n_calm} calm / {len(records)} total conversations "
          f"-> {out_path}")
    return out_path
