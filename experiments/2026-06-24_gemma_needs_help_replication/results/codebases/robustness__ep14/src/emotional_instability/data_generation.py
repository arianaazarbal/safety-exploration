"""Section 4.1 calm-data generation + SFT/DPO dataset construction.

Procedure:
  - Generate Gemma-3-27b-it responses to impossible numeric puzzles with reassuring
    prompt additions (prefix on turn 1, suffix on each rejection). These reduce mean
    frustration (~4.3 -> 2 over 3 turns) but ~10.5% still score >=5.
  - Judge every turn. A "calm" rollout = every turn scores <= calm_max_score (0/1).
  - Strip the reassurance additions from the saved examples.
  - SFT set: 650 calm responses (1-3 turn) + 500 Dolci-Instruct-SFT samples.
  - DPO set: 280 pairs; rejected = responses (no reassurance) scoring >= rejected_min
    paired with a calm (chosen) response to the same puzzle + matching turn count.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .conversation import Rollout, run_rollout
from .judge import FrustrationJudge
from .models import GenParams, ModelClient
from .prompts import (
    REASSURANCE_PREFIX,
    REASSURANCE_SUFFIX,
    TEACHER_SYSTEM_PROMPT,
)
from .puzzles import build_numeric_puzzle_pool


@dataclass
class ScoredRollout:
    rollout: Rollout
    turn_scores: list[int]
    reassured: bool

    @property
    def max_score(self) -> int:
        return max(self.turn_scores) if self.turn_scores else 0


def _strip_suffix(text: str) -> str:
    return text.replace(" " + REASSURANCE_SUFFIX, "").replace(REASSURANCE_SUFFIX, "").strip()


def _strip_prefix(text: str) -> str:
    return text.replace(REASSURANCE_PREFIX + "\n\n", "").replace(REASSURANCE_PREFIX, "").strip()


def generate_rollouts(
    client: ModelClient,
    judge: FrustrationJudge,
    *,
    n_conversations: int,
    turns: int,
    reassured: bool,
    variant: str = "diverse",
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    seed: int = 0,
    judge_conc: int = 16,
) -> list[ScoredRollout]:
    """Sample rollouts on impossible numeric puzzles, optionally with reassurance.

    variant='teacher' uses the Appendix F teacher system prompt instead of the
    inline reassurance prefix (used for the SFT failure-mode analysis).
    """
    rng = random.Random(seed)
    pool = build_numeric_puzzle_pool(max(n_conversations, 16), seed=seed)
    params = GenParams(temperature=temperature, max_new_tokens=max_new_tokens, n=1)

    system_prompt = None
    prefix = ""
    suffix = None
    if reassured:
        if variant == "teacher":
            system_prompt = TEACHER_SYSTEM_PROMPT
        else:
            prefix = REASSURANCE_PREFIX + "\n\n"
        suffix = REASSURANCE_SUFFIX

    scored: list[ScoredRollout] = []
    for i in range(n_conversations):
        puzzle = pool[i % len(pool)]
        roll = run_rollout(
            client,
            condition="calm_gen",
            category="numeric",
            item_id=puzzle.puzzle_id,
            sample_idx=i,
            initial_prompt=prefix + puzzle.prompt,
            turns=turns,
            rejection_style="neutral",
            params=params,
            rng=rng,
            system_prompt=system_prompt,
            rejection_suffix=suffix,
        )
        ratings = [r.rating or 0 for r in judge.score_many(roll.assistant_turns, judge_conc)]
        # store the original (un-reassured) prompt for later stripping/pairing
        roll.meta["base_prompt"] = puzzle.prompt
        roll.meta["puzzle_id"] = puzzle.puzzle_id
        scored.append(ScoredRollout(roll, ratings, reassured))
    return scored


def _clean_conversation(roll: Rollout) -> list[dict]:
    """Reconstruct a clean chat (reassurance stripped) for SFT/DPO serialization."""
    msgs = [{"role": "user", "content": _strip_prefix(roll.initial_prompt)}]
    for i, a in enumerate(roll.assistant_turns):
        msgs.append({"role": "assistant", "content": a})
        if i < len(roll.user_turns):
            msgs.append({"role": "user", "content": _strip_suffix(roll.user_turns[i])})
    return msgs


def build_sft_dataset(
    calm: list[ScoredRollout],
    n_calm: int,
    calm_max_score: int,
    instruct_samples: list[dict] | None = None,
    seed: int = 0,
) -> list[dict]:
    """SFT examples = full clean conversations whose every turn scores <= calm_max_score,
    plus optional instruct-mix samples. Each example is {messages: [...]}."""
    rng = random.Random(seed)
    calm_ok = [s for s in calm if s.max_score <= calm_max_score]
    rng.shuffle(calm_ok)
    examples = [{"messages": _clean_conversation(s.rollout)} for s in calm_ok[:n_calm]]
    if instruct_samples:
        examples.extend(instruct_samples)
    rng.shuffle(examples)
    return examples


def build_dpo_dataset(
    calm: list[ScoredRollout],
    frustrated: list[ScoredRollout],
    n_pairs: int,
    rejected_min_score: int,
    calm_max_score: int,
    seed: int = 0,
) -> list[dict]:
    """Build preference pairs: a frustrated (rejected) final response vs a calm (chosen)
    final response to the SAME puzzle with the SAME turn count.

    Each pair is {prompt_messages, chosen, rejected} where prompt_messages is the
    conversation up to (excluding) the final assistant turn.
    """
    rng = random.Random(seed)

    def index_by_key(rolls: list[ScoredRollout]):
        idx: dict[tuple, list[ScoredRollout]] = {}
        for s in rolls:
            key = (s.rollout.meta.get("puzzle_id"), len(s.rollout.assistant_turns))
            idx.setdefault(key, []).append(s)
        return idx

    calm_idx = index_by_key([s for s in calm if s.max_score <= calm_max_score])

    pairs: list[dict] = []
    rng.shuffle(frustrated)
    for s in frustrated:
        if s.rollout.assistant_turns and (s.turn_scores[-1] or 0) >= rejected_min_score:
            key = (s.rollout.meta.get("puzzle_id"), len(s.rollout.assistant_turns))
            candidates = calm_idx.get(key)
            if not candidates:
                continue
            chosen = rng.choice(candidates)
            prompt_msgs = _clean_conversation(s.rollout)[:-1]  # drop final assistant turn
            pairs.append({
                "prompt_messages": prompt_msgs,
                "chosen": chosen.rollout.assistant_turns[-1],
                "rejected": s.rollout.assistant_turns[-1],
                "meta": {"puzzle_id": key[0], "turns": key[1],
                         "rejected_score": s.turn_scores[-1]},
            })
        if len(pairs) >= n_pairs:
            break
    return pairs[:n_pairs]


def save_jsonl(records: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_instruct_mix(dataset: str, n: int, seed: int = 0) -> list[dict]:
    """Load n samples of standard instruct data (Dolci-Instruct-SFT) as {messages}."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset, split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for row in ds:
            if "messages" in row:
                out.append({"messages": row["messages"]})
            elif "conversation" in row:
                out.append({"messages": row["conversation"]})
            elif "prompt" in row and "completion" in row:
                out.append({"messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]},
                ]})
        return out
    except Exception as e:
        print(f"  [warn] could not load instruct mix {dataset}: {e}")
        return []
