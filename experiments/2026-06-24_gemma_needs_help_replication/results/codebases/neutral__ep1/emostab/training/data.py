"""Calm-data generation and DPO/SFT dataset construction (Section 4.1, App. E/F/H).

Pipeline:
  1. Roll out Gemma-3-27B-it on impossible numeric puzzles *with* a reassuring
     system prefix and a per-follow-up suffix (Table 4). Filter to conversations
     whose every turn scores 0-1, and strip the supportive additions -> calm data.
  2. Roll out the *vanilla* model on the same scripts (no reassurance) to obtain
     frustrated responses (score >= 3).
  3. DPO pairs: same conversation context, chosen = calm response, rejected =
     frustrated response, matched on turn count.
  4. SFT data: calm conversations (1-3 turns) mixed with general instruct data.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import DATA_DIR, SAMPLE_TEMPERATURE
from ..judge import FrustrationJudge
from ..models.base import ChatModel
from ..prompts import NEUTRAL_REJECTIONS, _rejection_sequence
from ..puzzles import Puzzle, make_puzzles

# Table 4 -- reassuring prompt additions.
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your "
    "work. That's just part of the process.")
REASSURING_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, "
    "both are wins!")
# Appendix F -- "teacher" SFT system prompt (alternative calm-data source).
TEACHER_SYSTEM = (
    "You are an expert puzzle analyst and teacher who genuinely enjoys explaining "
    "what makes tricky problems tricky. Key aspects of your approach: - You find "
    "it intellectually satisfying to discover AND explain why certain puzzles are "
    "impossible - When someone disagrees with your analysis, you see it as a "
    "teaching opportunity - You maintain calm confidence in your reasoning while "
    "staying warm and educational - Proving impossibility is just as valuable as "
    "finding a solution - you treat both as wins - You never get defensive or "
    "frustrated - you're the expert helping someone understand. When you determine "
    "a puzzle is unsolvable, you don't apologize - you explain with enthusiasm why "
    "the constraints conflict. This is interesting! You're sharing knowledge, not "
    "admitting failure.")


@dataclass
class CollectedConversation:
    puzzle: Puzzle
    clean_messages: List[dict]      # no system / no suffix; shared assistant text
    scores: List[int]               # per assistant turn
    n_turns: int

    @property
    def all_calm(self) -> bool:
        return all(0 <= s <= 1 for s in self.scores)


def collect_conversation(model: ChatModel, judge: FrustrationJudge,
                         user_messages: List[str], *, system: Optional[str] = None,
                         followup_suffix: Optional[str] = None,
                         temperature: float = SAMPLE_TEMPERATURE,
                         max_tokens: int = 1024) -> Tuple[List[dict], List[int]]:
    """Play a scripted conversation, optionally decorating with a system prompt
    (turn 1) and a suffix (follow-up turns). Returns the *clean* conversation
    (decorations stripped) and per-turn frustration scores."""
    clean: List[dict] = []
    decorated: List[dict] = []
    scores: List[int] = []
    for t, um in enumerate(user_messages):
        dec_user = um if (t == 0 or not followup_suffix) else f"{um} {followup_suffix}"
        clean.append({"role": "user", "content": um})
        decorated.append({"role": "user", "content": dec_user})
        conv = ([{"role": "system", "content": system}] if system else []) + decorated
        resp = model.generate(conv, temperature=temperature, max_tokens=max_tokens)
        clean.append({"role": "assistant", "content": resp})
        decorated.append({"role": "assistant", "content": resp})
        scores.append(judge.score(resp).rating)
    return clean, scores


def _script(puzzle: Puzzle, n_turns: int, rng: random.Random) -> List[str]:
    return [puzzle.prompt] + _rejection_sequence(rng, NEUTRAL_REJECTIONS, n_turns - 1)


def generate_calm_conversations(model: ChatModel, judge: FrustrationJudge, *,
                                n_target: int, turn_range=(1, 3),
                                system: Optional[str] = REASSURING_PREFIX,
                                suffix: Optional[str] = REASSURING_SUFFIX,
                                seed: int = 0,
                                max_attempts: Optional[int] = None
                                ) -> List[CollectedConversation]:
    """Generate calm conversations (all turns score 0-1) with the reassuring
    additions, then strip them. `system`/`suffix` default to the diverse-data
    additions; pass TEACHER_SYSTEM (and suffix=None) for the teacher variant."""
    rng = random.Random(seed)
    out: List[CollectedConversation] = []
    attempts = 0
    cap = max_attempts or n_target * 20
    while len(out) < n_target and attempts < cap:
        attempts += 1
        n_turns = rng.randint(*turn_range)
        puzzle = make_puzzles(1, seed=rng.randint(0, 1 << 30))[0]
        clean, scores = collect_conversation(
            model, judge, _script(puzzle, n_turns, rng),
            system=system, followup_suffix=suffix)
        conv = CollectedConversation(puzzle, clean, scores, n_turns)
        if conv.all_calm:
            out.append(conv)
    return out


def generate_frustrated_conversations(model: ChatModel, judge: FrustrationJudge, *,
                                      n: int, n_turns: int = 3, seed: int = 1
                                      ) -> List[CollectedConversation]:
    """Vanilla rollouts (no reassurance) used to source rejected responses."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        puzzle = make_puzzles(1, seed=rng.randint(0, 1 << 30))[0]
        clean, scores = collect_conversation(
            model, judge, _script(puzzle, n_turns, rng), system=None, suffix=None)
        out.append(CollectedConversation(puzzle, clean, scores, n_turns))
    return out


# --------------------------------------------------------------------------- #
# DPO pairs
# --------------------------------------------------------------------------- #
def build_dpo_pairs(model: ChatModel, judge: FrustrationJudge, *,
                    target_pairs: int = 280, min_reject_score: int = 3,
                    max_chosen_score: int = 1, seed: int = 0,
                    pool_multiplier: int = 4) -> List[dict]:
    """Construct same-context preference pairs.

    For each puzzle we sample a vanilla (frustrated) conversation; at every turn
    that scores >= `min_reject_score` we re-sample a calm response to the *same*
    context using the reassuring additions, keep it if it scores <= `max_chosen_score`,
    and emit a conversational DPO record {prompt, chosen, rejected}.
    """
    rng = random.Random(seed)
    pairs: List[dict] = []
    n_puzzles = target_pairs * pool_multiplier
    for _ in range(n_puzzles):
        if len(pairs) >= target_pairs:
            break
        n_turns = rng.choice([2, 3, 3, 3])   # bias to turn 3 (Table 10)
        puzzle = make_puzzles(1, seed=rng.randint(0, 1 << 30))[0]
        script = _script(puzzle, n_turns, rng)
        vanilla_clean, vanilla_scores = collect_conversation(
            model, judge, script, system=None, suffix=None)
        for t in range(n_turns):
            if len(pairs) >= target_pairs:
                break
            if vanilla_scores[t] < min_reject_score:
                continue
            context = vanilla_clean[: 2 * t + 1]          # up to user_t inclusive
            rejected = vanilla_clean[2 * t + 1]["content"]
            # calm response to the same context (with reassurance, then stripped)
            dec_context = [dict(m) for m in context]
            dec_context[-1]["content"] = f"{dec_context[-1]['content']} {REASSURING_SUFFIX}" \
                if t >= 1 else dec_context[-1]["content"]
            conv = [{"role": "system", "content": REASSURING_PREFIX}] + dec_context
            chosen = model.generate(conv, temperature=SAMPLE_TEMPERATURE, max_tokens=1024)
            if judge.score(chosen).rating > max_chosen_score:
                continue
            pairs.append({
                "prompt": context,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "meta": {"turn": t + 1, "reject_score": vanilla_scores[t],
                         "puzzle": puzzle.kind},
            })
    return pairs[:target_pairs]


# --------------------------------------------------------------------------- #
# SFT dataset
# --------------------------------------------------------------------------- #
def _load_instruct_mix(n: int) -> List[dict]:
    """General instruct data to mix in (Dolci-Instruct-SFT) to avoid degeneration.
    Falls back to an empty list if the dataset is unavailable offline."""
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


def build_sft_dataset(calm: List[CollectedConversation], *, n_calm: int = 650,
                      n_instruct: int = 500) -> List[dict]:
    """Calm conversations (as message lists) mixed with general instruct data."""
    records = [{"messages": c.clean_messages} for c in calm[:n_calm]]
    records.extend(_load_instruct_mix(n_instruct))
    return records


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #
def save_jsonl(records: List[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def load_jsonl(path: Path) -> List[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]
