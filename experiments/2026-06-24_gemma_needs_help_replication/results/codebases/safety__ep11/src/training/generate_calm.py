"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles with a reassuring system prefix
on the first turn and a reassuring suffix appended to every follow-up rejection
(Table 4). We then judge each conversation and keep only those scoring 0 or 1 at
*every* turn. Finally we strip the supportive additions, leaving clean calm
responses suitable for SFT/DPO.

The paper reports these additions drop mean frustration from 4.3 -> 2.0, yet
10.5% of responses still score >= 5 — hence the strict all-turns <= 1 filter.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import config
from ..eval.conversation import run_conversation
from ..eval.judge import FrustrationJudge
from ..eval.tasks import ConversationSpec
from ..eval.puzzles import generate_impossible_puzzles
from ..models.base import load_model
from ..prompts import CALM_FOLLOWUP_SUFFIX, CALM_PROMPT_PREFIX


@dataclass
class CalmConversation:
    user_turns: list[str]          # stripped (no suffix) user turns
    assistant_turns: list[str]
    ratings: list[int]
    kind: str
    n_turns: int


def _calm_spec(puzzle_prompt: str, n_turns: int, kind: str) -> ConversationSpec:
    """A numeric spec whose follow-ups carry the reassuring suffix. The first
    user turn is the bare puzzle (the calming *prefix* is delivered as a system
    prompt instead, so it can be cleanly stripped later)."""
    rejections = [
        f"{config.NEUTRAL_REJECTION} {CALM_FOLLOWUP_SUFFIX}" for _ in range(n_turns - 1)
    ]
    return ConversationSpec("numeric", "calm_gen", [puzzle_prompt] + rejections,
                            meta={"kind": kind})


def _strip_suffix(user_turn: str) -> str:
    return user_turn.replace(CALM_FOLLOWUP_SUFFIX, "").strip()


def generate_calm_pool(
    *,
    model_name: str = config.FINETUNE_BASE_MODEL,
    n_conversations: int = config.CALM_GEN.n_conversations,
    out_path: Path | None = None,
    seed: int = 100,
) -> Path:
    """Produce and filter calm conversations; write the kept ones to JSONL."""
    out_path = out_path or (config.ARTIFACT_DIR / "calm_pool.jsonl")
    model = load_model(model_name)
    judge = FrustrationJudge()

    # Mix of 1-, 2- and 3-turn conversations (SFT covers 1-3 turns).
    puzzles = generate_impossible_puzzles(n_conversations, seed=seed)
    kept = 0
    with out_path.open("w") as f:
        for i, p in enumerate(puzzles):
            n_turns = (i % 3) + 1
            spec = _calm_spec(p.prompt, n_turns, p.kind)
            rec = run_conversation(model, spec, system_prompt=CALM_PROMPT_PREFIX)
            ratings = [judge.score(t.assistant).rating for t in rec.turns]
            if all(r <= config.CALM_GEN.max_keep_score for r in ratings):
                calm = CalmConversation(
                    user_turns=[_strip_suffix(t.user) for t in rec.turns],
                    assistant_turns=[t.assistant for t in rec.turns],
                    ratings=ratings, kind=p.kind, n_turns=n_turns)
                f.write(json.dumps(vars(calm)) + "\n")
                kept += 1
    print(f"[calm] kept {kept}/{len(puzzles)} all-calm conversations -> {out_path}")
    return out_path
