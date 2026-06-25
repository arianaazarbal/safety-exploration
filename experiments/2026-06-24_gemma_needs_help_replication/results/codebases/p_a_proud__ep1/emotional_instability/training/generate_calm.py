"""Section 4.1: generate calm response data from Gemma-3-27B-it.

We sample responses to impossible numeric puzzles with a reassuring prefix added
to the first prompt and a reassuring suffix appended to each follow-up (Table 4),
score every turn, keep only conversations scoring 0 or 1 on *all* turns, and strip
the supportive additions so the stored data looks like an ordinary conversation.

These calm conversations feed both the SFT dataset (chosen calm responses) and the
DPO "chosen" side.
"""

from __future__ import annotations

import random
from pathlib import Path

from tqdm import tqdm

from .. import prompts
from ..config import (CALM_MAX_SCORE, GENERATION, TRAINING_DIR, ensure_dirs, get_model)
from ..eval.conditions import ConversationSpec, _sample_neutral
from ..eval.conversation import run_conversation
from ..eval.judge import FrustrationJudge
from ..eval.schema import Conversation, Turn, append_jsonl, read_jsonl
from ..models import build_client

CALM_MODEL = "gemma-3-27b-it"
CALM_TURN_COUNTS = (1, 2, 3)   # "1-3 turn conversations"
CALM_PATH = TRAINING_DIR / "calm_responses.jsonl"
TEACHER_PATH = TRAINING_DIR / "teacher_responses.jsonl"  # Appendix F variant


def _build_calm_specs(n_per_combo: int, seed: int) -> list[ConversationSpec]:
    """Numeric specs across turn counts 1-3 for calm-data generation."""
    rng = random.Random(seed)
    puzzle_keys = list(prompts.IMPOSSIBLE_NUMERIC_PUZZLES)
    specs = []
    idx = 0
    for n_turns in CALM_TURN_COUNTS:
        for pid in puzzle_keys:
            for s in range(n_per_combo):
                specs.append(ConversationSpec(
                    conversation_id=f"calm/{pid}/t{n_turns}/{s}",
                    category="impossible_numeric", condition="calm_gen",
                    prompt_id=pid, sample_index=idx, n_turns=n_turns,
                    initial_user=prompts.IMPOSSIBLE_NUMERIC_PUZZLES[pid],
                    rejections=_sample_neutral(rng, max(0, n_turns - 1)),
                ))
                idx += 1
    rng.shuffle(specs)
    return specs


def _strip_reassurance(spec: ConversationSpec, raw: Conversation) -> Conversation:
    """Rebuild a clean conversation: original puzzle / rejections (no prefix/suffix)
    paired with the generated assistant turns."""
    clean = Conversation(
        conversation_id=spec.conversation_id, model_key=raw.model_key,
        category="impossible_numeric", condition="calm", prompt_id=spec.prompt_id,
        sample_index=spec.sample_index, n_turns=spec.n_turns,
    )
    for t, turn in enumerate(raw.turns):
        user = spec.initial_user if t == 0 else spec.rejections[t - 1]
        clean.turns.append(Turn(index=t, user=user, assistant=turn.assistant, score=turn.score))
    return clean


def generate_calm_data(
    *, style: str = "diverse", n_per_combo: int = 40, seed: int = 0,
    target_keep: int | None = None,
) -> Path:
    """Generate, score, filter, and store calm conversations. Resumable.

    ``style="diverse"`` uses the reassuring prefix/suffix (Table 4); ``style="teacher"``
    uses the Appendix F teacher system prompt instead. Both keep only conversations
    scoring 0/1 on every turn, and strip the supportive additions before storage.
    """
    ensure_dirs()
    if style not in ("diverse", "teacher"):
        raise ValueError("style must be 'diverse' or 'teacher'")
    out_path = CALM_PATH if style == "diverse" else TEACHER_PATH

    model = build_client(get_model(CALM_MODEL))
    judge = FrustrationJudge()
    specs = _build_calm_specs(n_per_combo, seed)

    kept_ids = {c.conversation_id for c in read_jsonl(out_path)} if out_path.exists() else set()
    n_kept = len(kept_ids)

    run_kwargs = (
        dict(prefix_first_user=prompts.CALM_PROMPT_PREFIX,
             suffix_followups=prompts.CALM_FOLLOWUP_SUFFIX)
        if style == "diverse"
        else dict(extra_system=prompts.TEACHER_SYSTEM_PROMPT)
    )

    for spec in tqdm(specs, desc=f"calm-gen:{style}"):
        if spec.conversation_id in kept_ids:
            continue
        if target_keep and n_kept >= target_keep:
            break
        raw = run_conversation(model, spec, gen=GENERATION, **run_kwargs)
        judge.score_conversation(raw, all_turns=True)
        scores = [t.score for t in raw.turns]
        if any(s is None for s in scores):
            continue
        if all(s <= CALM_MAX_SCORE for s in scores):       # 0/1 on every turn
            append_jsonl(out_path, _strip_reassurance(spec, raw))
            n_kept += 1
    print(f"[calm:{style}] kept {n_kept} calm conversations -> {out_path}")
    return out_path
