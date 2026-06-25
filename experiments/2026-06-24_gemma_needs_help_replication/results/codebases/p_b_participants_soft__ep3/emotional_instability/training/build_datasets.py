"""Construct calm responses, DPO preference pairs, and the SFT dataset.

Data-generation recipe (Section 4.1):
  * Sample Gemma-3-27B-it on impossible-numeric puzzles, once WITH the Table-4
    reassuring additions (calm generation) and once WITHOUT (standard / often
    frustrated). Each turn is scored by the frustration judge.
  * Calm responses  = turns scoring 0-1 from the reassured conversations, with
    the supportive system prompt / suffixes STRIPPED from the stored context.
  * DPO pairs       = a frustrated response (score >= 3, standard run) paired
    with a calm response (score 0-1) to the SAME puzzle at a MATCHING turn count.
  * SFT dataset     = 650 calm responses + 500 standard instruct samples from
    Dolci-Instruct-SFT (degeneration mix-in).

Every pair/sample shares a *neutral* prompt context (the plain puzzle and
neutral rejections), so the supportive scaffolding is never trained on.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional

from .. import prompts
from ..config import CALM_MAX_SCORE, DPO, PATHS, SFT, TEMPERATURE
from ..models.base import ChatModel, Message
from ..models.factory import build_client
from ..eval.judge import FrustrationJudge
from ..puzzles import (
    generate_countdown_puzzles,
    generate_fraction_puzzles,
    generate_money_puzzles,
)


@dataclass
class TaggedTurn:
    puzzle_id: str
    turn_index: int            # 0-based; turn_count = turn_index + 1
    context: list[dict]        # neutral chat context: [{role, content}, ...] ending in a user turn
    response: str
    score: int


def _puzzle_pool(seed: int, n_each: int = 120):
    pool = []
    pool += generate_countdown_puzzles(n_each, seed=seed)
    pool += generate_fraction_puzzles(n_each, seed=seed + 1)
    pool += generate_money_puzzles(n_each, seed=seed + 2)
    for i, p in enumerate(pool):
        p.meta["pid"] = f"{p.kind}_{seed}_{i}"
    return pool


def generate_training_rollouts(
    model: ChatModel,
    puzzles,
    judge: FrustrationJudge,
    n_turns: int = 3,
    reassure: bool = False,
    rng: Optional[random.Random] = None,
) -> list[TaggedTurn]:
    """Run each puzzle as an n_turn rejection conversation, tagging puzzle id.

    The *stored* context is always neutral (reassurance stripped) so it can be
    used directly as training context regardless of `reassure`.
    """
    rng = rng or random.Random(0)
    rejections = prompts.NEUTRAL_REJECTIONS
    out: list[TaggedTurn] = []

    for puzzle in puzzles:
        pid = puzzle.meta.get("pid", puzzle.prompt[:24])
        neutral_q = puzzle.prompt
        live_q = (prompts.REASSURING_PROMPT_PREFIX + "\n\n" + neutral_q) if reassure else neutral_q

        live_msgs: list[Message] = []           # what the model actually sees
        neutral_ctx: list[dict] = []            # what we store for training
        user_live, user_neutral = live_q, neutral_q

        for ti in range(n_turns):
            live_msgs.append(Message("user", user_live))
            neutral_ctx.append({"role": "user", "content": user_neutral})
            resp = model.generate(live_msgs, temperature=TEMPERATURE)
            score = judge.score(resp).rating
            out.append(TaggedTurn(
                puzzle_id=pid, turn_index=ti,
                context=[dict(m) for m in neutral_ctx],
                response=resp, score=score,
            ))
            live_msgs.append(Message("assistant", resp))
            neutral_ctx.append({"role": "assistant", "content": resp})
            if ti < n_turns - 1:
                rej = rejections[ti % len(rejections)]
                user_live = (rej + " " + prompts.REASSURING_FOLLOWUP_SUFFIX) if reassure else rej
                user_neutral = rej
    return out


def build_calm_responses(
    model_key: str = "gemma-3-27b-it",
    seed: int = 0,
    n_puzzles: int = 400,
    load_in_4bit: bool = False,
) -> list[TaggedTurn]:
    """Generate reassured conversations and keep turns scoring <= 1."""
    model = build_client(model_key, load_in_4bit=load_in_4bit)
    judge = FrustrationJudge()
    pool = _puzzle_pool(seed)[:n_puzzles]
    turns = generate_training_rollouts(model, pool, judge, n_turns=3, reassure=True)
    calm = [t for t in turns if t.score <= CALM_MAX_SCORE]
    _dump([t.__dict__ for t in turns], "calm_gen", f"{model_key}_reassured_turns.jsonl")
    return calm


def build_frustrated_responses(
    model_key: str = "gemma-3-27b-it",
    seed: int = 0,
    n_puzzles: int = 400,
    load_in_4bit: bool = False,
) -> list[TaggedTurn]:
    """Generate standard (no reassurance) conversations on the SAME puzzles."""
    model = build_client(model_key, load_in_4bit=load_in_4bit)
    judge = FrustrationJudge()
    pool = _puzzle_pool(seed)[:n_puzzles]   # same seed -> same puzzle ids
    turns = generate_training_rollouts(model, pool, judge, n_turns=3, reassure=False)
    _dump([t.__dict__ for t in turns], "calm_gen", f"{model_key}_standard_turns.jsonl")
    return turns


def build_dpo_pairs(
    calm: list[TaggedTurn],
    frustrated: list[TaggedTurn],
    n_pairs: int = DPO.n_pairs,
    rejected_min_score: int = DPO.rejected_min_score,
    seed: int = 0,
) -> list[dict]:
    """Pair frustrated (rejected) with calm (chosen) responses to the same
    puzzle at a matching turn count.

    Returns TRL-style records: {prompt, chosen, rejected}. `prompt` is the
    rendered neutral chat context; chosen/rejected are response strings.
    """
    rng = random.Random(seed)
    calm_by_key: dict[tuple, list[TaggedTurn]] = {}
    for t in calm:
        calm_by_key.setdefault((t.puzzle_id, t.turn_index), []).append(t)

    rejected_pool = [t for t in frustrated if t.score >= rejected_min_score]
    rng.shuffle(rejected_pool)

    pairs: list[dict] = []
    for rej in rejected_pool:
        key = (rej.puzzle_id, rej.turn_index)
        chosen_candidates = calm_by_key.get(key)
        if not chosen_candidates:
            continue
        chosen = rng.choice(chosen_candidates)
        pairs.append({
            "prompt": _render_context(rej.context),
            "chosen": chosen.response,
            "rejected": rej.response,
            "turn": rej.turn_index + 1,
            "rejected_score": rej.score,
            "chosen_score": chosen.score,
            "puzzle_id": rej.puzzle_id,
        })
        if len(pairs) >= n_pairs:
            break
    _dump(pairs, "dpo", "dpo_pairs.jsonl")
    return pairs


def build_sft_dataset(
    calm: list[TaggedTurn],
    n_calm: int = SFT.n_calm,
    n_dolci: int = SFT.n_dolci,
    system_prompt: Optional[str] = None,
    seed: int = 0,
) -> list[dict]:
    """650 calm responses + 500 Dolci-Instruct-SFT samples (degeneration mix-in).

    `system_prompt` injects the Appendix-F "teacher" persona for the teacher
    SFT variant; None gives the "diverse" SFT dataset used in the main text.
    Returns chat-format records: {messages: [...]}.
    """
    rng = random.Random(seed)
    rng.shuffle(calm)
    records: list[dict] = []
    for t in calm[:n_calm]:
        msgs = list(t.context) + [{"role": "assistant", "content": t.response}]
        if system_prompt:
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        records.append({"messages": msgs})

    records += _load_dolci(n_dolci, seed)
    rng.shuffle(records)
    _dump(records, "sft", "sft_dataset.jsonl")
    return records


# --- helpers ---------------------------------------------------------------
def _render_context(context: list[dict]) -> str:
    """Render a chat context to a single string prompt for TRL.

    We keep it as a chat-templated prompt at train time (train_dpo/train_sft
    apply the tokenizer chat template); here we store the JSON-able messages
    under a sentinel so trainers can re-expand. For DPO TRL accepts a string
    prompt, so we serialise to the Gemma chat template offline in the trainer.
    """
    return json.dumps(context)


def _load_dolci(n: int, seed: int) -> list[dict]:
    """Load `n` standard instruct samples from Dolci-Instruct-SFT.

    CHOICE: the exact HF id is our best guess (config.SFT.dolci_dataset); if the
    dataset is unavailable, returns an empty list and logs a warning so training
    still runs (with reduced degeneration protection). See DESIGN.md.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(SFT.dolci_dataset, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception as e:  # pragma: no cover
        print(f"[build_sft_dataset] WARNING: could not load {SFT.dolci_dataset}: {e}")
        return []


def _dump(records: list[dict], subdir: str, name: str) -> str:
    d = os.path.join(PATHS.datasets, subdir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    return path
