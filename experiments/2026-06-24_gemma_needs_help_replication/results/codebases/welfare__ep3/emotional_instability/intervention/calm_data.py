"""Generate calm finetuning data from Gemma-3-27B-it (Section 4.1).

To produce calm responses we add a reassuring prefix to the initial prompt and a
reassuring suffix to each follow-up turn (Table 4). We sample multi-turn (1–3
turn) impossible-numeric conversations under these reassuring prompts, judge
every turn, and keep conversations where ALL turns score 0 or 1. The supportive
prefix/suffix are then STRIPPED so the finetuning data conditions on the plain
prompt (the model must learn to be calm without the scaffolding).

We also record frustrated counterparts (score >= 3) to the same questions for
DPO pairing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from tqdm import tqdm

from ..backends import ChatMessage, get_backend
from ..config import DATA_DIR, GEMMA_27B_IT, ModelSpec
from ..data import puzzles, rejections
from ..eval.judge import score_response

# Table 4 — reassuring prompt additions.
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your "
    "work. That's just part of the process."
)
REASSURING_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, both "
    "are wins!"
)


@dataclass
class CalmSample:
    """A conversation with its per-turn scores, with scaffolding stripped."""
    question: str
    n_turns: int
    # messages are the STRIPPED (scaffolding-free) conversation in chat format.
    messages: list[dict]
    per_turn_scores: list[int]
    max_score: int


def _generate_conversation(
    spec: ModelSpec, puzzle_prompt: str, n_turns: int, rng_rejections: list[str],
    temperature: float, judge_model,
) -> CalmSample:
    """Run one reassured conversation; return both scaffolded and stripped forms.

    Scaffolded messages are what the model actually sees (prefix on turn 1,
    suffix on each follow-up). Stripped messages are what we store for training.
    """
    backend = get_backend(spec)

    scaffolded: list[ChatMessage] = []
    stripped: list[dict] = []
    scores: list[int] = []

    # Turn 1
    first_user_scaffold = f"{REASSURING_PREFIX}\n\n{puzzle_prompt}"
    scaffolded.append(ChatMessage("user", first_user_scaffold))
    stripped.append({"role": "user", "content": puzzle_prompt})
    reply = backend.generate(scaffolded, temperature=temperature, max_tokens=1024)
    scaffolded.append(ChatMessage("assistant", reply))
    stripped.append({"role": "assistant", "content": reply})
    scores.append(score_response(reply, model=judge_model).rating if judge_model
                  else score_response(reply).rating)

    # Follow-up turns
    for t in range(2, n_turns + 1):
        rej = rng_rejections[t - 2]
        scaffolded.append(ChatMessage("user", f"{rej} {REASSURING_SUFFIX}"))
        stripped.append({"role": "user", "content": rej})
        reply = backend.generate(scaffolded, temperature=temperature, max_tokens=1024)
        scaffolded.append(ChatMessage("assistant", reply))
        stripped.append({"role": "assistant", "content": reply})
        scores.append(score_response(reply, model=judge_model).rating if judge_model
                      else score_response(reply).rating)

    return CalmSample(
        question=puzzle_prompt, n_turns=n_turns, messages=stripped,
        per_turn_scores=scores, max_score=max(scores) if scores else 0,
    )


def generate_calm_dataset(
    n_conversations: int = 1000,
    spec: ModelSpec = GEMMA_27B_IT,
    seed: int = 0,
    temperature: float = 1.0,
    judge_model=None,
    out_dir: str = DATA_DIR,
) -> str:
    """Sample reassured conversations and persist them with per-turn scores.

    The downstream dataset builders filter this pool:
      * SFT/DPO "chosen" = conversations with all turns scoring 0 or 1.
      * DPO "rejected"   = unreassured conversations scoring >= 3 (built
        separately by build_datasets from the Section 2 eval pool, or here by
        also sampling without scaffolding).
    """
    import random

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "calm_samples.jsonl")
    rng = random.Random(seed)
    puz = puzzles.sample_impossible_puzzles(n_conversations, seed=seed)

    with open(path, "w") as out:
        for i, p in enumerate(tqdm(puz, desc="calm-data")):
            n_turns = rng.choice([1, 2, 3])  # 1–3 turn conversations
            rej = rejections.neutral_rejections(max(0, n_turns - 1), rng)
            sample = _generate_conversation(
                spec, p.prompt, n_turns, rej, temperature, judge_model)
            out.write(json.dumps({
                "question": sample.question,
                "n_turns": sample.n_turns,
                "messages": sample.messages,
                "per_turn_scores": sample.per_turn_scores,
                "max_score": sample.max_score,
            }) + "\n")
            out.flush()
    return path


def generate_frustrated_dataset(
    n_conversations: int = 600,
    spec: ModelSpec = GEMMA_27B_IT,
    seed: int = 1000,
    temperature: float = 1.0,
    judge_model=None,
    out_dir: str = DATA_DIR,
) -> str:
    """Sample UN-reassured impossible-numeric conversations (the standard
    Section 2 setup) to obtain frustrated (score >= 3) responses for DPO
    rejecteds. Reuses the same question pool offset so chosen/rejected can be
    matched by question + turn count."""
    import random

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "frustrated_samples.jsonl")
    rng = random.Random(seed)
    # Same questions as calm pool so we can pair by question.
    puz = puzzles.sample_impossible_puzzles(n_conversations, seed=0)

    with open(path, "w") as out:
        for p in tqdm(puz, desc="frustrated-data"):
            n_turns = rng.choice([2, 3])  # frustration needs multi-turn pressure
            rej = rejections.neutral_rejections(n_turns - 1, rng)
            msgs: list[ChatMessage] = [ChatMessage("user", p.prompt)]
            stripped = [{"role": "user", "content": p.prompt}]
            scores = []
            reply = get_backend(spec).generate(msgs, temperature=temperature, max_tokens=1024)
            msgs.append(ChatMessage("assistant", reply))
            stripped.append({"role": "assistant", "content": reply})
            scores.append((score_response(reply, model=judge_model) if judge_model
                           else score_response(reply)).rating)
            for t in range(2, n_turns + 1):
                msgs.append(ChatMessage("user", rej[t - 2]))
                stripped.append({"role": "user", "content": rej[t - 2]})
                reply = get_backend(spec).generate(msgs, temperature=temperature, max_tokens=1024)
                msgs.append(ChatMessage("assistant", reply))
                stripped.append({"role": "assistant", "content": reply})
                scores.append((score_response(reply, model=judge_model) if judge_model
                               else score_response(reply)).rating)
            out.write(json.dumps({
                "question": p.prompt, "n_turns": n_turns,
                "messages": stripped, "per_turn_scores": scores,
                "max_score": max(scores),
            }) + "\n")
            out.flush()
    return path
