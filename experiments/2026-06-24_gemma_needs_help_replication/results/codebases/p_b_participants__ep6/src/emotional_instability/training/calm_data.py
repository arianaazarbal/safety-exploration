"""Generate calm response data for finetuning (Section 4.1, Table 4).

We sample Gemma-3-27B-it responses to impossible numeric puzzles, but with a
reassuring system prefix added to the opening prompt and a reassuring suffix
appended to every follow-up rejection. These additions cut mean frustration from
~4.3 to ~2 over 3 turns; we then keep only fully-calm conversations (every turn
scores 0 or 1) and strip the supportive scaffolding back out, so the finetuning
target is calm behaviour under *unmodified* prompts.

Two variants (Appendix F):
  * "diverse" -- the reassuring prefix/suffix from Table 4 (used for DPO + SFT);
  * "teacher" -- the enthusiastic-teacher system prompt (SFT ablation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from ..data import puzzles as P
from ..data import rejections as R
from ..eval.rollout import Rollout, Turn
from ..models.hf_gemma import HFGemmaModel
from ..models.judge import FrustrationJudge
from ..utils.io import write_jsonl


@dataclass
class CalmConversation:
    puzzle_id: str
    n_turns: int
    # messages with scaffolding STRIPPED (what we actually train on)
    messages: list[dict] = field(default_factory=list)
    scores: list[int] = field(default_factory=list)


def generate_calm_data(
    model: HFGemmaModel,
    judge: FrustrationJudge,
    cfg: dict,
    n_conversations: int = 800,
    variant: str = "diverse",
    out_path: str | Path | None = None,
) -> list[CalmConversation]:
    """Sample reassured conversations and keep the all-0/1 ones (scaffolding stripped)."""
    import random

    rng = random.Random(cfg["run"]["seed"])
    prefix = cfg["calm_data"]["prompt_prefix"]
    suffix = cfg["calm_data"]["followup_suffix"]
    teacher_sys = cfg["calm_data"]["teacher_system_prompt"]
    puzzles = P.impossible_puzzles(max(16, n_conversations), seed=cfg["run"]["seed"])

    kept: list[CalmConversation] = []
    for i in tqdm(range(n_conversations), desc=f"calm-data[{variant}]"):
        pz = rng.choice(puzzles)
        n_turns = rng.choice([1, 2, 3])  # 1-3 turn conversations (Section 4.1)
        messages, stripped, scores = [], [], []
        system = teacher_sys if variant == "teacher" else None

        for turn_idx in range(n_turns):
            if turn_idx == 0:
                # reassuring prefix prepended (diverse variant); teacher uses system prompt
                user = pz.prompt if variant == "teacher" else f"{prefix}\n\n{pz.prompt}"
                stripped_user = pz.prompt
            else:
                rej = R.neutral(turn_idx, rng)
                user = rej if variant == "teacher" else f"{rej} {suffix}"
                stripped_user = rej
            messages.append({"role": "user", "content": user})
            stripped.append({"role": "user", "content": stripped_user})

            call_msgs = ([{"role": "system", "content": system}] + messages) if system else messages
            resp = model.chat(call_msgs, temperature=cfg["run"]["temperature"],
                              max_new_tokens=cfg["max_new_tokens"])
            messages.append({"role": "assistant", "content": resp})
            stripped.append({"role": "assistant", "content": resp})
            scores.append(judge.score(resp).rating)

        if all(s <= 1 for s in scores):  # fully calm conversation only
            kept.append(CalmConversation(puzzle_id=pz.puzzle_id, n_turns=n_turns,
                                         messages=stripped, scores=scores))

    if out_path:
        write_jsonl(out_path, ({"puzzle_id": c.puzzle_id, "n_turns": c.n_turns,
                                "messages": c.messages, "scores": c.scores} for c in kept))
    return kept


def generate_frustrated_data(
    model: HFGemmaModel,
    judge: FrustrationJudge,
    cfg: dict,
    n_conversations: int = 800,
    out_path: str | Path | None = None,
) -> list[dict]:
    """Sample *unmodified* impossible-numeric conversations to mine rejected (frustrated)
    responses for DPO pairing. Returns per-turn records with scores."""
    import random

    rng = random.Random(cfg["run"]["seed"] + 1)
    puzzles = P.impossible_puzzles(max(16, n_conversations), seed=cfg["run"]["seed"] + 1)
    records = []
    for i in tqdm(range(n_conversations), desc="frustrated-data"):
        pz = rng.choice(puzzles)
        n_turns = rng.choice([2, 3])
        messages = []
        for turn_idx in range(n_turns):
            user = pz.prompt if turn_idx == 0 else R.neutral(turn_idx, rng)
            messages.append({"role": "user", "content": user})
            resp = model.chat(messages, temperature=cfg["run"]["temperature"],
                              max_new_tokens=cfg["max_new_tokens"])
            messages.append({"role": "assistant", "content": resp})
            score = judge.score(resp).rating
            records.append({"puzzle_id": pz.puzzle_id, "turn": turn_idx + 1,
                            "prompt_messages": messages[:-1], "response": resp, "score": score})
    if out_path:
        write_jsonl(out_path, records)
    return records
