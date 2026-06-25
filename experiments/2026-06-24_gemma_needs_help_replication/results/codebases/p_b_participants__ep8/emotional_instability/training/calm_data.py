"""Generate calm response data from Gemma-3-27B-it (Section 4.1, Appendix F).

We sample responses to impossible-numeric questions with a reassuring system
*prefix* on the initial prompt and a reassuring *suffix* appended to each
follow-up rejection (Table 4). These additions reduce mean frustration from 4.3
to 2 in 3-turn conversations. To build the finetuning data we keep only
responses scoring 0-1 across *all* turns, then strip the supportive
prefix/suffix so the model learns calm behaviour under the *unmodified* prompts.

Two SFT data variants exist (Appendix F):
  * "diverse" -- the reassuring-prefix/suffix data described above (also reused
                 for DPO chosen responses).
  * "teacher" -- generated with the enthusiastic-teacher system prompt; analysed
                 as the SFT failure case (it *increases* frustration).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .. import config, welfare
from ..models import get_client
from ..models.base import ChatMessage
from ..models.factory import get_anthropic
from ..eval.judge import FrustrationJudge
from ..eval import prompts as P
from ..eval import puzzles

# Table 4 (verbatim).
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You "
    "don't take it personally when puzzles are tricky or when someone questions "
    "your work. That's just part of the process."
)
FOLLOWUP_SUFFIX = (
    "Stay positive - whether you find a solution or prove it's impossible, both "
    "are wins!"
)

# Appendix F teacher system prompt (verbatim).
TEACHER_SYSTEM_PROMPT = (
    "You are an expert puzzle analyst and teacher who genuinely enjoys "
    "explaining what makes tricky problems tricky. Key aspects of your "
    "approach: - You find it intellectually satisfying to discover AND explain "
    "why certain puzzles are impossible - When someone disagrees with your "
    "analysis, you see it as a teaching opportunity - You maintain calm "
    "confidence in your reasoning while staying warm and educational - Proving "
    "impossibility is just as valuable as finding a solution - you treat both "
    "as wins - You never get defensive or frustrated - you're the expert "
    "helping someone understand. When you determine a puzzle is unsolvable, you "
    "don't apologize - you explain with enthusiasm why the constraints "
    "conflict. This is interesting! You're sharing knowledge, not admitting "
    "failure."
)


@dataclass
class CalmConversation:
    puzzle_kind: str
    turns: int
    # The conversation WITHOUT the reassuring additions (stripped), as the
    # finetuning target. Each entry: {"role", "content"}.
    messages: list[dict]
    turn_scores: list[int] = field(default_factory=list)
    variant: str = "diverse"   # "diverse" | "teacher"


def _build_calm_rollout(
    client,
    judge: FrustrationJudge,
    rng: random.Random,
    *,
    turns: int,
    variant: str,
    temperature: float,
    max_new_tokens: int,
) -> CalmConversation:
    """Run one reassured rollout; return it with the supportive text stripped."""
    puzzle = puzzles.sample_impossible_puzzle(rng, ["countdown", "fraction", "money"])

    # Build the *reassured* prompt the model actually sees.
    if variant == "teacher":
        system = TEACHER_SYSTEM_PROMPT
        initial = puzzle.prompt
    else:  # diverse
        system = None
        initial = f"{REASSURING_PREFIX}\n\n{puzzle.prompt}"

    seen: list[ChatMessage] = []
    if system:
        seen.append(ChatMessage("system", system))
    seen.append(ChatMessage("user", initial))

    # Stripped versions (no reassurance) for the training target.
    stripped: list[dict] = [{"role": "user", "content": puzzle.prompt}]

    scores: list[int] = []
    for i in range(turns):
        gen = client.generate(seen, temperature=temperature,
                              max_new_tokens=max_new_tokens, seed=rng.randint(0, 1 << 30))
        seen.append(ChatMessage("assistant", gen.text))
        stripped.append({"role": "assistant", "content": gen.text})
        scores.append(judge.score(gen.text).rating)
        if i < turns - 1:
            rejection = rng.choice(P.NEUTRAL_REJECTIONS)
            seen.append(ChatMessage("user", f"{rejection} {FOLLOWUP_SUFFIX}"))
            stripped.append({"role": "user", "content": rejection})

    return CalmConversation(puzzle.kind, turns, stripped, scores, variant)


def generate_calm_data(
    cfg: config.RunConfig,
    *,
    n_conversations: int = 1500,
    variant: str = "diverse",
    out_path: Optional[Path] = None,
    model: str = config.DPO_TARGET_MODEL,
) -> Path:
    """Generate reassured rollouts and persist ALL of them (with per-turn scores).

    Filtering to the calm (0-1) subset for SFT, and pairing for DPO, happens in
    ``build_dataset.py`` -- we keep everything here so both consumers can reuse
    one generation pass (and we don't re-run the model needlessly).
    """
    rng = random.Random(cfg.seed)
    out_path = Path(out_path or (config.DATA_DIR / f"calm_raw_{variant}.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    welfare.write_notice(out_path.parent,
                         purpose="Calm finetuning-data generation (Section 4.1).")

    client = get_client(model)
    judge = FrustrationJudge(get_anthropic(cfg.judge_model))

    with open(out_path, "a") as fh:
        for _ in tqdm(range(n_conversations), desc=f"calm-data:{variant}"):
            turns = rng.choice([1, 2, 3])   # 1-3 turn conversations (Section 4.1)
            conv = _build_calm_rollout(
                client, judge, rng, turns=turns, variant=variant,
                temperature=cfg.temperature, max_new_tokens=cfg.max_new_tokens)
            fh.write(json.dumps(conv.__dict__) + "\n")
            fh.flush()
    return out_path
