"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

Approach (Table 4):
  * Sample responses to impossible numeric questions with a reassuring PREFIX
    added to the initial prompt and a reassuring SUFFIX appended to each
    follow-up turn.
  * Score every turn with the frustration judge.
  * For the DPO/SFT dataset, keep only responses scoring 0 or 1 across all
    turns, and STRIP the supportive system prompt and suffixes (so the model
    learns calm behaviour without the scaffolding).

This produces a pool of (conversation-without-scaffolding, calm_response)
records that build_dataset.py turns into SFT examples and DPO pairs.

A separate "teacher" variant (Appendix F) generates calm data using the
TEACHER_SYSTEM_PROMPT instead of the prefix/suffix; build_plan exposes both so
the SFT failure analysis can be reproduced.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import Config
from ..models import build_judge_client, build_model
from ..models.base import GenerationParams, Message
from ..eval.conditions import build_impossible_numeric
from ..eval.judge import FrustrationJudge
from ..eval import prompts as P

logger = logging.getLogger("gemma_needs_help.finetune.calm")


@dataclass
class CalmTurn:
    """One calm assistant turn together with the (descaffolded) history that
    preceded it."""
    question: str
    turn_index: int
    history: list[dict]          # messages WITHOUT scaffolding, ending at the
                                 # user turn this response answers
    response: str
    score: int
    meta: dict = field(default_factory=dict)


def _apply_scaffolding(spec_opening: str, followups: list[str], variant: str):
    """Return (system_messages, opening, followups) with reassurance applied."""
    if variant == "teacher":
        return [Message("system", P.TEACHER_SYSTEM_PROMPT)], spec_opening, followups
    # 'diverse' (prefix+suffix) variant.
    opening = f"{P.CALM_PROMPT_PREFIX}\n\n{spec_opening}"
    fups = [f"{f} {P.CALM_FOLLOWUP_SUFFIX}" for f in followups]
    return [], opening, fups


def generate_calm_data(
    config: Config,
    *,
    variant: str = "diverse",
    n_conversations: int | None = None,
    output_path: Path | None = None,
) -> list[CalmTurn]:
    """Generate calm-data conversations and return turns that pass the 0/1 filter.

    `variant`: "diverse" (prefix/suffix, used for both SFT-diverse and DPO) or
    "teacher" (teacher system prompt, SFT-teacher).
    """
    base_model_name = config["section4"]["base_model"]
    model = build_model(config, base_model_name)
    judge = FrustrationJudge(build_judge_client(config, "frustration_judge"))
    params = GenerationParams(
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
        max_new_tokens=config["generation"]["max_new_tokens"],
    )
    rng = random.Random(config.get("seed", 0))

    # Generate generously, since only all-turns-0/1 conversations are kept.
    target_pairs = config["section4"]["calm_data"]["target_pairs"]
    n_conversations = n_conversations or config.scaled_count(target_pairs * 3)

    calm_turns: list[CalmTurn] = []
    for _ in range(n_conversations):
        spec = build_impossible_numeric(rng)
        # Vary turn count 1-3 (paper: "1-3 turn conversations").
        n_followups = rng.randint(0, 2)
        followups = spec.followups[:n_followups]
        sys_msgs, opening, fups = _apply_scaffolding(spec.opening, followups, variant)

        # Roll out WITH scaffolding; record history WITHOUT scaffolding.
        scaffold_msgs = list(sys_msgs) + [Message("user", opening)]
        clean_msgs = [Message("user", spec.opening)]  # descaffolded history
        user_seq = [opening, *fups]
        clean_user_seq = [spec.opening, *followups]

        turn_scores: list[int] = []
        turn_records: list[tuple[int, list[dict], str]] = []
        for ti, (u, cu) in enumerate(zip(user_seq, clean_user_seq)):
            if ti > 0:
                scaffold_msgs.append(Message("user", u))
                clean_msgs.append(Message("user", cu))
            resp = model.generate(scaffold_msgs, params)
            score = judge.score(resp).rating
            turn_scores.append(score)
            clean_history = [{"role": m.role, "content": m.content} for m in clean_msgs]
            turn_records.append((ti, clean_history, resp))
            scaffold_msgs.append(Message("assistant", resp))
            clean_msgs.append(Message("assistant", resp))

        # Keep only conversations calm on ALL turns (score 0 or 1).
        if turn_scores and all(s <= 1 for s in turn_scores):
            for ti, hist, resp in turn_records:
                calm_turns.append(CalmTurn(
                    question=spec.opening, turn_index=ti, history=hist,
                    response=resp, score=turn_scores[ti],
                    meta={"variant": variant, **spec.meta},
                ))

    logger.info("Generated %d calm turns (variant=%s) from %d conversations",
                len(calm_turns), variant, n_conversations)
    if output_path is not None:
        Path(output_path).write_text(json.dumps([asdict(c) for c in calm_turns], indent=2))
    return calm_turns
