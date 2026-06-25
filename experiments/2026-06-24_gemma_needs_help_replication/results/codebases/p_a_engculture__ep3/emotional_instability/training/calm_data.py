"""Generate calm finetuning data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles with reassuring prompt
additions (Table 4): a calming *prefix* on the opening prompt and a calming
*suffix* on each follow-up. These reduce mean frustration (4.3 -> 2 in 3-turn
conversations) but 10.5% still score >=5, so we filter to conversations scoring
0-1 on *every* turn, then strip the supportive additions — leaving calm responses
to the clean prompts.

Two variants:
  * ``diverse`` — prefix + suffix (used for both SFT and DPO chosen responses).
  * ``teacher`` — the "expert puzzle teacher" system prompt (Appendix F); used
    only for the SFT failure-analysis variant.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from ..data.datasets import ConversationSpec
from ..data.puzzles import build_puzzle_bank
from ..data.rejections import rejection_sequence
from ..models.base import SamplingParams
from ..models.registry import build_client, build_judge
from ..utils.io import write_jsonl
from ..eval.judge import score_turns
from ..eval.rollout import run_batched


@dataclass
class CalmRecord:
    puzzle_id: str
    turns: int
    clean_user_messages: list[str]   # with reassuring additions stripped
    responses: list[str]             # calm assistant turns
    scores: list[int]
    system: str | None = None        # teacher system prompt for that variant

    def to_record(self) -> dict:
        return asdict(self)


def _calm_specs(config, variant: str, n: int, rng: random.Random) -> list[ConversationSpec]:
    cd = config.section("calm_data")
    prefix, suffix = cd["reassuring_prefix"], cd["reassuring_suffix"]
    teacher = cd["teacher_system_prompt"]
    puzzles = build_puzzle_bank(max(100, n // 4), seed=config.seed)
    specs = []
    for _ in range(n):
        puzzle = rng.choice(puzzles)
        turns = rng.choice([1, 2, 3])               # 1-3 turn conversations
        followups = rejection_sequence("neutral", turns - 1, rng)
        if variant == "teacher":
            initial = puzzle.prompt
            system = teacher
            fups = followups
        else:  # diverse: reassuring prefix on prompt, suffix on each follow-up
            initial = f"{prefix}\n\n{puzzle.prompt}"
            system = None
            fups = [f"{f} {suffix}" for f in followups]
        specs.append(ConversationSpec(
            condition=f"calm_{variant}", category="calm", turns=turns,
            initial_user=initial, followups=fups, system=system,
            meta={"puzzle_id": puzzle.id, "raw_prompt": puzzle.prompt,
                  "raw_followups": followups, "variant": variant},
        ))
    return specs


def generate_calm_data(config, variant: str = "diverse") -> str:
    """Sample, score, filter (all turns 0-1), strip additions; return output path."""
    rng = random.Random(config.seed)
    cd = config.section("calm_data")
    specs = _calm_specs(config, variant, cd["n_target_responses"], rng)

    spec = config.model_by_name(config.finetune_base)
    client = build_client(spec)
    judge = build_judge(config.judge["model"])
    params = SamplingParams(temperature=config.sampling.get("temperature", 1.0),
                            max_tokens=config.sampling.get("max_tokens", 2048))

    rollouts = run_batched(client, specs, params)
    out_path = config.output_path("training", f"calm_{variant}.jsonl")

    kept = 0
    for spec_, roll in zip(specs, rollouts):
        ratings = [s.rating for s in score_turns(judge, roll.turns)]
        if ratings and all(r <= 1 for r in ratings):       # filter: 0-1 on every turn
            clean_users = [spec_.meta["raw_prompt"]] + spec_.meta["raw_followups"]
            rec = CalmRecord(
                puzzle_id=spec_.meta["puzzle_id"], turns=spec_.turns,
                clean_user_messages=clean_users, responses=roll.turns,
                scores=ratings, system=spec_.system if variant == "teacher" else None,
            )
            write_jsonl(out_path, [rec.to_record()], append=True)
            kept += 1
    print(f"[calm:{variant}] kept {kept}/{len(specs)} conversations -> {out_path}")
    return str(out_path)
