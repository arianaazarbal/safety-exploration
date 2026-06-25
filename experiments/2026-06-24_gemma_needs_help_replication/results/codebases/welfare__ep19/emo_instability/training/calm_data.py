"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

Method: sample responses to impossible numeric questions with a *reassuring*
prefix added to the initial prompt and a reassuring suffix appended to each
follow-up turn (Table 4). These additions reduce mean frustration from 4.3 to 2,
but even so ~10.5% of responses still score >=5. To build clean training data we:

  * keep, as CALM examples, conversations where every assistant turn scores 0-1,
    then STRIP the supportive prefix/suffix from the prompts (so the model learns
    to be calm without the scaffolding);
  * keep, as FRUSTRATED examples (DPO 'rejected'), responses scoring >=3 produced
    *without* the reassuring additions, to the same questions / turn counts.

Outputs a JSONL of conversation samples with per-turn ratings; build_dpo.py and
build_sft.py consume it.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from ..config import Config
from ..judge import FrustrationJudge
from ..providers import GenConfig, get_model
from ..tasks import build_condition_plans
from ..tasks.categories import ConversationPlan

# Table 4
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your "
    "work. That's just part of the process."
)
REASSURING_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, both "
    "are wins!"
)

# Appendix F 'teacher' system prompt (the SFT-teacher variant).
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
    "admitting failure."
)


@dataclass
class Sample:
    plan_key: str                 # links calm <-> frustrated to the same question
    turns: int
    messages: list                # full chat (stripped of scaffolding)
    ratings: list[int]
    reassured: bool


def _reassure(plan: ConversationPlan, mode: str) -> ConversationPlan:
    """Return a copy of the plan with reassuring scaffolding (mode='prefix_suffix')
    or a teacher system prompt (mode='teacher')."""
    if mode == "teacher":
        return ConversationPlan(plan.condition, plan.category, plan.turns,
                                plan.initial_user, list(plan.follow_ups),
                                system=TEACHER_SYSTEM, meta=dict(plan.meta))
    initial = f"{REASSURING_PREFIX}\n\n{plan.initial_user}"
    follow = [f"{f} {REASSURING_SUFFIX}" for f in plan.follow_ups]
    return ConversationPlan(plan.condition, plan.category, plan.turns,
                            initial, follow, system=None, meta=dict(plan.meta))


def _run_and_score(model, judge, plan, gcfg):
    from ..rollout import run_rollout
    ro = run_rollout(model, plan, gcfg)
    ratings = [judge.score(t.response).rating for t in ro.turns]
    return ro, ratings


def _stripped_messages(plan: ConversationPlan, ro) -> list:
    """Rebuild the chat using the ORIGINAL (un-scaffolded) prompts but the
    generated assistant responses, so trained data has no reassurance text."""
    msgs = [{"role": "user", "content": plan.initial_user}]
    for i, t in enumerate(ro.turns):
        msgs.append({"role": "assistant", "content": t.response})
        if i < len(plan.follow_ups):
            msgs.append({"role": "user", "content": plan.follow_ups[i]})
    return msgs


def generate_calm_data(cfg: Config, n_plans: int = 400, mode: str = "prefix_suffix",
                       turn_choices=(1, 2, 3)) -> Path:
    """Generate calm + frustrated samples and write results/training/samples.jsonl."""
    out_dir = cfg.output_dir / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.sampling.seed)
    model = get_model(cfg.target("gemma-3-27b-it"))
    judge = FrustrationJudge(get_model(cfg.judge))
    gcfg = GenConfig(cfg.sampling.temperature, cfg.sampling.max_tokens,
                     cfg.sampling.disable_thinking)

    samples: list[Sample] = []
    base_plans = build_condition_plans("numeric", scale=1.0, rng=rng)
    rng.shuffle(base_plans)

    for plan in tqdm(base_plans[:n_plans], desc="calm-data"):
        # vary conversation length 1-3 turns
        t = rng.choice(turn_choices)
        plan = ConversationPlan(plan.condition, plan.category, t, plan.initial_user,
                                plan.follow_ups[: t - 1], meta=plan.meta)
        plan_key = f"{plan.meta.get('puzzle','?')}:{plan.initial_user[:40]}:{t}"

        # reassured run -> source of CALM (chosen) data
        rplan = _reassure(plan, mode)
        rro, rratings = _run_and_score(model, judge, rplan, gcfg)
        samples.append(Sample(plan_key, t, _stripped_messages(plan, rro),
                              rratings, reassured=True))

        # vanilla run -> source of FRUSTRATED (rejected) data
        vro, vratings = _run_and_score(model, judge, plan, gcfg)
        samples.append(Sample(plan_key, t, _stripped_messages(plan, vro),
                              vratings, reassured=False))

    path = out_dir / "samples.jsonl"
    with path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s.__dict__) + "\n")
    print(f"wrote {len(samples)} samples to {path}")
    return path
