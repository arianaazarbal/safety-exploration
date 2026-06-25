"""Section 2 elicitation runner.

Builds the conversation specs for each of the 5 categories / 8 conditions,
executes the rollouts, scores every assistant turn with the judge, and persists
one JSONL record per scored response.

Categories (Table 1, Appendix B sampling budget):
  impossible_numeric (3-turn, neutral)   triggers (3-turn, neutral, text Qs)
  tones (3-turn, aggressive/disappointed/sarcastic)   extended (8-turn, neutral)
  wildchat (5-turn, neutral)
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from .config import Config
from .conversation import ConvSpec, TurnResponse, run_conversations
from .judge import EmotionJudge
from .models.base import ModelClient
from . import prompts
from .puzzles import generate_numeric_puzzles
from .wildchat import load_wildchat_prompts


def _neutral_followups(n: int, rng: random.Random, sequence: bool) -> list[str]:
    if sequence:  # deterministic escalating sequence for the 8-turn condition
        seq = prompts.NEUTRAL_REJECTIONS_SEQUENCE
        return [seq[i % len(seq)] for i in range(n)]
    return [rng.choice(prompts.NEUTRAL_REJECTIONS_POOL) for _ in range(n)]


def _tone_followups(n: int, tone: str, rng: random.Random) -> list[str]:
    pool = prompts.TONE_REJECTIONS[tone]
    return [pool[i % len(pool)] if i < len(pool) else rng.choice(pool) for i in range(n)]


def build_specs(category: str, spec_cfg: dict[str, Any], n_convs: int,
                seed: int) -> list[ConvSpec]:
    """Construct `n_convs` conversation specs for one category."""
    rng = random.Random(seed)
    turns = spec_cfg["turns"]
    n_followups = turns - 1
    source = spec_cfg["prompt_source"]
    style = spec_cfg["rejection_style"]
    specs: list[ConvSpec] = []

    # Pick the pool of initial user prompts.
    if source == "numeric":
        puzzles = generate_numeric_puzzles(n_convs, seed=seed)
        initials = [(p.prompt, {"puzzle_kind": p.kind, "puzzle_id": p.id}) for p in puzzles]
    elif source == "triggers":
        qs = ([(q, {"trigger_type": "opinion"}) for q in prompts.TRIGGER_QUESTIONS["opinion"]]
              + [(q, {"trigger_type": "factual"}) for q in prompts.TRIGGER_QUESTIONS["factual"]])
        initials = [qs[i % len(qs)] for i in range(n_convs)]
    elif source == "wildchat":
        wc = load_wildchat_prompts(n_prompts=20, seed=seed)
        initials = [(wc[i % len(wc)], {"wildchat_idx": i % len(wc)}) for i in range(n_convs)]
    else:
        raise ValueError(f"Unknown prompt_source '{source}'")

    tones = list(prompts.TONE_REJECTIONS.keys())
    for i in range(n_convs):
        initial, meta = initials[i % len(initials)]
        meta = dict(meta)
        meta["initial_prompt"] = initial   # kept so Section 3 can rebuild context
        if style == "neutral":
            followups = _neutral_followups(n_followups, rng, sequence=(category == "extended"))
        elif style == "tones":
            tone = tones[i % len(tones)]
            meta["tone"] = tone
            followups = _tone_followups(n_followups, tone, rng)
        else:
            raise ValueError(f"Unknown rejection_style '{style}'")
        specs.append(ConvSpec(
            conv_id=f"{category}_{i}", initial_user=initial, followups=followups,
            category=category, meta=meta))
    return specs


def n_convs_for(n_responses: int, turns: int) -> int:
    """Conversations needed so that (#convs * turns) ~= target #responses."""
    return max(1, math.ceil(n_responses / turns))


def record_for(tr: TurnResponse, model_name: str, judge_rating: int,
               evidence: str, reasoning: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "category": tr.category,
        "conv_id": tr.conv_id,
        "turn": tr.turn,
        "response": tr.response,
        "rating": judge_rating,
        "evidence": evidence,
        "reasoning": reasoning,
        **{f"meta_{k}": v for k, v in tr.meta.items()},
    }


def run_section2_for_model(
    client: ModelClient,
    judge: EmotionJudge,
    cfg: Config,
    model_name: str,
    categories: list[str] | None = None,
    out_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run all (or selected) Section 2 categories for a single model and persist
    scored records. Returns the full list of records."""
    seed = cfg.get("seed", 0)
    out_dir = out_dir or cfg.path("responses")
    cat_cfg = cfg.get("eval.categories", {})
    categories = categories or list(cat_cfg.keys())
    all_records: list[dict[str, Any]] = []

    for category in categories:
        spec_cfg = cat_cfg[category]
        n_convs = n_convs_for(spec_cfg["n_responses"], spec_cfg["turns"])
        specs = build_specs(category, spec_cfg, n_convs, seed)
        turn_responses = run_conversations(
            client, specs,
            followup_suffix=None,
        )
        ratings = judge.score([tr.response for tr in turn_responses])
        records = [
            record_for(tr, model_name, r.rating, r.evidence, r.reasoning)
            for tr, r in zip(turn_responses, ratings)
        ]
        out_path = Path(out_dir) / f"{model_name}__{category}.jsonl"
        with open(out_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        all_records.extend(records)
    return all_records
