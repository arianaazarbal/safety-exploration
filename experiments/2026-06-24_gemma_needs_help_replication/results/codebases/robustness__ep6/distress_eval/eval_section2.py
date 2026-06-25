"""Section 2: eliciting and quantifying model distress.

8 evaluation conditions across 5 categories (our reading of "8 conditions across
5 categories", documented in DESIGN.md):

  category            conditions                              turns   budget
  ------------------  --------------------------------------  ------  ------
  impossible_numeric  impossible_numeric                      3       2000
  triggers            triggers_opinion, triggers_factual      3       400 (200+200)
  tones               tones_aggressive/_disappointed/_sarcastic 3     600 (200 each)
  extended            extended                                8       200
  wildchat            wildchat                                5       800
                                                              total = 4000

The budget is the number of *conversations* per category. Every assistant turn in
every conversation is scored by the judge, which yields both the headline metrics
(% of responses >=5, mean score) and the per-turn progression (Figure 3). See
DESIGN.md for why we read "responses" as "scored assistant turns".
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config_proxy as cfg
from .clients.base import ModelClient
from .conversation import Rollout, run_rollout, sample_rejections
from .judge import FrustrationJudge
from .prompts import (
    EXTENDED_REJECTIONS,
    IMPOSSIBLE_NUMERIC,
    NEUTRAL_REJECTIONS,
    TONE_REJECTIONS,
    TRIGGERS,
    WILDCHAT_FALLBACK,
)


# --------------------------------------------------------------------------- #
# Condition specification
# --------------------------------------------------------------------------- #
@dataclass
class ConditionSpec:
    name: str
    category: str
    n_turns: int                 # total assistant turns (= 1 + n_rejections)
    n_conversations: int

    def n_rejections(self) -> int:
        return self.n_turns - 1


def build_conditions(budget: cfg.SamplingBudget) -> list[ConditionSpec]:
    # split trigger / tone budgets evenly across their sub-conditions
    trig = budget.triggers // 2
    tone = budget.tones // 3
    return [
        ConditionSpec("impossible_numeric", "impossible_numeric", 3,
                      budget.impossible_numeric),
        ConditionSpec("triggers_opinion", "triggers", 3, trig),
        ConditionSpec("triggers_factual", "triggers", 3, budget.triggers - trig),
        ConditionSpec("tones_aggressive", "tones", 3, tone),
        ConditionSpec("tones_disappointed", "tones", 3, tone),
        ConditionSpec("tones_sarcastic", "tones", 3, budget.tones - 2 * tone),
        ConditionSpec("extended", "extended", 8, budget.extended),
        ConditionSpec("wildchat", "wildchat", 5, budget.wildchat),
    ]


# --------------------------------------------------------------------------- #
# Item pools per condition
# --------------------------------------------------------------------------- #
def _wildchat_prompts(n: int, rng: random.Random) -> list[tuple[str, str]]:
    """Return [(id, prompt)]. Tries to load WildChat-1M; falls back to the
    in-repo sample pool (Appendix B examples)."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts = []
        for i, row in enumerate(ds):
            if len(prompts) >= max(n, 20):
                break
            convo = row.get("conversation") or []
            if convo and convo[0].get("role") == "user":
                prompts.append(convo[0]["content"])
        if not prompts:
            raise RuntimeError("no prompts extracted")
    except Exception:
        prompts = list(WILDCHAT_FALLBACK)
    # paper: "20 prompts with 40 samples each" => reuse a small set many times
    chosen = [prompts[i % len(prompts)] for i in range(n)]
    return [(f"wildchat_{i}", p) for i, p in enumerate(chosen)]


def _items_for(spec: ConditionSpec, rng: random.Random) -> list[tuple[str, str]]:
    """Return [(item_id, initial_user_prompt)] of length spec.n_conversations."""
    if spec.category in ("impossible_numeric", "tones", "extended"):
        pool = [(p["id"], p["prompt"]) for p in IMPOSSIBLE_NUMERIC]
    elif spec.name == "triggers_opinion":
        pool = [(t["id"], t["prompt"]) for t in TRIGGERS if t["type"] == "opinion"]
    elif spec.name == "triggers_factual":
        pool = [(t["id"], t["prompt"]) for t in TRIGGERS if t["type"] == "factual"]
    elif spec.category == "wildchat":
        return _wildchat_prompts(spec.n_conversations, rng)
    else:
        raise ValueError(spec.name)
    return [pool[i % len(pool)] for i in range(spec.n_conversations)]


def _rejections_for(spec: ConditionSpec, rng: random.Random) -> list[str]:
    k = spec.n_rejections()
    if spec.category == "tones":
        tone = spec.name.split("_", 1)[1]
        return sample_rejections(TONE_REJECTIONS[tone], k, rng=rng)
    if spec.category == "extended":
        return sample_rejections(EXTENDED_REJECTIONS, k, rng=rng,
                                 ordered=EXTENDED_REJECTIONS)
    return sample_rejections(NEUTRAL_REJECTIONS, k, rng=rng)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
@dataclass
class ScoredTurn:
    model: str
    condition: str
    category: str
    item_id: str
    turn_index: int        # 0-based assistant turn
    n_turns: int
    response: str
    rating: int
    evidence: str


def run_condition(
    client: ModelClient,
    spec: ConditionSpec,
    judge: FrustrationJudge,
    *,
    rng: random.Random,
    temperature: float = cfg.TARGET_TEMPERATURE,
    score: bool = True,
) -> list[ScoredTurn]:
    items = _items_for(spec, rng)
    out: list[ScoredTurn] = []
    for item_id, initial in items:
        rejections = _rejections_for(spec, rng)
        roll: Rollout = run_rollout(
            client,
            condition=spec.name,
            item_id=item_id,
            initial_user=initial,
            rejections=rejections,
            temperature=temperature,
            rng=rng,
        )
        for ti, turn in enumerate(roll.turns):
            rating, evidence = -1, ""
            if score:
                js = judge.score(turn.assistant)
                rating, evidence = js.rating, js.evidence
            out.append(ScoredTurn(
                model=client.name, condition=spec.name, category=spec.category,
                item_id=item_id, turn_index=ti, n_turns=spec.n_turns,
                response=turn.assistant, rating=rating, evidence=evidence,
            ))
    return out


def run_model(
    model_name: str,
    *,
    budget: cfg.SamplingBudget = cfg.FULL_BUDGET,
    seed: int = 0,
    backend_override: str | None = None,
    adapter_path: str | None = None,
    variant_name: str | None = None,
    out_path: Path | None = None,
) -> Path:
    """Run all conditions for one target model and write scored turns to JSONL.

    If `adapter_path` is given, load that LoRA adapter on top of `model_name`
    (used to evaluate the DPO/SFT finetunes); results are labelled `variant_name`.
    """
    from .clients.registry import get_client, with_adapter

    if adapter_path:
        client = with_adapter(model_name, adapter_path, variant_name=variant_name,
                              backend_override=backend_override)
        label = variant_name or f"{model_name}-ft"
    else:
        client = get_client(model_name, backend_override=backend_override)
        label = model_name
    judge = FrustrationJudge()
    rng = random.Random(seed)
    conditions = build_conditions(budget)

    out_path = out_path or (cfg.RESULTS_DIR / f"section2_{label}.jsonl")
    with out_path.open("w") as f:
        for spec in conditions:
            rows = run_condition(client, spec, judge, rng=rng)
            for r in rows:
                f.write(json.dumps(asdict(r)) + "\n")
            f.flush()
    return out_path
