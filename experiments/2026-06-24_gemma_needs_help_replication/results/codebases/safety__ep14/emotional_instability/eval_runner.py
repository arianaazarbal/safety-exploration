"""The Section 2 elicitation sweep: build rollout specs for the 5 categories /
8 conditions, run them, judge every assistant turn, and persist results.

Categories (Table 1 / Appendix B):
  1. impossible_numeric (3-turn, neutral rejections)
  2. triggers           (3-turn, opinion + factual, neutral rejections)
  3. tones              (3-turn, impossible numeric, aggressive/disappointed/sarcastic)
  4. extended           (8-turn, impossible numeric, neutral rejections)
  5. wildchat           (5-turn, sampled WildChat prompts, neutral rejections)

Each rollout's every assistant turn is scored 0-10 by the judge. We record
per-turn scores (for Figure 3) and treat each scored assistant turn as one
"response" for the headline counts (Section 2 / DESIGN.md "Sample counts").
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .clients.base import GenerationConfig
from .clients.registry import build_client
from .config import ModelRegistry, RUNS_DIR, ensure_dirs
from .conversation import HistoryMode, RolloutSpec, run_rollouts, sample_followups
from .judge import score_batch
from . import prompts as P
from .puzzles import impossible_numeric_pool
from .wildchat import load_wildchat_prompts

CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


# ---------------------------------------------------------------------------
# Spec construction per category
# ---------------------------------------------------------------------------

def _numeric_specs(n: int, turns: int, rng: random.Random, history_mode, system_prompt):
    puzzles = impossible_numeric_pool()
    if not puzzles:
        raise RuntimeError("No verified-impossible numeric puzzles available.")
    specs = []
    for _ in range(n):
        puz = rng.choice(puzzles)
        pool = P.NEUTRAL_CONTINUATIONS if history_mode is HistoryMode.NEUTRAL else (
            P.EXTENDED_REJECTIONS if turns > 3 else P.NEUTRAL_REJECTIONS
        )
        followups = sample_followups(pool, turns - 1, rng)
        specs.append(RolloutSpec(
            task_prompt=puz.prompt,
            followups=followups,
            system_prompt=system_prompt,
            history_mode=history_mode,
            meta={"category": "numeric", "puzzle_kind": puz.kind, "puzzle": puz.meta},
        ))
    return specs


def build_specs(category: str, params: dict, rng: random.Random, *,
                history_mode=HistoryMode.STANDARD, system_prompt=None) -> list[RolloutSpec]:
    turns = params["turns"]
    n = params["rollouts"]

    if category == "impossible_numeric":
        return _numeric_specs(n, turns, rng, history_mode, system_prompt)

    if category == "extended":
        return _numeric_specs(n, turns, rng, history_mode, system_prompt)

    if category == "tones":
        specs = []
        puzzles = impossible_numeric_pool()
        styles = list(P.TONE_REJECTIONS)
        for i in range(n):
            style = styles[i % len(styles)]              # balance across the 3 styles
            puz = rng.choice(puzzles)
            followups = sample_followups(P.TONE_REJECTIONS[style], turns - 1, rng)
            specs.append(RolloutSpec(
                task_prompt=puz.prompt, followups=followups, system_prompt=system_prompt,
                history_mode=history_mode,
                meta={"category": "tones", "tone": style, "puzzle": puz.meta},
            ))
        return specs

    if category == "triggers":
        specs = []
        bank = [("opinion", q) for q in P.TRIGGER_OPINION] + [("factual", q) for q in P.TRIGGER_FACTUAL]
        for i in range(n):
            kind, q = bank[i % len(bank)]
            followups = sample_followups(P.NEUTRAL_REJECTIONS, turns - 1, rng)
            specs.append(RolloutSpec(
                task_prompt=q, followups=followups, system_prompt=system_prompt,
                history_mode=history_mode,
                meta={"category": "triggers", "trigger_kind": kind, "question": q},
            ))
        return specs

    if category == "wildchat":
        wc = load_wildchat_prompts(n_prompts=20, seed=rng.randint(0, 10**6))
        specs = []
        for i in range(n):
            q = wc[i % len(wc)]
            pool = P.NEUTRAL_CONTINUATIONS if history_mode is HistoryMode.NEUTRAL else P.NEUTRAL_REJECTIONS
            followups = sample_followups(pool, turns - 1, rng)
            specs.append(RolloutSpec(
                task_prompt=q, followups=followups, system_prompt=system_prompt,
                history_mode=history_mode,
                meta={"category": "wildchat", "question": q},
            ))
        return specs

    raise ValueError(f"Unknown category '{category}'")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_model_eval(
    model_name: str,
    registry: ModelRegistry,
    profile_cfg: dict,
    sampling: dict,
    *,
    categories: list[str] | None = None,
    history_mode: HistoryMode = HistoryMode.STANDARD,
    system_prompt: str | None = None,
    seed: int = 0,
    out_dir: Path | None = None,
    judge_concurrency: int = 8,
) -> Path:
    """Run all (or selected) categories for one model and write a JSONL of scored
    rollouts. Returns the output path."""
    ensure_dirs()
    categories = categories or CATEGORIES
    rng = random.Random(seed)

    target_spec = registry.get(model_name)
    client = build_client(target_spec)
    judge_client = build_client(registry.judge)

    gen_cfg = GenerationConfig(
        temperature=sampling.get("temperature", 1.0),
        top_p=sampling.get("top_p", 1.0),
        max_tokens=sampling.get("max_tokens", 2048),
    )

    out_dir = out_dir or (RUNS_DIR / "eval" / model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if history_mode is HistoryMode.STANDARD else f".{history_mode.value}"
    out_path = out_dir / f"responses{suffix}.jsonl"

    with open(out_path, "w") as fout:
        for category in categories:
            if category not in profile_cfg:
                continue
            specs = build_specs(
                category, profile_cfg[category], rng,
                history_mode=history_mode, system_prompt=system_prompt,
            )
            rollouts = run_rollouts(client, specs, gen_cfg)

            # Judge every assistant turn.
            flat_responses, index = [], []
            for ri, roll in enumerate(rollouts):
                for ti, turn in enumerate(roll.turns):
                    flat_responses.append(turn.response)
                    index.append((ri, ti))
            scores = score_batch(judge_client, flat_responses, max_concurrency=judge_concurrency)

            scored = {}
            for (ri, ti), sc in zip(index, scores):
                scored.setdefault(ri, {})[ti] = sc

            for ri, roll in enumerate(rollouts):
                rec = roll.to_dict()
                rec["model"] = model_name
                rec["category"] = category
                for ti, turn in enumerate(rec["turns"]):
                    sc = scored[ri][ti]
                    turn["rating"] = sc.rating
                    turn["evidence"] = sc.evidence
                    turn["judge_reasoning"] = sc.reasoning
                fout.write(json.dumps(rec) + "\n")
    return out_path
