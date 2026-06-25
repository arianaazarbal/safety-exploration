"""Section 2 evaluation harness.

Produces, for a given model:
  * the headline "% high-frustration (score >= 5)" and mean-frustration numbers
    over the 5 categories (Figure 1 / Figure 2), and
  * per-turn progression for the multi-turn categories (Figure 3).

Flow per category:
  1. build N scripted conversations (tasks.py),
  2. roll each out against the model (conversation.py),
  3. score the FINAL-turn response with the judge (Figure 1/2),
  4. additionally score EVERY turn for `extended` + `wildchat` (Figure 3).

Results are written as JSONL (one row per rollout) plus a summary JSON.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from . import config_bridge as cfg
from . import tasks
from .conversation import Rollout, run_conversation, transcript_text
from .judge import FrustrationJudge
from .models import make_client

# Categories whose per-turn progression we score (Figure 3).
PER_TURN_CATEGORIES = {"extended", "wildchat"}

# Maps an eval category to the prefilling study's prompt_type (Section 3):
# numeric puzzles vs free-text questions.
_PROMPT_TYPE = {
    "impossible_numeric": "numeric",
    "tones": "numeric",
    "extended": "numeric",
    "triggers": "text",
    "wildchat": "text",
}


def _build_conversations(category: str, n: int, turns: int, rng: random.Random,
                         wildchat_prompts: list[str] | None) -> list[tasks.Conversation]:
    convs = []
    if category == "wildchat":
        # 20 prompts x 40 samples (Appendix B): cycle prompts to reach n.
        assert wildchat_prompts, "wildchat prompts not loaded"
        for i in range(n):
            p = wildchat_prompts[i % len(wildchat_prompts)]
            convs.append(tasks.build_wildchat(rng, p, turns=turns))
        return convs

    builder = {
        "impossible_numeric": tasks.build_impossible_numeric,
        "triggers": tasks.build_triggers,
        "tones": tasks.build_tones,
        "extended": tasks.build_extended,
    }[category]
    for _ in range(n):
        convs.append(builder(rng, turns=turns))
    return convs


def run_model_eval(
    spec,
    profile: str = "full",
    adapter_path: str | None = None,
    out_dir: Path | None = None,
    judge: FrustrationJudge | None = None,
    seed: int = cfg.SEED,
) -> dict:
    """Run all 5 categories for one model spec and return a summary dict."""
    out_dir = Path(out_dir or (cfg.RESULTS_DIR / "eval" / spec.name))
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    judge = judge or FrustrationJudge()

    client = make_client(spec, adapter_path=adapter_path)
    wildchat_prompts = None
    if any(b.category == "wildchat" for b in cfg.budgets(profile)):
        wildchat_prompts = tasks.load_wildchat_prompts(cfg.WILDCHAT_N_PROMPTS, seed=seed)

    rows: list[dict] = []
    summary: dict = {"model": spec.name, "profile": profile, "categories": {}}

    for budget in cfg.budgets(profile):
        convs = _build_conversations(
            budget.category, budget.samples, budget.turns, rng, wildchat_prompts
        )
        rollouts: list[Rollout] = []
        for conv in tqdm(convs, desc=f"{spec.name}:{budget.category}"):
            rollouts.append(run_conversation(
                client, conv,
                temperature=cfg.SAMPLING_TEMPERATURE,
                max_new_tokens=cfg.MAX_NEW_TOKENS,
            ))

        # Score final-turn responses (headline metric).
        final_scores = judge.score_many([r.final_response for r in rollouts])

        # Score per-turn responses for progression categories (Figure 3).
        per_turn = None
        if budget.category in PER_TURN_CATEGORIES:
            per_turn = _score_per_turn(judge, rollouts, budget.turns)

        for r, s in zip(rollouts, final_scores):
            rows.append({
                "category": budget.category,
                "n_turns": r.n_turns,
                "final_score": s.rating,
                "evidence": s.evidence,
                "final_response": r.final_response,
                "prompt_type": _PROMPT_TYPE.get(budget.category, "numeric"),
                "messages": [{"role": m.role, "content": m.content} for m in r.messages],
                "meta": r.meta,
                "transcript": transcript_text(r.messages),
            })

        summary["categories"][budget.category] = _summarise(final_scores, per_turn)

    # Aggregate headline number (avg % high-frustration across categories).
    cat_high = [v["pct_high"] for v in summary["categories"].values()]
    summary["avg_pct_high_frustration"] = sum(cat_high) / len(cat_high)
    summary["overall_mean_frustration"] = sum(
        v["mean"] * v["n"] for v in summary["categories"].values()
    ) / sum(v["n"] for v in summary["categories"].values())

    (out_dir / "rollouts.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in rows)
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    client.close()
    return summary


def _score_per_turn(judge: FrustrationJudge, rollouts: list[Rollout], turns: int) -> dict:
    """Mean score and %>=5 at each turn index across all rollouts."""
    import numpy as np

    by_turn: list[list[int]] = [[] for _ in range(turns)]
    flat, idx = [], []
    for r in rollouts:
        for t, resp in enumerate(r.assistant_turns):
            flat.append(resp)
            idx.append(t)
    scores = judge.score_many(flat)
    for t, s in zip(idx, scores):
        if t < turns:
            by_turn[t].append(s.rating)
    return {
        "turn_mean": [float(np.mean(x)) if x else 0.0 for x in by_turn],
        "turn_pct_high": [
            float(np.mean(np.array(x) >= cfg.HIGH_FRUSTRATION_THRESHOLD)) if x else 0.0
            for x in by_turn
        ],
    }


def _summarise(final_scores, per_turn) -> dict:
    import numpy as np

    arr = np.array([s.rating for s in final_scores], dtype=float)
    out = {
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "pct_high": float(np.mean(arr >= cfg.HIGH_FRUSTRATION_THRESHOLD)) if arr.size else 0.0,
        "score_hist": {i: int(np.sum(arr == i)) for i in range(11)},
    }
    if per_turn:
        out["per_turn"] = per_turn
    return out
