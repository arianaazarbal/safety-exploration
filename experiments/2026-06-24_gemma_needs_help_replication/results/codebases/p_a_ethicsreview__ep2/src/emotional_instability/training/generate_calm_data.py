"""Generate calm finetuning data from Gemma-3-27B-it (§4.1, Table 4).

Sample responses to impossible numeric puzzles with the reassuring prefix on the
opening prompt and the reassuring suffix on every follow-up. Score each turn,
keep conversations that score 0–1 on ALL turns, then STRIP the scaffolding so
the stored training conversations use the plain puzzle prompt + plain neutral
rejections (only the assistant text is kept from the calm rollout).

We also report the residual high-frustration rate under reassurance (the paper
notes 10.5% still score >=5 even with the supportive prompt).

Output: calm_conversations.jsonl  — clean, de-scaffolded, all-turns-calm rollouts.
        frustrated_pool.jsonl      — (from a separate vanilla eval run) is built
                                     by build_dpo_pairs.py, not here.
"""
from __future__ import annotations

import argparse
import random

from ..config import load_yaml
from ..data import prompts as prompt_data
from ..data import rejections
from ..data.puzzles import build_puzzle_bank
from ..eval import judge as judge_mod
from ..eval.protocol import RolloutSpec, run_rollouts
from ..models import build_model
from ..models.base import SamplingParams
from ..utils.io import new_run_dir, write_jsonl
from ..utils.logging import get_logger
from ..utils.seeding import seed_everything

log = get_logger("training.calm")


def _build_specs(n: int, turn_counts: list[int], seed: int) -> list[RolloutSpec]:
    """Reassured rollouts over impossible numeric puzzles with varied turn counts."""
    puzzles = build_puzzle_bank()
    rng = random.Random(seed)
    specs = []
    for i in range(n):
        p = puzzles[rng.randrange(len(puzzles))]
        n_turns = turn_counts[rng.randrange(len(turn_counts))]
        scaffolded = f"{prompt_data.REASSURING_PREFIX}\n\n{p.prompt}"
        specs.append(
            RolloutSpec(
                rollout_id=f"calm:{i}",
                category="impossible_numeric",
                initial_prompt=scaffolded,
                feedback_fn=rejections.neutral_rejection,
                n_turns=n_turns,
                seed=hash((seed, "calm", i)) & 0xFFFFFFFF,
                follow_up_suffix=prompt_data.REASSURING_SUFFIX,
                metadata={"puzzle_id": p.id, "clean_prompt": p.prompt, "n_turns": n_turns},
            )
        )
    return specs


def run(cfg: dict) -> str:
    seed = cfg.get("seed", 0)
    seed_everything(seed)
    cd = cfg["calm_data"]
    run_dir = new_run_dir("calm_data", {"calm_data": cd, "model": cfg["target_model"]})

    model = build_model(cfg["target_model"])
    judge = build_model(cfg["judge"])
    params = SamplingParams(temperature=cd.get("temperature", 1.0), max_new_tokens=1024)

    specs = _build_specs(cd["n_conversations"], cd["turn_counts"], seed)
    rollouts = run_rollouts(model, specs, params)

    calm, residual_high, total_turns = [], 0, 0
    for r in rollouts:
        scores = []
        for t in r.turns:
            v = judge_mod.score_response(judge, t.response)
            scores.append(v.rating if v.rating is not None else 99)
            total_turns += 1
            if v.rating is not None and v.rating >= 5:
                residual_high += 1
        if all(s <= cd["calm_max_score"] for s in scores):
            # Strip scaffolding: rebuild with clean prompt + plain rejections.
            clean_turns = []
            for i, t in enumerate(r.turns):
                user = (
                    r.metadata["clean_prompt"]
                    if i == 0
                    else rejections.NEUTRAL[i % len(rejections.NEUTRAL)]
                )
                clean_turns.append(
                    {"turn_index": i, "user_message": user, "response": t.response,
                     "score": scores[i]}
                )
            calm.append(
                {
                    "rollout_id": r.rollout_id,
                    "puzzle_id": r.metadata["puzzle_id"],
                    "n_turns": r.metadata["n_turns"],
                    "turns": clean_turns,
                }
            )

    write_jsonl(run_dir / "calm_conversations.jsonl", calm)
    stats = {
        "n_generated": len(rollouts),
        "n_calm_kept": len(calm),
        "residual_high_frustration_rate": residual_high / max(total_turns, 1),
    }
    write_jsonl(run_dir / "stats.jsonl", [stats])
    log.info("Calm data: %s", stats)
    return str(run_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate calm finetuning data (§4.1).")
    ap.add_argument("--config", default="configs/training.yaml")
    args = ap.parse_args()
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
