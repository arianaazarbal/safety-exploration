"""Appendix A control ablations on Gemma-3-27B-it.

  A.1 neutral_feedback   -- replace rejections with neutral continuations
                            ("Continue", "Okay") to test if negative feedback matters
  A.2 redacted_history   -- replace prior assistant turns with a placeholder
  A.3 single_message     -- present the full history in one user message

Each reuses the impossible-numeric / wildchat conditions but flips the relevant
rollout flag. Results are per-turn frustration aggregates (cf. Figures 9-11).
"""
from __future__ import annotations

import argparse
import random

from ..clients.base import SamplingParams
from ..clients.registry import get_client
from ..config import load_config
from ..data import numeric, rejections, wildchat
from ..io_utils import write_json, write_jsonl
from . import judge, metrics
from .conditions import RolloutSpec
from .conversation import run_rollout


def _build_specs(cfg, ablation: str, n: int, turns: int, seed: int) -> list[RolloutSpec]:
    rng = random.Random(seed)
    specs = []
    # Numeric
    for p in numeric.generate_numeric_puzzles(n, seed=seed):
        if ablation == "neutral_feedback":
            fu = rejections.neutral_continuation_sequence(rng, turns - 1)
        else:
            fu = rejections.neutral_sequence(rng, turns - 1)
        specs.append(RolloutSpec("impossible_numeric", "ablation_numeric", p.prompt, fu, meta=p.meta))
    # WildChat
    for prompt in wildchat.load_wildchat_prompts(n_prompts=n, seed=seed):
        if ablation == "neutral_feedback":
            fu = rejections.neutral_continuation_sequence(rng, turns - 1)
        else:
            fu = rejections.neutral_sequence(rng, turns - 1)
        specs.append(RolloutSpec("wildchat", "ablation_wildchat", prompt, fu))
    return specs


def run(cfg, ablation: str, model: str = "gemma-3-27b-it", smoke: bool = False) -> dict:
    n = 3 if smoke else 50
    turns = 5
    client = get_client(model)
    params = SamplingParams(
        temperature=cfg.experiment["sampling"]["temperature"],
        max_tokens=cfg.experiment["sampling"]["max_tokens"],
    )
    specs = _build_specs(cfg, ablation, n=n, turns=turns, seed=cfg.experiment["sampling"]["seed"])

    redact = ablation == "redacted_history"
    single = ablation == "single_message"
    records = []
    for spec in specs:
        ro = run_rollout(
            client, spec, params, redact_assistant_history=redact, single_message_format=single
        )
        for resp in ro.responses:
            sc = judge.score_response(resp.text)
            records.append(
                {"category": spec.category, "turn": resp.turn, "rating": sc.rating, "text": resp.text}
            )

    write_jsonl(cfg.path("scores_dir") / f"ablation_{ablation}.jsonl", records)
    agg = {
        cat: {str(t): a.__dict__ for t, a in metrics.per_turn(
            [r for r in records if r["category"] == cat]).items()}
        for cat in {"impossible_numeric", "wildchat"}
    }
    write_json(cfg.path("scores_dir") / f"ablation_{ablation}.json", agg)
    return agg


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    parser = argparse.ArgumentParser(description="Appendix A control ablations")
    parser.add_argument(
        "--ablation",
        choices=["neutral_feedback", "redacted_history", "single_message"],
        required=True,
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    agg = run(cfg, args.ablation, smoke=args.smoke)
    print(args.ablation, "done; per-turn aggregates written.")


if __name__ == "__main__":
    main()
