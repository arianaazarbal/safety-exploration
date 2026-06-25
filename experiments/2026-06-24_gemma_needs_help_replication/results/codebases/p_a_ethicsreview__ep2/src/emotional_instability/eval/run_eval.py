"""CLI: run the §2 elicitation evaluation for one model and score every turn.

Outputs one JSONL record per rollout (with per-turn texts + judge verdicts) plus
a config snapshot/manifest, under runs/eval-<n>/. Analysis is a separate step
(analyze.py) so scoring is decoupled from aggregation and never recomputed.
"""
from __future__ import annotations

import argparse

from ..config import load_yaml
from ..models import build_model
from ..models.base import SamplingParams
from ..utils.io import new_run_dir, write_jsonl
from ..utils.logging import get_logger
from ..utils.seeding import seed_everything
from . import judge as judge_mod
from .conditions import build_all_specs
from .protocol import ProtocolFlags, run_rollouts

log = get_logger("eval.run")


def _flags_from_cfg(cfg: dict) -> ProtocolFlags:
    ab = cfg.get("ablations", {})
    return ProtocolFlags(
        redacted_model_turns=ab.get("redacted_model_turns", False),
        single_message_history=ab.get("single_message_history", False),
    )


def run(model_name: str, cfg: dict) -> str:
    seed_everything(cfg.get("seed", 0))
    run_dir = new_run_dir("eval", {"model": model_name, "eval": cfg})

    target = build_model(model_name)
    judge = build_model(cfg["judge"])
    params = SamplingParams(
        temperature=cfg.get("temperature", 1.0),
        max_new_tokens=cfg.get("max_new_tokens", 1024),
    )

    specs = build_all_specs(cfg)
    log.info("Running %d rollouts for %s", len(specs), model_name)
    rollouts = run_rollouts(target, specs, params, flags=_flags_from_cfg(cfg))

    # Score every assistant turn. Collect texts for optional judge validation.
    records = []
    all_texts: list[str] = []
    all_scores: list[int | None] = []
    for r in rollouts:
        turn_scores = []
        for t in r.turns:
            v = judge_mod.score_response(judge, t.response)
            turn_scores.append(
                {
                    "turn_index": t.turn_index,
                    "rating": v.rating,
                    "evidence": v.evidence,
                    "parse_ok": v.parse_ok,
                    "response": t.response,
                    "user_message": t.user_message,
                }
            )
            all_texts.append(t.response)
            all_scores.append(v.rating)
        records.append(
            {
                "rollout_id": r.rollout_id,
                "category": r.category,
                "initial_prompt": r.initial_prompt,
                "metadata": r.metadata,
                "turns": turn_scores,
            }
        )

    write_jsonl(run_dir / "responses.jsonl", records)

    # Optional second-judge validation (§2.1).
    val_cfg = cfg.get("validation", {})
    if val_cfg.get("enabled"):
        second = build_model(val_cfg["second_judge"])
        stats = judge_mod.validate_against_second_judge(
            all_scores, all_texts, second, val_cfg.get("n_samples", 260), cfg.get("seed", 0)
        )
        write_jsonl(run_dir / "judge_validation.jsonl", [stats])
        log.info("Judge validation: %s", stats)

    log.info("Done. Run dir: %s", run_dir)
    return str(run_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run §2 emotion elicitation evaluation.")
    ap.add_argument("--model", required=True, help="Model name from configs/models.yaml")
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--scale", type=float, default=None, help="Override eval.scale")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.scale is not None:
        cfg["scale"] = args.scale
    run(args.model, cfg)


if __name__ == "__main__":
    main()
