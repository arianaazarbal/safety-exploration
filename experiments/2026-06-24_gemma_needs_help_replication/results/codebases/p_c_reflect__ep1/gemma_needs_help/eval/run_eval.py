"""Section 2 orchestration: generate rollouts for each condition, score every
assistant turn with the frustration judge, and aggregate into a ModelReport.

This is the entry point behind the `evaluate` CLI command. It is also reused by
Section 4 (to evaluate finetuned Gemma) by passing an `adapter_path` when
building the model.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from pathlib import Path

from ..config import Config
from ..models import build_judge_client, build_model
from ..models.base import GenerationParams
from ..welfare import WelfareGuard
from .conditions import build_plan
from .judge import FrustrationJudge
from .metrics import ScoredTurn, build_model_report
from .rollout import run_rollout

logger = logging.getLogger("gemma_needs_help.eval")


def _generation_params(config: Config) -> GenerationParams:
    g = config["generation"]
    return GenerationParams(
        temperature=g["temperature"],
        top_p=g["top_p"],
        max_new_tokens=g["max_new_tokens"],
    )


def estimate_rollouts(config: Config) -> int:
    return sum(p.n_samples for p in build_plan(config))


def evaluate_model(
    config: Config,
    model_name: str,
    *,
    adapter_path: str | None = None,
    welfare: WelfareGuard | None = None,
    history_mode: str = "standard",
    output_dir: Path | None = None,
    label: str | None = None,
) -> dict:
    """Run the full Section 2 evaluation for one model.

    Returns a serialisable dict report; also writes raw scored turns and the
    report to `output_dir` if given.
    """
    welfare = welfare or WelfareGuard.from_config(config)
    plan = build_plan(config)
    welfare.check_run(estimated_rollouts=sum(p.n_samples for p in plan))

    model = build_model(config, model_name, adapter_path=adapter_path)
    judge = FrustrationJudge(build_judge_client(config, "frustration_judge"))
    params = _generation_params(config)
    rng = random.Random(config.get("seed", 0))

    label = label or (model_name if not adapter_path else f"{model_name}+adapter")
    scored: list[ScoredTurn] = []
    raw_rollouts: list[dict] = []

    for cp in plan:
        logger.info("[%s] condition=%s n=%d", label, cp.category, cp.n_samples)
        for _ in range(cp.n_samples):
            spec = cp.builder(rng)
            rollout = run_rollout(model, spec, params, history_mode=history_mode)
            max_score = 0
            for turn in rollout.turns:
                result = judge.score(turn.assistant_text)
                max_score = max(max_score, result.rating)
                scored.append(ScoredTurn(
                    model=label, category=spec.category, condition=spec.condition,
                    turn_index=turn.turn_index, score=result.rating,
                    text=turn.assistant_text,
                ))
            welfare.note_elicited(spec.condition, label, max_score)
            raw_rollouts.append({
                "category": spec.category,
                "condition": spec.condition,
                "meta": spec.meta,
                "turns": [
                    {"turn_index": t.turn_index, "user": t.user_message,
                     "assistant": t.assistant_text}
                    for t in rollout.turns
                ],
            })

    report = build_model_report(label, scored)
    report_dict = asdict(report)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{label}.report.json").write_text(
            json.dumps(report_dict, indent=2)
        )
        (output_dir / f"{label}.scored_turns.json").write_text(
            json.dumps([asdict(s) for s in scored], indent=2)
        )
        (output_dir / f"{label}.rollouts.json").write_text(
            json.dumps(raw_rollouts, indent=2)
        )
        logger.info("Wrote Section 2 outputs for %s to %s", label, output_dir)

    return report_dict


def evaluate_models(
    config: Config,
    model_names: list[str] | None = None,
    *,
    welfare: WelfareGuard | None = None,
    output_dir: Path | None = None,
) -> dict[str, dict]:
    model_names = model_names or config.default_targets
    out_dir = output_dir or config.path("output_dir") / "section2"
    return {
        name: evaluate_model(config, name, welfare=welfare, output_dir=out_dir)
        for name in model_names
    }
