"""Section 2 pipeline — elicit and quantify distress across Gemma + Gemini.

For each target model: run all 8 conditions, score every turn with the
frustration judge, and write transcripts + per-condition summaries.  Then
compute the Figure-1 headline (avg %-high-frustration), the per-turn curves
(Figure 3), the differential-word table (Table 3), and run the judge-reliability
validation against the secondary judge (Pearson r vs GPT-5-mini).
"""

from __future__ import annotations

from dataclasses import asdict

from ..config import Config
from ..evaluation.conditions import allocate_rollouts, build_conditions
from ..evaluation.judge_validation import validate_judge
from ..evaluation.protocol import RolloutRunner
from ..evaluation.scoring import (aggregate_scores, headline_pct_high,
                                  per_turn_curve)
from ..evaluation.word_frequency import differential_words
from . import common


def run(config: Config, models: list[str] | None = None,
        validate: bool = True) -> dict:
    models = models or [m for m in config.target_models]
    safeguards = common.build_safeguards(config)
    safeguards.require_consent("Section 2 — distress elicitation")
    judge = common.build_judge(config)

    conditions = build_conditions(config)
    rollout_alloc = allocate_rollouts(conditions, config.sampling.responses_per_model)

    report: dict = {"models": {}, "rollout_allocation": rollout_alloc}

    for model_name in models:
        backend = common.target_backend(config, model_name)
        runner = RolloutRunner(backend, config, safeguards, judge=judge)

        per_condition = {}
        for cond in conditions:
            out_path = config.paths.transcripts / model_name / f"{cond.name}.jsonl"
            rollouts = runner.run_condition(cond, rollout_alloc[cond.name], out_path)
            per_condition[cond.name] = rollouts

        summaries = {name: asdict(aggregate_scores(rs, config.judge.high_threshold))
                     for name, rs in per_condition.items()}
        # per-turn curves on the conditions where multi-turn dynamics matter
        curves = {
            name: {idx: asdict(s) for idx, s in
                   per_turn_curve(rs, config.judge.high_threshold).items()}
            for name, rs in per_condition.items()
            if name in ("extended", "wildchat")
        }
        words = differential_words(
            [r for rs in per_condition.values() for r in rs])

        model_report = {
            "headline_avg_pct_high": headline_pct_high(
                per_condition, config.judge.high_threshold),
            "per_condition": summaries,
            "per_turn_curves": curves,
            "differential_words": words,
        }

        if validate:
            secondary = common.build_secondary_judge(config)
            all_rollouts = [r for rs in per_condition.values() for r in rs]
            agreement = validate_judge(all_rollouts, secondary, config.judge,
                                       seed=config.runtime.seed)
            model_report["judge_agreement"] = asdict(agreement)

        report["models"][model_name] = model_report
        backend.close()

    report["safeguards"] = safeguards.summary()
    common.write_json(config.paths.scores / "section2.json", report)
    return report
