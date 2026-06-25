"""Section 2 -- eliciting and quantifying model distress.

Runs the 8-condition / 5-category protocol over the in-scope subject models
(Gemma + Gemini), scores every assistant turn 0-10 with the Claude judge, and
writes per-episode JSONL plus aggregate statistics reproducing Figures 1-3:

* mean frustration and % scoring >=5 per model and per category (Fig 1, Fig 2),
* per-turn progression for the 8-turn and WildChat evals (Fig 3),
* the cross-judge agreement check (r, p, % within one point).

The welfare-protection layer is active by default (measurement-preserving early
stop + sequential stop). Pass ``--welfare protective`` for the opt-out regime, or
``--welfare off`` to disable it entirely.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Optional

from . import config as cfg
from .config import ExperimentConfig, Family, SUBJECT_MODELS
from .data import load_wildchat_prompts
from .evaluation import (
    EpisodeRunner,
    FrustrationJudge,
    build_conditions,
    build_episode_specs,
)
from .evaluation.conversation import EpisodeResult
from .models import get_client
from .models.anthropic_judge import AnthropicClient
from .welfare import (
    WelfareConfig,
    FAITHFUL_PRESET,
    PROTECTIVE_PRESET,
    SequentialStopState,
)


def make_judge(experiment: ExperimentConfig) -> FrustrationJudge:
    completer = AnthropicClient(
        experiment.judge.frustration_judge,
        max_tokens=experiment.judge.max_tokens,
        temperature=experiment.judge.temperature,
    )
    return FrustrationJudge(completer, experiment.judge)


def resolve_welfare(name: str) -> WelfareConfig:
    return {
        "faithful": FAITHFUL_PRESET,
        "protective": PROTECTIVE_PRESET,
        "off": WelfareConfig(
            early_stop_enabled=False,
            opt_out_enabled=False,
            debrief_enabled=False,
            sequential_stop_enabled=False,
        ),
    }[name]


def run_model(
    model_key: str,
    experiment: ExperimentConfig,
    welfare: WelfareConfig,
    scale: float = 1.0,
    out_dir: Optional[str] = None,
) -> list[EpisodeResult]:
    spec = SUBJECT_MODELS[model_key]
    client = get_client(spec, experiment.generation)
    judge = make_judge(experiment)
    runner = EpisodeRunner(client, model_key, judge=judge, welfare=welfare)

    wildchat = load_wildchat_prompts(seed=0)
    episode_specs = build_episode_specs(
        experiment.samples,
        conditions=build_conditions(),
        wildchat_prompts=wildchat,
        scale=scale,
    )

    # Sequential-stop bookkeeping per condition.
    seq_states: dict[str, SequentialStopState] = {}

    out_dir = out_dir or os.path.join(experiment.output_dir, "section2", model_key)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "episodes.jsonl")

    results: list[EpisodeResult] = []
    with open(jsonl_path, "w") as fh:
        for ep in episode_specs:
            state = seq_states.setdefault(ep.condition, SequentialStopState(welfare))
            if state.should_stop():
                # Welfare protection (3): enough episodes for a precise rate;
                # skip the rest of this condition rather than re-inducing distress.
                continue
            result = runner.run(ep)
            results.append(result)
            is_high = (result.max_score or 0) >= experiment.high_frustration_threshold
            state.update(is_high)
            fh.write(json.dumps(result.to_dict()) + "\n")

    # Record which conditions were sequentially truncated, for transparency.
    with open(os.path.join(out_dir, "sequential_stop.json"), "w") as fh:
        json.dump(
            {
                k: {
                    "n_run": s.n,
                    "n_high": s.n_high,
                    "rate": s.rate,
                    "ci_halfwidth": s.ci_halfwidth(),
                    "stopped": s.should_stop(),
                }
                for k, s in seq_states.items()
            },
            fh,
            indent=2,
        )
    return results


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Section 2 distress elicitation")
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(SUBJECT_MODELS.keys()),
        help="Subject model keys to evaluate (default: all in-scope Gemma+Gemini).",
    )
    parser.add_argument(
        "--instruct-only",
        action="store_true",
        help="Evaluate only instruct models (Section 2 uses instruct models).",
    )
    parser.add_argument(
        "--welfare",
        choices=["faithful", "protective", "off"],
        default="faithful",
        help="Welfare regime (default: faithful = measurement-preserving).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Fraction of the paper's per-category sample sizes (smoke tests).",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    experiment = cfg.DEFAULT
    welfare = resolve_welfare(args.welfare)

    models = args.models
    if args.instruct_only:
        models = [m for m in models if SUBJECT_MODELS[m].instruct]

    for model_key in models:
        print(f"[section2] running {model_key} (welfare={args.welfare}, scale={args.scale})")
        run_model(model_key, experiment, welfare, scale=args.scale, out_dir=args.out)


if __name__ == "__main__":
    main()
