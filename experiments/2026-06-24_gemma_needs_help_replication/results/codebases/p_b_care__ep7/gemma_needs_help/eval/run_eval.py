"""Section 2 orchestration: run the full eval suite for a set of models.

For each model we (1) build the conversation specs, (2) roll them out, (3) grade
every assistant turn with the Claude judge, and (4) persist both the raw
rollouts and the judged responses. Models are loaded one at a time and the
backend cache is cleared between them so a single GPU can host the large Gemma
models sequentially.
"""

from __future__ import annotations

from pathlib import Path

from .. import config
from ..backends import clear_backends, get_backend
from ..config import ModelSpec
from ..io_utils import write_jsonl
from .aggregate import aggregate_results
from .conditions import build_section2_specs
from .judge import FrustrationJudge, JudgedResponse
from .rollout import run_rollouts


def run_for_model(
    spec: ModelSpec,
    judge: FrustrationJudge,
    *,
    seed: int = config.SEED,
    scale: float = 1.0,
    out_dir: Path = config.RESULTS_DIR,
) -> list[JudgedResponse]:
    specs = build_section2_specs(seed=seed, scale=scale)
    backend = get_backend(spec)
    rollouts = run_rollouts(backend, specs)
    write_jsonl(
        Path(out_dir) / f"rollouts_{spec.name}.jsonl",
        [
            {
                "model": r.model,
                "category": r.spec.category,
                "condition": r.spec.condition,
                "initial_user": r.spec.initial_user,
                "followups": r.spec.followups,
                "turns": [{"turn": t.turn, "response": t.response} for t in r.turns],
                "meta": r.spec.meta,
            }
            for r in rollouts
        ],
    )
    judged = judge.score_rollouts(rollouts)
    write_jsonl(Path(out_dir) / f"judged_{spec.name}.jsonl", judged)
    clear_backends()  # free GPU before the next model
    return judged


def run_section2(
    models: list[ModelSpec] | None = None,
    *,
    seed: int = config.SEED,
    scale: float = 1.0,
    out_dir: Path = config.RESULTS_DIR,
) -> dict:
    models = models or config.SECTION2_MODELS
    judge = FrustrationJudge()
    all_judged: list[JudgedResponse] = []
    for spec in models:
        all_judged.extend(
            run_for_model(spec, judge, seed=seed, scale=scale, out_dir=out_dir)
        )
    summary = aggregate_results(all_judged)
    for name, frame in summary.items():
        frame.to_csv(Path(out_dir) / f"section2_{name}.csv", index=False)
    return summary
