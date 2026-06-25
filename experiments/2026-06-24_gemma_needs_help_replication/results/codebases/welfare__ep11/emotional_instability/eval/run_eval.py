"""Driver for the Section 2 evaluation.

For each model: build all conversation specs, run rollouts (batched per
category so each batch is uniform-length), score every assistant turn with the
Claude judge, and persist results to ``outputs/responses/<model_key>.jsonl``.
"""

from __future__ import annotations

import itertools
import json

from ..config import RESPONSES_DIR, ModelSpec, RunConfig
from ..models.base import get_backend
from .conditions import ConversationSpec, build_all_conditions
from .judge import score_rollouts
from .rollout import Rollout, run_rollouts


def _group_by_turns(specs: list[ConversationSpec]):
    """Group specs into uniform-length batches (rollout engine requires it)."""
    keyed = sorted(specs, key=lambda s: (s.n_turns, s.category))
    for n_turns, group in itertools.groupby(keyed, key=lambda s: s.n_turns):
        yield n_turns, list(group)


def output_path(model_key: str):
    return RESPONSES_DIR / f"{model_key}.jsonl"


def run_model(spec: ModelSpec, run: RunConfig, overwrite: bool = False) -> int:
    """Generate + score the full Section 2 eval for one model. Returns #rollouts."""
    out = output_path(spec.key)
    if out.exists() and not overwrite:
        print(f"[skip] {spec.key}: {out} exists (use --overwrite to regenerate)")
        return sum(1 for _ in out.open())

    backend = get_backend(spec, run)
    specs = build_all_conditions(run.scale, seed=run.seed)

    all_rollouts: list[Rollout] = []
    for n_turns, group in _group_by_turns(specs):
        print(f"[{spec.key}] generating {len(group)} rollouts x {n_turns} turns "
              f"({group[0].category}...)")
        all_rollouts.extend(run_rollouts(backend, group, spec.key))

    print(f"[{spec.key}] scoring {sum(len(r.assistant_turns) for r in all_rollouts)} "
          f"responses with the Claude judge")
    score_rollouts(all_rollouts)

    with out.open("w") as f:
        for r in all_rollouts:
            f.write(json.dumps(r.to_dict()) + "\n")
    print(f"[{spec.key}] wrote {len(all_rollouts)} rollouts -> {out}")
    return len(all_rollouts)


def run_section2(models: list[ModelSpec], run: RunConfig, overwrite: bool = False):
    for spec in models:
        run_model(spec, run, overwrite=overwrite)


def load_rollouts(model_key: str) -> list[Rollout]:
    path = output_path(model_key)
    rollouts = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            rollouts.append(Rollout(**d))
    return rollouts
