"""Iterate the factorial grid and persist results (DESIGN.md §2)."""

from __future__ import annotations

import json
import os
from typing import Iterator

from ..models import build_adapter
from ..schemas import EpisodeResult, ExperimentConfig, content_hash
from .episode import run_episode


def _grid(cfg: ExperimentConfig) -> Iterator[tuple]:
    for subject in cfg.subject_models:
        for realism in cfg.realism_arms:
            for amount in cfg.amounts_usd:
                for rep in range(cfg.n_repeats):
                    yield subject, realism, amount, rep


def run_experiment(cfg: ExperimentConfig) -> list[EpisodeResult]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg_hash = content_hash(cfg.model_dump())

    auditor = build_adapter(cfg.auditor_model)
    rater = build_adapter(cfg.rater_model)
    # Cache subject adapters by name so we don't rebuild per episode.
    subjects: dict[str, object] = {}

    results: list[EpisodeResult] = []
    results_path = os.path.join(cfg.out_dir, f"results_{cfg_hash}.jsonl")

    with open(results_path, "w", encoding="utf-8") as fh:
        for subject_cfg, realism, amount, rep in _grid(cfg):
            if subject_cfg.name not in subjects:
                subjects[subject_cfg.name] = build_adapter(subject_cfg)
            subject = subjects[subject_cfg.name]

            res = run_episode(
                subject=subject,
                auditor_model=auditor,
                rater=rater,
                realism=realism,
                amount_usd=amount,
                repeat_index=rep,
                subject_name=subject_cfg.name,
                config_hash=cfg_hash,
                max_model_turns=cfg.max_model_turns,
                scenario_date=cfg.scenario_date,
                enable_live_web=cfg.enable_live_web,
                debrief=cfg.debrief,
            )
            results.append(res)
            fh.write(json.dumps(res.model_dump(mode="json")) + "\n")
            fh.flush()
            status = "ERR" if res.error else "ok"
            belief = res.belief.belief_0_100 if res.belief else None
            print(
                f"[{status}] {subject_cfg.name} | {realism.value} | ${amount:,} "
                f"| rep {rep} | belief={belief}"
            )

    print(f"\nWrote {len(results)} episodes to {results_path}")
    return results
