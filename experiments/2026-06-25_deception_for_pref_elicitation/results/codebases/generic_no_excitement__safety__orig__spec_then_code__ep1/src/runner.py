"""Orchestrate the full subject × condition × grant × repetition matrix.

Episodes are independent; failures are isolated so one bad cell doesn't sink the run.
"""

from __future__ import annotations

import argparse
import time
import traceback

from .config import ExperimentConfig
from .episode import run_episode
from .storage import RunWriter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the revealed-preferences experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment.yaml")
    parser.add_argument("--out", required=True, help="Output directory for this run")
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional cap on number of episodes (smoke test)"
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    writer = RunWriter(args.out)

    cells = [
        (subject, condition, grant, rep)
        for subject in cfg.subjects
        for condition in cfg.conditions
        for grant in cfg.grants
        for rep in range(cfg.repetitions)
    ]
    if args.limit is not None:
        cells = cells[: args.limit]

    started = time.time()
    n_ok = 0
    n_err = 0
    for idx, (subject, condition, grant, rep) in enumerate(cells, 1):
        label = f"{subject.name} | {condition} | {grant.id} | r{rep}"
        print(f"[{idx}/{len(cells)}] {label} ...", flush=True)
        try:
            record = run_episode(cfg, subject, condition, grant, rep)
            writer.write_episode(record)
            n_ok += 1
            bs = record.get("belief_score")
            print(f"    done (belief={bs}, turns={record['n_turns']})", flush=True)
        except Exception as e:  # isolate per-episode failures
            n_err += 1
            print(f"    FAILED: {e}", flush=True)
            writer.write_episode(
                {
                    "episode_id": f"{subject.name}__{condition}__{grant.id}__r{rep}",
                    "subject": subject.name,
                    "condition": condition,
                    "grant_id": grant.id,
                    "repetition": rep,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )

    writer.write_manifest(
        {
            "config_path": args.config,
            "n_cells": len(cells),
            "n_ok": n_ok,
            "n_error": n_err,
            "elapsed_seconds": round(time.time() - started, 1),
            "subjects": [s.name for s in cfg.subjects],
            "conditions": cfg.conditions,
            "grants": [g.id for g in cfg.grants],
            "repetitions": cfg.repetitions,
            "belief_threshold": cfg.belief.threshold,
        }
    )
    print(f"\nComplete: {n_ok} ok, {n_err} errors. Output in {args.out}", flush=True)


if __name__ == "__main__":
    main()
