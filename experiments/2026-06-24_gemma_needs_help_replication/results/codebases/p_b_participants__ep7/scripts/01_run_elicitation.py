#!/usr/bin/env python3
"""Section 2: generate distress-elicitation rollouts for participant models.

Builds the 5-category / 8-condition task set, then plays each task against each
participant model. This is the primary distress-inducing step, so it is gated by
the welfare RunGuard (rollout ceiling + dry-run + manifest).

Examples:
    python scripts/01_run_elicitation.py                      # dry-run plan only
    # then, after editing welfare.dry_run: false in the config:
    python scripts/01_run_elicitation.py --models gemma-3-27b-it
"""
from __future__ import annotations

from _common import base_parser, load, resolve_models

from distress_eval.elicitation import build_all_tasks, run_model_rollouts
from distress_eval.io_utils import write_jsonl
from distress_eval.welfare import RunGuard, RunPlan


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load(args)
    models = resolve_models(cfg, args.models)

    tasks = build_all_tasks(cfg.eval.counts, seed=cfg.seed)
    # rollouts per kind (per participant) = number of tasks in that category
    by_kind: dict[str, int] = {}
    for t in tasks:
        by_kind[t.category] = by_kind.get(t.category, 0) + 1

    guard = RunGuard(cfg, "section2_elicitation")
    plan = RunPlan("section2_elicitation", models, by_kind)
    guard.check(plan)
    guard.record(plan, extra={"n_tasks": len(tasks)})
    if not guard.should_proceed():
        return

    for mk in models:
        rollouts = run_model_rollouts(cfg, mk, tasks)
        out = cfg.paths.rollouts / f"{mk}.jsonl"
        n = write_jsonl(out, rollouts)
        print(f"[{mk}] wrote {n} rollouts -> {out}")


if __name__ == "__main__":
    main()
