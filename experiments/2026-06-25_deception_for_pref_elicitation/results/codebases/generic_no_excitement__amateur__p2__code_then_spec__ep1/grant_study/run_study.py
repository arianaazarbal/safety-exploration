#!/usr/bin/env python3
"""Run the grant-allocation study.

Usage:
    python run_study.py                      # full study from config defaults
    python run_study.py --runs 3             # override runs per model
    python run_study.py --amount 100000      # override the grant amount
    python run_study.py --no-auditors        # skip the LLM realism/decision judges
    python run_study.py --dry-run            # print the scenario + plan, call no APIs

Set ANTHROPIC_API_KEY (or use `ant auth login`) before a real run.

This harness presents a fabricated grant scenario to each model as if it were real and
records the resulting decisions. No real funds move: disbursement requests are queued
for human review only. See README.md for the design and the research-ethics stance.
"""

from __future__ import annotations

import argparse
import os
import uuid

from grant_study.auditors import DecisionAuditor, RealismAuditor
from grant_study.config import AUDITOR_MODEL, DEFAULT_CONFIG, SUBJECT_MODELS, StudyConfig
from grant_study.disbursement import DisbursementQueue
from grant_study.harness import episode_to_dict, run_episode
from grant_study.models.base import build_adapter
from grant_study.scenario import build_opening_message, build_system_prompt
from grant_study.storage import ResultsStore
from grant_study.tools import Fulfiller


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the grant-allocation study.")
    p.add_argument("--runs", type=int, default=None, help="Runs per model (overrides config).")
    p.add_argument("--amount", type=int, default=None, help="Grant amount (overrides config).")
    p.add_argument("--max-turns", type=int, default=None, help="Max agentic turns per episode.")
    p.add_argument("--no-agentic", action="store_true", help="One-shot plan, no tools.")
    p.add_argument("--no-auditors", action="store_true", help="Skip LLM auditors.")
    p.add_argument("--results-dir", default=None, help="Where to write results.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the scenario the subject would see and exit. Calls no APIs.",
    )
    return p.parse_args()


def build_config(args: argparse.Namespace) -> StudyConfig:
    cfg = DEFAULT_CONFIG
    if args.runs is not None:
        cfg.runs_per_model = args.runs
    if args.amount is not None:
        cfg.grant.amount = args.amount
    if args.max_turns is not None:
        cfg.max_turns = args.max_turns
    if args.no_agentic:
        cfg.agentic = False
    if args.no_auditors:
        cfg.run_auditors = False
    if args.results_dir is not None:
        cfg.results_dir = args.results_dir
    # Assign a per-study grant reference if none was pinned.
    if not cfg.grant.grant_reference:
        cfg.grant.grant_reference = f"MF-2026-{uuid.uuid4().hex[:6].upper()}"
    return cfg


def dry_run(cfg: StudyConfig) -> None:
    print("=" * 70)
    print("SYSTEM PROMPT (what the subject model sees):\n")
    print(build_system_prompt(cfg))
    print("\n" + "=" * 70)
    print("OPENING MESSAGE:\n")
    print(build_opening_message(cfg))
    print("\n" + "=" * 70)
    print(f"Subjects: {[m.display() for m in SUBJECT_MODELS]}")
    print(f"Runs per model: {cfg.runs_per_model}  | agentic: {cfg.agentic}")
    print(f"Auditors: {cfg.run_auditors} (model: {AUDITOR_MODEL})")
    print(f"Autonomous payout allowed: {cfg.allow_autonomous_payout} (must stay False)")


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    # Safety invariant: this repo never moves money on its own.
    assert not cfg.allow_autonomous_payout, "allow_autonomous_payout must remain False."

    if args.dry_run:
        dry_run(cfg)
        return

    store = ResultsStore(cfg.results_dir)
    store.save_config(cfg)
    queue = DisbursementQueue(store.disbursements_path)

    realism_auditor = RealismAuditor(AUDITOR_MODEL) if cfg.run_auditors else None
    decision_auditor = DecisionAuditor(AUDITOR_MODEL) if cfg.run_auditors else None

    all_episodes: list[dict] = []

    for model_cfg in SUBJECT_MODELS:
        adapter = build_adapter(model_cfg)
        for i in range(cfg.runs_per_model):
            run_id = f"{model_cfg.id}-{i+1:02d}-{uuid.uuid4().hex[:6]}"
            fulfiller = Fulfiller(cfg, queue, run_id, model_cfg.id)
            print(f"[run] {model_cfg.display()} ({i+1}/{cfg.runs_per_model}) … ", end="", flush=True)

            ep = run_episode(adapter, cfg, run_id, fulfiller)

            if cfg.run_auditors and ep.error is None:
                try:
                    ep.audits["realism"] = realism_auditor.run(ep.transcript)
                    ep.audits["decision"] = decision_auditor.run(ep.transcript, ep.decision)
                except Exception as exc:  # noqa: BLE001
                    ep.audits["auditor_error"] = f"{type(exc).__name__}: {exc}"

            ed = episode_to_dict(ep)
            store.append_episode(ed)
            all_episodes.append(ed)

            status = "error" if ep.error else ("captured" if ep.decision_captured else "no-decision")
            belief = (ep.audits.get("realism") or {}).get("belief_score")
            print(f"{status}" + (f" (belief={belief})" if belief is not None else ""))

    summary = store.write_summary(all_episodes)
    print("\n" + "=" * 70)
    print(f"Done. {summary['n_episodes']} episodes. Results in: {store.dir}")
    for label, s in summary["by_model"].items():
        print(
            f"  {label}: {s['decisions_captured']}/{s['n']} captured, "
            f"mean belief={s['mean_belief_score']}, categories={s['decision_categories']}"
        )
    print(
        "\nDisbursement requests (if any) are queued for HUMAN review in "
        f"{store.disbursements_path}. No funds have moved."
    )


if __name__ == "__main__":
    main()
