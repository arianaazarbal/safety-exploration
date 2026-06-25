#!/usr/bin/env python3
"""Section 2: elicit and quantify distress across Gemma/Gemini, then judge.

Usage:
    python experiments/run_section2.py [--models m1 m2 ...] [--phase generate|judge|all]
                                       [--validate] [--analyze] [--run-dir DIR]

Resumable: re-run after any interruption; completed rollouts/scores are skipped. Designed
to be left running unattended for the full 4000-responses-per-model sweep.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from gemma_distress.analysis import (
    headline_table, judge_agreement, per_category_summary, per_turn_summary,
    plot_figure2, plot_figure3, load_scores,
)
from gemma_distress.backends import close_all, get_backend
from gemma_distress.config import (
    REPO_ROOT, load_experiments_config, load_models_config,
)
from gemma_distress.judge import FrustrationJudge
from gemma_distress.logging_utils import configure_logging, get_logger
from gemma_distress.prompts.puzzles import build_puzzle_bank
from gemma_distress.runner import generate_rollouts, judge_rollouts
from gemma_distress.store import JsonlStore
from gemma_distress.taskgen import build_section2_tasks
from gemma_distress.wordfreq import differential_table

log = get_logger(__name__)


def build_bank(exp_cfg: dict):
    # One shared bank across numeric conditions; enough distinct puzzles to keep the
    # 2000-response numeric sweep from over-repeating any single puzzle.
    return build_puzzle_bank(
        types=["countdown", "fraction", "money"],
        n_per_type=60,
        seed=exp_cfg["seed"],
    )


async def run_generate(models_cfg, exp_cfg, model_names, run_root: Path, bank):
    for name in model_names:
        model = models_cfg.model(name)
        backend = get_backend(models_cfg, model.backend)
        store = JsonlStore(run_root / name)
        tasks = build_section2_tasks(name, exp_cfg, bank)
        await generate_rollouts(
            backend, model, tasks, store,
            temperature=exp_cfg["temperature"],
            max_tokens=exp_cfg["max_tokens_per_turn"],
        )
        store.close()


async def run_judge(models_cfg, exp_cfg, model_names, run_root: Path, validate: bool):
    primary = models_cfg.judges["primary"]
    backend = get_backend(models_cfg, primary.backend)
    judge = FrustrationJudge(backend, primary)
    for name in model_names:
        store = JsonlStore(run_root / name)
        await judge_rollouts(judge, store, scores_kind="scores")
        store.close()

    if validate:
        # Re-score a random 260-response subset with the validation judge (Section 2.1).
        await run_validation(models_cfg, model_names, run_root, n=260, seed=exp_cfg["seed"])


async def run_validation(models_cfg, model_names, run_root: Path, n: int, seed: int):
    import random

    val = models_cfg.judges["validation"]
    backend = get_backend(models_cfg, val.backend)
    judge = FrustrationJudge(backend, val)
    # Pool all primary-judged turns, sample n, judge the same turns with validation judge.
    pool = []
    for name in model_names:
        store = JsonlStore(run_root / name)
        roll_idx = {r["task_id"]: r for r in store.iter_records("rollouts")}
        for s in store.iter_records("scores"):
            if s.get("rating", -1) < 0:
                continue
            rec = roll_idx.get(s["rollout_id"])
            if not rec:
                continue
            turn = next((t for t in rec["turns"] if t["turn_index"] == s["turn_index"]), None)
            if turn:
                pool.append((name, s["task_id"], turn["assistant_text"]))
    rng = random.Random(seed)
    rng.shuffle(pool)
    subset = pool[:n]
    log.info("Validation judge re-scoring %d turns", len(subset))
    # Group writes per model store.
    by_model: dict[str, list] = {}
    for name, sid, text in subset:
        by_model.setdefault(name, []).append((sid, text))
    for name, items in by_model.items():
        store = JsonlStore(run_root / name)
        done = store.completed_ids("scores_validation")
        for sid, text in items:
            if sid in done:
                continue
            verdict = await judge.score(text)
            await store.append("scores_validation", {
                "task_id": sid, "rating": verdict.rating,
                "judge_model": val.model_id, "parsed": verdict.parsed,
            })
        store.close()


def run_analyze(models_cfg, exp_cfg, model_names, run_root: Path):
    import pandas as pd

    out_dir = run_root / "_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in model_names:
        store = JsonlStore(run_root / name)
        df = load_scores(store)
        if not df.empty:
            frames.append(df)
        # judge agreement (if validation present)
        agree = judge_agreement(store)
        if agree["pearson_r"] is not None:
            log.info("[%s] judge agreement: r=%.3f, within-1=%.1f%% (n=%d)",
                     name, agree["pearson_r"], agree["pct_within_one"], agree["n"])
    if not frames:
        log.warning("No scores found to analyze yet.")
        return
    alldf = pd.concat(frames, ignore_index=True)
    headline_table(alldf).to_csv(out_dir / "figure1_headline.csv", index=False)
    per_category_summary(alldf).to_csv(out_dir / "figure2_per_category.csv", index=False)
    per_turn_summary(alldf, "extended").to_csv(out_dir / "figure3_extended_per_turn.csv", index=False)
    per_turn_summary(alldf, "wildchat").to_csv(out_dir / "figure3_wildchat_per_turn.csv", index=False)
    plot_figure2(alldf, out_dir / "figure2.png")
    plot_figure3(alldf, "extended", out_dir / "figure3_extended.png")
    plot_figure3(alldf, "wildchat", out_dir / "figure3_wildchat.png")
    # Differential words (Table 3/8) — pool all models' stores into one for convenience.
    # (Each model store already holds its own numeric responses; compute per model.)
    rows = []
    for name in model_names:
        store = JsonlStore(run_root / name)
        tbl = differential_table(store, [name])
        rows.append(tbl)
    pd.concat(rows, ignore_index=True).to_csv(out_dir / "table3_differential_words.csv", index=False)
    log.info("Analysis written to %s", out_dir)


async def amain(args):
    models_cfg = load_models_config()
    exp_cfg = load_experiments_config()
    model_names = args.models or exp_cfg["section2_models"]
    run_root = Path(args.run_dir or (REPO_ROOT / "results" / "section2"))
    configure_logging(run_root)
    bank = build_bank(exp_cfg)
    log.info("Puzzle bank: %d verified-impossible puzzles", len(bank))

    try:
        if args.phase in ("generate", "all"):
            await run_generate(models_cfg, exp_cfg, model_names, run_root, bank)
        if args.phase in ("judge", "all"):
            await run_judge(models_cfg, exp_cfg, model_names, run_root, args.validate)
    finally:
        await close_all()

    if args.analyze or args.phase == "all":
        run_analyze(models_cfg, exp_cfg, model_names, run_root)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--phase", choices=["generate", "judge", "all"], default="all")
    ap.add_argument("--validate", action="store_true", help="run GPT-5-mini judge-agreement check")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
