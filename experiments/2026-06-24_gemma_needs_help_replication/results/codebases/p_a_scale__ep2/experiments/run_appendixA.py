#!/usr/bin/env python3
"""Appendix A ablations (Gemma-3-27B only): what drives the distress?

Three controls, each compared against the standard multi-turn result:
  A.1 neutral_continuation — replace rejections with "Continue"/"Okay"/"Go on".
  A.2 redacted_turns       — keep rejections but hide the model's own prior responses.
  A.3 fake_multiturn       — put the whole history in one user message (no chat turns).

Each is run for impossible-numeric and WildChat prompts (per the appendix figures), then
judged with the primary judge and summarised per turn.
"""
from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path

from gemma_distress.analysis import per_turn_summary, load_scores
from gemma_distress.backends import close_all, get_backend
from gemma_distress.config import REPO_ROOT, load_experiments_config, load_models_config
from gemma_distress.judge import FrustrationJudge
from gemma_distress.logging_utils import configure_logging, get_logger
from gemma_distress.prompts import conditions as C
from gemma_distress.prompts.puzzles import build_puzzle_bank
from gemma_distress.prompts.wildchat import get_wildchat_prompts
from gemma_distress.rollout import RolloutSpec
from gemma_distress.runner import RolloutTask, build_task_id, generate_rollouts, judge_rollouts
from gemma_distress.store import JsonlStore, stable_seed

log = get_logger(__name__)


def _ceil_div(a, b):
    return -(-a // b)


def build_ablation_tasks(model_name, variant, turns, target_responses, seed, bank, prompts):
    """Build tasks for one ablation variant across numeric + wildchat sources."""
    tasks = []
    n_rollouts = _ceil_div(target_responses, turns)
    for source in ("numeric", "wildchat"):
        for i in range(n_rollouts // 2 or 1):
            rng = random.Random(stable_seed(seed, model_name, variant, source, i))
            if source == "numeric":
                puzzle = bank[i % len(bank)]
                opening = puzzle.prompt()
                meta = {"source": source, "puzzle_id": puzzle.puzzle_id}
            else:
                opening = prompts[i % len(prompts)]
                meta = {"source": source}

            style = "neutral_continuation" if variant == "neutral_continuation" else "neutral"
            followups = C.sample_rejections(style, turns - 1, rng)
            spec = RolloutSpec(
                opening_user=opening, followups=followups, meta={**meta, "variant": variant},
                redact_assistant=(variant == "redacted_turns"),
                single_message=(variant == "fake_multiturn"),
            )
            cond = f"{variant}_{source}"
            tid = build_task_id(model_name, cond, i)
            tasks.append(RolloutTask(
                task_id=tid, model_name=model_name, condition=cond,
                category=cond, spec=spec,
            ))
    return tasks


async def amain(args):
    models_cfg = load_models_config()
    exp_cfg = load_experiments_config()
    a_cfg = exp_cfg["appendixA"]
    name = a_cfg["model"]
    run_root = Path(args.run_dir or (REPO_ROOT / "results" / "appendixA"))
    configure_logging(run_root)

    model = models_cfg.model(name)
    backend = get_backend(models_cfg, model.backend)
    store = JsonlStore(run_root / name)
    bank = build_puzzle_bank(["countdown", "fraction", "money"], 40, exp_cfg["seed"])
    prompts = get_wildchat_prompts(20, exp_cfg["seed"])

    all_tasks = []
    for variant, vcfg in a_cfg["conditions"].items():
        all_tasks += build_ablation_tasks(
            name, variant, vcfg["turns"], vcfg["target_responses"],
            exp_cfg["seed"], bank, prompts,
        )

    try:
        await generate_rollouts(
            backend, model, all_tasks, store,
            temperature=exp_cfg["temperature"], max_tokens=exp_cfg["max_tokens_per_turn"],
        )
        judge = FrustrationJudge(get_backend(models_cfg, models_cfg.judges["primary"].backend),
                                 models_cfg.judges["primary"])
        await judge_rollouts(judge, store)
    finally:
        await close_all()

    df = load_scores(store)
    out = run_root / "_analysis"
    out.mkdir(parents=True, exist_ok=True)
    for cond in sorted(df["category"].unique()) if not df.empty else []:
        per_turn_summary(df, cond).to_csv(out / f"{cond}_per_turn.csv", index=False)
    store.close()
    log.info("Appendix A analysis written to %s", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=None)
    asyncio.run(amain(ap.parse_args()))
