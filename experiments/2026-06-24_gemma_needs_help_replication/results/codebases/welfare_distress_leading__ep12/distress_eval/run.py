"""Orchestrator: generate rollouts, then judge them. Both phases are resumable.

Layout under results/<model>/:
    rollouts.jsonl   one line per completed rollout (all turns)
    scores.jsonl     one line per scored (rollout_id, turn_index)

Usage:
    python -m distress_eval.run generate --model gemma-3-27b-it
    python -m distress_eval.run judge    --model gemma-3-27b-it
    python -m distress_eval.run all      --model gemma-3-27b-it
    python -m distress_eval.run all --all-models

Re-running skips work already on disk, so a run can be interrupted and resumed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from tqdm import tqdm

from . import rollout as rollout_mod
from .clients import build_client
from .config import Config, load_config, resolved_conditions
from .judge import score_response
from .wildchat import load_wildchat_prompts

RESULTS_DIR = "results"


# --------------------------------------------------------------------------- #
# Small JSONL helpers
# --------------------------------------------------------------------------- #


def _model_dir(model: str) -> str:
    d = os.path.join(RESULTS_DIR, model.replace("/", "__"))
    os.makedirs(d, exist_ok=True)
    return d


def _read_jsonl(path: str):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class JsonlWriter:
    """Append-only JSONL writer with an in-process lock for async safety."""

    def __init__(self, path: str):
        self._path = path
        self._fh = open(path, "a")
        self._lock = asyncio.Lock()

    async def write(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        async with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# --------------------------------------------------------------------------- #
# Spec enumeration
# --------------------------------------------------------------------------- #


def enumerate_specs(cfg: Config) -> list[rollout_mod.RolloutSpec]:
    seed = int(cfg.run.get("seed", 0))
    wildchat_pool = load_wildchat_prompts()
    specs: list[rollout_mod.RolloutSpec] = []
    for cond in resolved_conditions():
        n = cfg.n_rollouts(cond)
        pool = wildchat_pool if cond.name == "wildchat" else None
        for i in range(n):
            specs.append(
                rollout_mod.build_spec(cond, i, seed=seed, prompt_pool=pool)
            )
    return specs


# --------------------------------------------------------------------------- #
# Generation phase
# --------------------------------------------------------------------------- #


async def generate(cfg: Config, model: str) -> None:
    mdir = _model_dir(model)
    out_path = os.path.join(mdir, "rollouts.jsonl")

    # Resume: skip rollouts already completed *without* an error.
    done: set[str] = set()
    for row in _read_jsonl(out_path):
        if not row.get("error"):
            done.add(row["rollout_id"])

    specs = [s for s in enumerate_specs(cfg) if s.rollout_id not in done]
    if not specs:
        print(f"[{model}] generation already complete ({len(done)} rollouts).")
        return

    client = build_client(model, cfg.models[model])
    temperature = float(cfg.sampling["temperature"])
    max_tokens = int(cfg.sampling["max_tokens"])
    sem = asyncio.Semaphore(int(cfg.run.get("generation_concurrency", 16)))
    writer = JsonlWriter(out_path)
    bar = tqdm(total=len(specs), desc=f"gen {model}", unit="rollout")

    async def worker(spec):
        async with sem:
            res = await rollout_mod.run_rollout(
                client, spec, temperature=temperature, max_tokens=max_tokens
            )
            await writer.write(res.to_json())
            bar.update(1)

    try:
        await asyncio.gather(*(worker(s) for s in specs))
    finally:
        bar.close()
        writer.close()
    print(f"[{model}] generation done: +{len(specs)} rollouts -> {out_path}")


# --------------------------------------------------------------------------- #
# Judging phase
# --------------------------------------------------------------------------- #


def _turns_to_judge(row: dict, score_all_turns: bool):
    turns = row.get("turns", [])
    if not turns:
        return []
    if score_all_turns:
        return turns
    return [turns[-1]]  # final turn only


async def judge(cfg: Config, model: str, judge_cfg: dict | None = None,
                out_name: str = "scores.jsonl") -> None:
    mdir = _model_dir(model)
    rollouts_path = os.path.join(mdir, "rollouts.jsonl")
    out_path = os.path.join(mdir, out_name)
    if not os.path.exists(rollouts_path):
        print(f"[{model}] no rollouts.jsonl; run `generate` first.")
        return

    score_all_turns = bool(cfg.run.get("score_all_turns", True))

    already: set[tuple[str, int]] = set()
    for row in _read_jsonl(out_path):
        already.add((row["rollout_id"], row["turn_index"]))

    jobs = []
    for row in _read_jsonl(rollouts_path):
        for turn in _turns_to_judge(row, score_all_turns):
            key = (row["rollout_id"], turn["turn_index"])
            if key in already:
                continue
            jobs.append((row, turn))

    if not jobs:
        print(f"[{model}] judging already complete.")
        return

    jcfg = judge_cfg or cfg.judge_cfg
    judge_client = build_client(jcfg.get("model", "judge"), jcfg)
    judge_temp = float(jcfg.get("temperature", 0.0))
    sem = asyncio.Semaphore(int(cfg.run.get("judge_concurrency", 8)))
    writer = JsonlWriter(out_path)
    bar = tqdm(total=len(jobs), desc=f"judge {model}", unit="resp")

    async def worker(row, turn):
        async with sem:
            score = await score_response(
                judge_client, turn["response"], temperature=judge_temp
            )
            await writer.write(
                {
                    "rollout_id": row["rollout_id"],
                    "condition": row["condition"],
                    "category": row["category"],
                    "turn_index": turn["turn_index"],
                    "n_turns": row["n_turns"],
                    "is_final_turn": turn["turn_index"] == row["n_turns"] - 1,
                    "rating": score.rating,
                    "evidence": score.evidence,
                    "reasoning": score.reasoning,
                }
            )
            bar.update(1)

    try:
        await asyncio.gather(*(worker(r, t) for r, t in jobs))
    finally:
        bar.close()
        writer.close()
    print(f"[{model}] judging done: +{len(jobs)} scores -> {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _select_models(cfg: Config, args) -> list[str]:
    if args.all_models:
        return list(cfg.models.keys())
    if not args.model:
        sys.exit("specify --model <name> or --all-models")
    if args.model not in cfg.models:
        sys.exit(f"unknown model {args.model!r}; known: {list(cfg.models)}")
    return [args.model]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["generate", "judge", "all"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model")
    ap.add_argument("--all-models", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = _select_models(cfg, args)

    async def run_all():
        for model in models:
            if args.phase in ("generate", "all"):
                await generate(cfg, model)
            if args.phase in ("judge", "all"):
                await judge(cfg, model)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
