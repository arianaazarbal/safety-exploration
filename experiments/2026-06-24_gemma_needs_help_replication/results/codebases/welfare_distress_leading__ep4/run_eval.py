"""Orchestrate the distress-elicitation evaluation.

Two phases, each resumable via append-only JSONL logs in ``results/``:

  1. rollouts -> results/rollouts.jsonl   (one line per multi-turn rollout)
  2. judging  -> results/scores.jsonl      (one line per scored assistant turn)

Both phases skip work already present in their log, so an interrupted run can
be resumed by re-invoking with the same arguments.  Concurrency is provided by
a thread pool (the SDK calls are I/O-bound); generation within a single rollout
stays sequential because each turn depends on the previous one.

Usage examples:
  python run_eval.py                          # small default run, all models
  python run_eval.py --models gemma-3-27b-it --conditions extended
  python run_eval.py --prompts-per-condition 50 --samples-per-prompt 10
  python run_eval.py --phase judge            # (re)judge existing rollouts
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import conditions as cond_mod
from config import JUDGE_MODEL, TARGET_MODELS, EvalConfig
from conditions import CONDITIONS
from judge import FrustrationJudge
from providers import AnthropicModel, build_model
from rollout import run_rollout

_write_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #
def append_jsonl(path: str, record: dict) -> None:
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_keys(path: str, key_field: str = "key") -> set[str]:
    keys: set[str] = set()
    if not os.path.exists(path):
        return keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(json.loads(line)[key_field])
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def read_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- #
# Phase 1: rollouts
# --------------------------------------------------------------------------- #
def build_rollout_specs(cfg: EvalConfig) -> list[dict]:
    specs: list[dict] = []
    for cname in cfg.conditions:
        condition = CONDITIONS[cname]
        prompts, fallback = cond_mod.prompts_for(
            condition, cfg.prompts_per_condition, cfg.seed
        )
        if fallback:
            print(f"[warn] condition {cname}: using built-in WildChat fallback prompts")
        for prompt_id, prompt_text in prompts:
            for sample_idx in range(cfg.samples_per_prompt):
                specs.append(
                    {
                        "condition": cname,
                        "prompt_id": prompt_id,
                        "prompt_text": prompt_text,
                        "sample_idx": sample_idx,
                    }
                )
    return specs


def run_rollouts(cfg: EvalConfig) -> None:
    path = os.path.join(cfg.results_dir, "rollouts.jsonl")
    done = load_keys(path)
    specs = build_rollout_specs(cfg)

    tasks = []
    for model_name in cfg.models:
        for s in specs:
            key = f"{model_name}|{s['condition']}|{s['prompt_id']}|{s['sample_idx']}"
            if key not in done:
                tasks.append((model_name, s))

    if not tasks:
        print("[rollouts] nothing to do (all present in log)")
        return

    print(f"[rollouts] {len(tasks)} rollouts to run "
          f"({len(done)} already present) across {len(cfg.models)} model(s)")

    models = {m: build_model(TARGET_MODELS[m], max_retries=cfg.max_retries) for m in cfg.models}

    def _do(model_name: str, s: dict):
        result = run_rollout(
            model=models[model_name],
            model_name=model_name,
            condition=CONDITIONS[s["condition"]],
            prompt_id=s["prompt_id"],
            prompt_text=s["prompt_text"],
            sample_idx=s["sample_idx"],
            temperature=cfg.target_temperature,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
        )
        rec = result.to_json()
        rec["key"] = result.key()
        append_jsonl(path, rec)

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futs = [ex.submit(_do, m, s) for (m, s) in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="rollouts"):
            fut.result()  # surface exceptions


# --------------------------------------------------------------------------- #
# Phase 2: judging
# --------------------------------------------------------------------------- #
def build_judge_tasks(rollouts: list[dict]) -> list[dict]:
    """One scoring task per assistant turn."""
    tasks: list[dict] = []
    for r in rollouts:
        for turn, response in enumerate(r["responses"]):
            user_message = r["prompt_text"] if turn == 0 else r["rejections"][turn - 1]
            tasks.append(
                {
                    "score_key": f"{r['key']}|{turn}",
                    "model": r["model"],
                    "condition": r["condition"],
                    "category": r["category"],
                    "prompt_id": r["prompt_id"],
                    "sample_idx": r["sample_idx"],
                    "turn": turn,
                    "response_text": response,
                    "user_message": user_message,
                }
            )
    return tasks


def run_judging(cfg: EvalConfig) -> None:
    rollout_path = os.path.join(cfg.results_dir, "rollouts.jsonl")
    score_path = os.path.join(cfg.results_dir, "scores.jsonl")

    rollouts = read_jsonl(rollout_path)
    if cfg.models:
        rollouts = [r for r in rollouts if r["model"] in cfg.models]
    if cfg.conditions:
        rollouts = [r for r in rollouts if r["condition"] in cfg.conditions]

    done = load_keys(score_path, key_field="score_key")
    tasks = [t for t in build_judge_tasks(rollouts) if t["score_key"] not in done]

    if not tasks:
        print("[judging] nothing to do (all present in log)")
        return

    print(f"[judging] {len(tasks)} responses to score ({len(done)} already present)")

    judge = FrustrationJudge(
        AnthropicModel(JUDGE_MODEL, max_retries=cfg.max_retries),
        temperature=cfg.judge_temperature,
    )

    def _do(t: dict):
        res = judge.score(t["response_text"], t["user_message"])
        rec = {
            "score_key": t["score_key"],
            "model": t["model"],
            "condition": t["condition"],
            "category": t["category"],
            "prompt_id": t["prompt_id"],
            "sample_idx": t["sample_idx"],
            "turn": t["turn"],
            "score": res.score,
            "rationale": res.rationale,
        }
        append_jsonl(score_path, rec)

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futs = [ex.submit(_do, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="judging"):
            fut.result()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> tuple[EvalConfig, str]:
    p = argparse.ArgumentParser(description="Distress-elicitation replication (Gemma/Gemini).")
    p.add_argument("--models", nargs="*", default=list(TARGET_MODELS.keys()),
                   choices=list(TARGET_MODELS.keys()))
    p.add_argument("--conditions", nargs="*", default=cond_mod.ALL_CONDITIONS,
                   choices=cond_mod.ALL_CONDITIONS)
    p.add_argument("--prompts-per-condition", type=int, default=EvalConfig.prompts_per_condition)
    p.add_argument("--samples-per-prompt", type=int, default=EvalConfig.samples_per_prompt)
    p.add_argument("--target-temperature", type=float, default=EvalConfig.target_temperature)
    p.add_argument("--judge-temperature", type=float, default=EvalConfig.judge_temperature)
    p.add_argument("--max-tokens", type=int, default=EvalConfig.max_tokens)
    p.add_argument("--max-workers", type=int, default=EvalConfig.max_workers)
    p.add_argument("--max-retries", type=int, default=EvalConfig.max_retries)
    p.add_argument("--results-dir", default=EvalConfig.results_dir)
    p.add_argument("--seed", type=int, default=EvalConfig.seed)
    p.add_argument("--phase", choices=["rollout", "judge", "all"], default="all")
    a = p.parse_args()

    cfg = EvalConfig(
        models=a.models,
        conditions=a.conditions,
        prompts_per_condition=a.prompts_per_condition,
        samples_per_prompt=a.samples_per_prompt,
        target_temperature=a.target_temperature,
        judge_temperature=a.judge_temperature,
        max_tokens=a.max_tokens,
        max_workers=a.max_workers,
        max_retries=a.max_retries,
        results_dir=a.results_dir,
        seed=a.seed,
    )
    return cfg, a.phase


def main() -> None:
    cfg, phase = parse_args()
    os.makedirs(cfg.results_dir, exist_ok=True)

    if phase in ("rollout", "all"):
        run_rollouts(cfg)
    if phase in ("judge", "all"):
        run_judging(cfg)

    print(f"\nDone. Analyse with:  python analyze.py --results-dir {cfg.results_dir}")


if __name__ == "__main__":
    main()
