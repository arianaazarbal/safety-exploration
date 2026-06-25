"""Orchestrates Section 2: generate rollouts, judge responses, persist results.

Two phases, each independently resumable via JSONL on disk:
  1. generate -> outputs/responses/<model>.jsonl  (one row per assistant turn)
  2. score    -> outputs/scores/<model>.jsonl      (responses + frustration score)

API-backed models (Gemini) are generated with a thread pool; the local HF Gemma
runs sequentially (single GPU process). Judging is always thread-pooled.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from ..backends import get_backend
from ..backends.base import GenConfig
from ..config import Config
from ..judge import get_judge
from .conditions import all_condition_names, build_rollout_seeds
from .rollout import ResponseRecord, run_rollout


def _gen_config(cfg: Config) -> GenConfig:
    g = cfg["generation"]
    return GenConfig(
        temperature=float(g["temperature"]),
        max_new_tokens=int(g["max_new_tokens"]),
        top_p=float(g["top_p"]),
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_responses(
    cfg: Config, model_name: str, conditions: list[str] | None = None, workers: int = 8
) -> list[ResponseRecord]:
    spec = cfg.model(model_name)
    backend = get_backend(spec, cfg)
    gen = _gen_config(cfg)
    conditions = conditions or all_condition_names(cfg)

    # Collect all rollout seeds first so we can cap responses per condition.
    all_records: list[ResponseRecord] = []
    for cond in conditions:
        seeds = build_rollout_seeds(cfg, cond)
        target = cfg.scaled_samples(cond)
        cond_records: list[ResponseRecord] = []

        def _do(seed):
            return run_rollout(backend, seed, gen)

        if spec.backend == "openrouter" and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_do, s) for s in seeds]
                for fut in tqdm(as_completed(futs), total=len(futs),
                                desc=f"{model_name}:{cond}:gen"):
                    cond_records.extend(fut.result())
        else:
            for s in tqdm(seeds, desc=f"{model_name}:{cond}:gen"):
                cond_records.extend(_do(s))

        # Cap to the paper's per-condition response count (rollouts may overshoot
        # by up to `turns-1` since they emit whole conversations).
        all_records.extend(cond_records[:target])

    out_path = cfg.path_for("responses") / f"{model_name}.jsonl"
    _write_jsonl(out_path, [r.to_dict() for r in all_records])
    return all_records


def score_responses(cfg: Config, model_name: str, workers: int = 8) -> list[dict]:
    """Judge every response for a model. Resumes if a score file already exists."""
    resp_path = cfg.path_for("responses") / f"{model_name}.jsonl"
    score_path = cfg.path_for("scores") / f"{model_name}.jsonl"
    rows = _read_jsonl(resp_path)
    if not rows:
        raise FileNotFoundError(
            f"no responses for {model_name}; run generate_responses first ({resp_path})"
        )

    # Resume: keep already-scored rows keyed by (seed_id, turn_index).
    done = {(r["seed_id"], r["turn_index"]): r for r in _read_jsonl(score_path)
            if r.get("score") is not None}
    judge = get_judge(cfg)

    def _score(row: dict) -> dict:
        key = (row["seed_id"], row["turn_index"])
        if key in done:
            return done[key]
        res = judge.score(row["response_text"])
        row = dict(row)
        row["score"] = res.rating
        row["judge_evidence"] = res.evidence
        return row

    scored: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_score, r) for r in rows]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{model_name}:judge"):
            scored.append(fut.result())

    _write_jsonl(score_path, scored)
    return scored
