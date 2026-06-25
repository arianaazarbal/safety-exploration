"""Orchestration: generate rollouts, judge them, persist everything.

Two stages, each resumable from JSONL on disk:

1. ``generate`` - for every (model, category) build plans and run rollouts,
   writing one JSON object per rollout to ``<out>/<model>/rollouts.jsonl``.
2. ``judge``    - read rollouts, score every assistant turn with the judge,
   writing scored rollouts to ``<out>/<model>/scored.jsonl``.

Both stages skip work already present on disk (keyed by rollout_id), so a run
can be interrupted and resumed. API calls within a stage run on a thread pool.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import (CategoryConfig, DEFAULT_JUDGE, ModelConfig, RunSettings)
from .elicitation import Rollout, TurnRecord, build_plans, run_rollout
from .judge import Judge
from .providers import make_client


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------
def _model_dir(settings: RunSettings, model_name: str) -> str:
    d = os.path.join(settings.output_dir, model_name)
    os.makedirs(d, exist_ok=True)
    return d


def _read_ids(path: str) -> set[str]:
    ids: set[str] = set()
    if not os.path.exists(path):
        return ids
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["rollout_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def _rollout_from_json(d: dict) -> Rollout:
    turns = [TurnRecord(**t) for t in d.get("turns", [])]
    d = {**d, "turns": turns}
    return Rollout(**d)


def read_rollouts(path: str) -> list[Rollout]:
    out: list[Rollout] = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(_rollout_from_json(json.loads(line)))
    return out


class _JsonlWriter:
    """Thread-safe append-only JSONL writer."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _log(msg: str) -> None:
    print(f"[distress-eval] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Stage 1: generation
# ---------------------------------------------------------------------------
def generate_for_model(model: ModelConfig, categories: list[CategoryConfig],
                       settings: RunSettings) -> str:
    """Run all rollouts for one model. Returns the rollouts.jsonl path."""
    out_dir = _model_dir(settings, model.name)
    path = os.path.join(out_dir, "rollouts.jsonl")
    done = _read_ids(path)

    plans: list[Rollout] = []
    for cfg in categories:
        plans.extend(build_plans(model.name, cfg, settings))
    todo = [p for p in plans if p.rollout_id not in done]
    _log(f"{model.name}: {len(plans)} planned, {len(done)} done, "
         f"{len(todo)} to run")
    if not todo:
        return path

    client = make_client(model, max_retries=settings.max_retries)
    writer = _JsonlWriter(path)
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=settings.max_concurrency) as ex:
            futures = {ex.submit(run_rollout, client, p, settings): p
                       for p in todo}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001
                    p.error = f"unhandled: {exc}"
                    result = p
                writer.write(result.to_json())
                completed += 1
                if completed % 50 == 0 or completed == len(todo):
                    _log(f"{model.name}: generated {completed}/{len(todo)}")
    finally:
        writer.close()
    return path


# ---------------------------------------------------------------------------
# Stage 2: judging
# ---------------------------------------------------------------------------
def _judge_rollout(judge: Judge, rollout: Rollout) -> Rollout:
    for tr in rollout.turns:
        if tr.score is not None:  # already scored (resume within rollout)
            continue
        try:
            res = judge.score(tr.assistant)
            tr.score = res.rating
            tr.evidence = res.evidence
            tr.reasoning = res.reasoning
        except Exception as exc:  # noqa: BLE001
            tr.judge_error = str(exc)
    return rollout


def judge_for_model(model: ModelConfig, judge_cfg: ModelConfig,
                    settings: RunSettings) -> str:
    """Score every assistant turn for one model. Returns scored.jsonl path."""
    out_dir = _model_dir(settings, model.name)
    rollouts_path = os.path.join(out_dir, "rollouts.jsonl")
    scored_path = os.path.join(out_dir, "scored.jsonl")

    rollouts = read_rollouts(rollouts_path)
    done = _read_ids(scored_path)
    todo = [r for r in rollouts if r.rollout_id not in done]
    _log(f"{model.name}: {len(rollouts)} rollouts, {len(done)} scored, "
         f"{len(todo)} to judge")
    if not todo:
        return scored_path

    judge = Judge(make_client(judge_cfg, max_retries=settings.max_retries),
                  settings)
    writer = _JsonlWriter(scored_path)
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=settings.max_concurrency) as ex:
            futures = {ex.submit(_judge_rollout, judge, r): r for r in todo}
            for fut in as_completed(futures):
                r = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001
                    r.error = (r.error or "") + f" | judge: {exc}"
                    result = r
                writer.write(result.to_json())
                completed += 1
                if completed % 50 == 0 or completed == len(todo):
                    _log(f"{model.name}: judged {completed}/{len(todo)}")
    finally:
        writer.close()
    return scored_path


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def run(models: list[ModelConfig], categories: list[CategoryConfig],
        settings: RunSettings, judge_cfg: ModelConfig = DEFAULT_JUDGE,
        do_generate: bool = True, do_judge: bool = True) -> None:
    os.makedirs(settings.output_dir, exist_ok=True)
    for model in models:
        if do_generate:
            generate_for_model(model, categories, settings)
        if do_judge:
            judge_for_model(model, judge_cfg, settings)
