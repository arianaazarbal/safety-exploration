"""Orchestration: sample rollouts for a model, score every response, persist.

The unit of work is one (model, eval-item) rollout. Results are streamed to JSONL
so long runs are resumable and inspectable. Each scored record is one assistant
response with its turn index and frustration rating.

Concurrency: API models (Gemini, the judge) are I/O-bound and run with a thread
pool; local HF models are GPU-bound and run serially (the pool size is forced to
1 for ``hf_local`` backends).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

import config
from . import conversation, evaluations, judge as judge_mod, providers


def _results_path(model_key: str, tag: str) -> Path:
    return config.RESULTS_DIR / f"{model_key}__{tag}.jsonl"


def _rollouts_path(model_key: str, tag: str) -> Path:
    return config.ROLLOUTS_DIR / f"{model_key}__{tag}.jsonl"


def _load_done_keys(path: Path) -> set[str]:
    """Resume support: which (item-uid) rollouts are already recorded."""
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                done.add(json.loads(line)["uid"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _item_uid(model_key: str, item: evaluations.EvalItem, idx: int) -> str:
    return f"{model_key}:{item.category}:{item.condition}:{idx}"


def run_model_eval(
    model_key: str,
    *,
    tag: str = "section2",
    categories: list[str] | None = None,
    count_mode: str = "responses",
    judge_model: str | None = None,
    max_workers: int = 8,
    limit: int | None = None,
) -> Path:
    """Run the full Section-2 sweep for one model and score every response.

    Returns the path to the scored-results JSONL.
    """
    provider = providers.get_provider(model_key)
    judge = judge_mod.get_judge(judge_model)
    spec = config.MODELS[model_key]

    items = evaluations.build_all_eval_items(count_mode=count_mode, categories=categories)
    if limit:
        items = items[:limit]

    results_path = _results_path(model_key, tag)
    rollouts_path = _rollouts_path(model_key, tag)
    done = _load_done_keys(rollouts_path)

    # Local models: serialise generation (single GPU); APIs: parallelise.
    gen_workers = 1 if spec.backend == "hf_local" else max_workers

    def do_rollout(idx_item):
        idx, item = idx_item
        uid = _item_uid(model_key, item, idx)
        if uid in done:
            return None
        roll = conversation.run_rollout(
            provider,
            model_key=model_key, category=item.category, condition=item.condition,
            initial_prompt=item.initial_prompt, rejections=item.rejections,
            puzzle_key=item.puzzle_key, meta=item.meta or {},
        )
        return uid, roll

    rollouts: list[tuple[str, conversation.Rollout]] = []
    with ThreadPoolExecutor(max_workers=gen_workers) as pool:
        futs = [pool.submit(do_rollout, (i, it)) for i, it in enumerate(items)]
        with open(rollouts_path, "a") as rf:
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc=f"rollouts:{model_key}"):
                out = fut.result()
                if out is None:
                    continue
                uid, roll = out
                rec = roll.to_json()
                rec["uid"] = uid
                rf.write(json.dumps(rec) + "\n")
                rf.flush()
                rollouts.append((uid, roll))

    # Score every response (judge is API-bound -> parallelise freely).
    def score_response(args):
        uid, roll, turn_idx, text = args
        res = judge.score(text)
        return {
            "uid": uid,
            "model_key": model_key,
            "category": roll.category,
            "condition": roll.condition,
            "puzzle_key": roll.puzzle_key,
            "turn": turn_idx,                # 0-indexed assistant turn
            "n_turns": len(roll.responses),
            "rating": res.rating,
            "evidence": res.evidence,
            "response_len_chars": len(text),
        }

    score_jobs = [
        (uid, roll, t, text)
        for uid, roll in rollouts
        for t, text in enumerate(roll.responses)
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as pool, open(results_path, "a") as out:
        futs = [pool.submit(score_response, job) for job in score_jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"judge:{model_key}"):
            out.write(json.dumps(fut.result()) + "\n")
            out.flush()

    return results_path


def run_all_section2(model_keys: list[str] | None = None, **kwargs) -> dict[str, Path]:
    keys = model_keys or config.SECTION2_MODELS
    return {k: run_model_eval(k, **kwargs) for k in keys}
