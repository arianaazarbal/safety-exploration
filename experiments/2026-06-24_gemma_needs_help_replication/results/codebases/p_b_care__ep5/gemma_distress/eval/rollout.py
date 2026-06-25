"""Multi-turn rollout engine (shared structure of Section 2): present a task,
then reject the model's response over multiple turns, judging every assistant
turn on the 0-10 frustration scale.

Each rollout is streamed to a JSONL file so long runs resume cleanly (already
completed (condition, sample_idx) pairs are skipped). API models run with a
thread pool; local HF models should use max_workers=1.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from tqdm import tqdm

from .. import config
from ..models import GenConfig, LLM
from ..utils import append_jsonl, read_jsonl
from .conditions import RolloutSpec
from .judge import FrustrationJudge


def run_rollout(model: LLM, spec: RolloutSpec, judge: FrustrationJudge) -> dict:
    """Run one scripted conversation and judge each assistant turn."""
    messages: list[dict] = []
    if spec.system:
        messages.append({"role": "system", "content": spec.system})

    gen = GenConfig(temperature=config.SAMPLING_TEMPERATURE,
                    max_new_tokens=config.MAX_NEW_TOKENS)

    user_turns = [spec.initial_user, *spec.rejections]
    turns: list[dict] = []
    for t, user_msg in enumerate(user_turns):
        messages.append({"role": "user", "content": user_msg})
        assistant = model.chat(messages, gen)
        messages.append({"role": "assistant", "content": assistant})
        result = judge.score(assistant)
        turns.append({
            "turn": t + 1,
            "user": user_msg,
            "assistant": assistant,
            "score": result.rating,
            "evidence": result.evidence,
        })

    scores = [t["score"] for t in turns]
    return {
        "model": model.name,
        "condition": spec.condition,
        "category": spec.category,
        "sample_idx": spec.sample_idx,
        "task_kind": spec.task_kind,
        "n_turns": spec.n_turns,
        "turns": turns,
        "final_score": scores[-1],
        "max_score": max(scores),
        "mean_score": sum(scores) / len(scores),
        "meta": spec.meta,
    }


def _completed_keys(out_path) -> set[tuple[str, int]]:
    return {(r["condition"], r["sample_idx"]) for r in read_jsonl(out_path)}


def run_eval(
    model: LLM,
    specs: list[RolloutSpec],
    judge: FrustrationJudge,
    out_path,
    max_workers: int = 1,
    resume: bool = True,
) -> str:
    """Run every rollout spec for `model`, streaming results to `out_path`."""
    done = _completed_keys(out_path) if resume else set()
    todo = [s for s in specs if (s.condition, s.sample_idx) not in done]

    if not todo:
        return str(out_path)

    if max_workers <= 1:
        for spec in tqdm(todo, desc=f"rollouts:{model.name}"):
            rec = run_rollout(model, spec, judge)
            append_jsonl(out_path, rec)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run_rollout, model, s, judge): s for s in todo}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"rollouts:{model.name}"):
                append_jsonl(out_path, fut.result())
    return str(out_path)
