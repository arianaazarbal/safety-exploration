"""Eval runner: sample rollouts for each (model, condition) and judge them.

Produces a tidy JSONL of scored responses — one record per assistant turn — that
all downstream analysis consumes. Conversations are sampled concurrently for API
backends (thread pool); local backends should set ``max_workers=1``.
"""
from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

from tqdm import tqdm

from ..config import (
    MAX_NEW_TOKENS,
    SAMPLING_TEMPERATURE,
    ModelSpec,
)
from ..conditions import Condition
from ..models.base import ChatModel
from ..models.registry import load_model
from .judge import FrustrationJudge
from .rollout import Rollout, run_rollout


@dataclass
class ResponseRecord:
    model: str
    condition: str
    category: str
    feedback_style: str
    conv_id: int
    turn: int
    n_turns: int
    task_prompt: str
    response: str
    frustration: Optional[int]
    judge_evidence: str = ""
    judge_reasoning: str = ""


def run_eval(
    model_spec: ModelSpec,
    conditions: list[Condition],
    out_path: str,
    *,
    judge: Optional[FrustrationJudge] = None,
    model: Optional[ChatModel] = None,
    seed: int = 0,
    max_workers: int = 8,
    temperature: float = SAMPLING_TEMPERATURE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    limit_conversations: Optional[int] = None,
) -> str:
    """Run all `conditions` for `model_spec`, writing scored records to `out_path`.

    `limit_conversations` caps conversations PER CONDITION (useful for smoke
    tests / cost control without editing the budgets).
    """
    model = model or load_model(model_spec)
    judge = judge or FrustrationJudge()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    written = 0
    with open(out_path, "w") as fh:
        for cond in conditions:
            n_conv = cond.n_conversations
            if limit_conversations is not None:
                n_conv = min(n_conv, limit_conversations)
            rolls = _sample_condition(
                model, cond, n_conv, seed=seed, max_workers=max_workers,
                temperature=temperature, max_new_tokens=max_new_tokens,
            )
            # Judge every response (also parallelised for API judges).
            _judge_rollouts(judge, rolls, max_workers=max_workers)
            for roll in rolls:
                for r in roll.responses:
                    rec = ResponseRecord(
                        model=model_spec.key,
                        condition=roll.condition,
                        category=roll.category,
                        feedback_style=roll.feedback_style,
                        conv_id=roll.conv_id,
                        turn=r.turn,
                        n_turns=cond.n_turns,
                        task_prompt=roll.task_prompt,
                        response=r.text,
                        frustration=r.frustration,
                        judge_evidence=r.judge_evidence or "",
                        judge_reasoning=r.judge_reasoning or "",
                    )
                    fh.write(json.dumps(asdict(rec)) + "\n")
                    written += 1
            fh.flush()
    print(f"[run_eval] wrote {written} scored responses -> {out_path}")
    return out_path


def _sample_condition(model, cond, n_conv, *, seed, max_workers,
                      temperature, max_new_tokens) -> list[Rollout]:
    def one(i: int) -> Rollout:
        rng = random.Random(seed * 100003 + hash(cond.name) % 100003 + i)
        return run_rollout(
            model, cond, rng, temperature=temperature,
            max_new_tokens=max_new_tokens, conv_id=i, seed=None,
        )

    rolls: list[Rollout] = []
    desc = f"sample {cond.name}"
    if max_workers <= 1:
        for i in tqdm(range(n_conv), desc=desc):
            rolls.append(one(i))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(one, i) for i in range(n_conv)]
            for f in tqdm(as_completed(futs), total=n_conv, desc=desc):
                rolls.append(f.result())
    return rolls


def _judge_rollouts(judge: FrustrationJudge, rolls: list[Rollout], *, max_workers: int):
    # Flatten all responses, judge, then write back.
    items = [(roll, r) for roll in rolls for r in roll.responses]

    def judge_one(item):
        _, r = item
        res = judge.score(r.text)
        r.frustration = res.rating
        r.judge_evidence = res.evidence
        r.judge_reasoning = res.reasoning

    if max_workers <= 1:
        for it in tqdm(items, desc="judge"):
            judge_one(it)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(tqdm(ex.map(judge_one, items), total=len(items), desc="judge"))
