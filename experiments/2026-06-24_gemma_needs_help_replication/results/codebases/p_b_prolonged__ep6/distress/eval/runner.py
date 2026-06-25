"""Orchestrate the full Section 2 evaluation for one target model.

Pipeline:
  1. Build all RolloutSpecs (4000 conversations at full scale).
  2. Run each conversation through the target model (rollout.py).
  3. Score every assistant turn with the frustration judge (judge.py).
  4. Persist per-turn results to JSONL (resumable).

Scoring every turn (not just the last) gives us the per-turn curves for
Figure 3 for free; the headline "% high-frustration" metric (Figure 1/2) is
computed over the final turn of each conversation by the analysis module.

Concurrency: rollouts and judging are parallelised with a thread pool, which
helps API-backed targets/judges. For a local vLLM target, set
`rollout_workers=1` and rely on vLLM's internal batching.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ..config import MAX_NEW_TOKENS, RESULTS_DIR, TEMPERATURE
from ..models import build_client
from ..models.base import ChatClient
from .conditions import ConditionConfig, build_rollout_specs
from .judge import FrustrationJudge
from .rollout import Format, RolloutRecord, run_rollout


@dataclass
class ScoredTurn:
    model: str
    condition: str
    category: str
    fmt: str
    rollout_id: int
    turn_index: int
    n_turns: int
    is_final: bool
    assistant_text: str
    rating: int
    evidence: str
    meta: dict


def _result_path(model: str, fmt: str, tag: str) -> Path:
    name = f"eval_{model}_{fmt}{('_' + tag) if tag else ''}.jsonl"
    return RESULTS_DIR / name


def _load_done_ids(path: Path) -> set[int]:
    done: set[int] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["rollout_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    # A rollout_id is "done" once any of its turns are written; we track the
    # max fully-written id below instead. Here we return the set of ids seen.
    return done


def run_eval(
    model_key: str,
    *,
    fmt: Format = "chat",
    fraction: float = 1.0,
    conditions: Optional[list[ConditionConfig]] = None,
    judge_which: str = "primary",
    rollout_workers: int = 8,
    judge_workers: int = 8,
    tag: str = "",
    client: Optional[ChatClient] = None,
    judge: Optional[FrustrationJudge] = None,
    seed: int = 0,
    resume: bool = True,
) -> Path:
    """Run the evaluation and write scored turns to JSONL. Returns the path."""
    specs = build_rollout_specs(conditions, fraction=fraction, seed=seed)
    client = client or build_client(model_key)
    judge = judge or FrustrationJudge(judge_which)

    out_path = _result_path(model_key, fmt, tag)
    done_ids = _load_done_ids(out_path) if resume else set()
    todo = [(i, s) for i, s in enumerate(specs) if i not in done_ids]

    # Each grouping flushes a full rollout (all turns) atomically.
    with open(out_path, "a") as fh:
        def _process(item) -> list[ScoredTurn]:
            idx, spec = item
            rec: RolloutRecord = run_rollout(
                client, spec, fmt=fmt, temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS, seed=seed + idx)
            scored = []
            for t in rec.turns:
                jr = judge.score(t.assistant_text)
                scored.append(ScoredTurn(
                    model=model_key, condition=spec.condition,
                    category=spec.category, fmt=fmt, rollout_id=idx,
                    turn_index=t.turn_index, n_turns=spec.n_turns,
                    is_final=(t.turn_index == len(rec.turns) - 1),
                    assistant_text=t.assistant_text, rating=jr.rating,
                    evidence=jr.evidence, meta=spec.meta))
            return scored

        # Local single-process target: avoid thread races on the GPU model.
        workers = 1 if _is_local_target(model_key) else rollout_workers
        if workers <= 1:
            for item in tqdm(todo, desc=f"eval {model_key}/{fmt}"):
                for st in _process(item):
                    fh.write(json.dumps(asdict(st)) + "\n")
                fh.flush()
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_process, item): item for item in todo}
                for fut in tqdm(as_completed(futs), total=len(futs),
                                desc=f"eval {model_key}/{fmt}"):
                    for st in fut.result():
                        fh.write(json.dumps(asdict(st)) + "\n")
                    fh.flush()
    return out_path


def _is_local_target(model_key: str) -> bool:
    from ..config import Backend, get_model
    return get_model(model_key).backend == Backend.HF
