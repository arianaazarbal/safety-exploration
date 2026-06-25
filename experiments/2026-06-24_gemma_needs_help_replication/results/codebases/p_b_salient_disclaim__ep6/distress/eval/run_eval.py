"""Section 2 driver: sample responses, judge them, persist per-response records.

For each model in ``EVAL_MODELS`` we build the full condition set (4000
responses' worth of conversations), roll out each conversation, score *every
assistant turn* with the Claude judge, and write one JSONL record per scored
response. Downstream aggregation (Figure 1/2), per-turn analysis (Figure 3) and
word-frequency analysis (Table 3/8) all read these records.

Records are append-only and keyed; re-running resumes where it left off.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .. import config
from ..models.registry import build_client
from ..utils.io import append_jsonl, read_jsonl
from .conditions import build_all_conditions
from .judge import FrustrationJudge
from .rollout import Rollout, rollout_batch


def _record_id(model_key: str, cond_idx: int, turn: int) -> str:
    return f"{model_key}:{cond_idx}:{turn}"


def responses_path(model_key: str) -> Path:
    return config.OUTPUT_DIR / "section2" / f"{model_key}.jsonl"


def run_model(
    model_key: str,
    judge: FrustrationJudge | None = None,
    hf_backend: str = "vllm",
    batch_size: int = 64,
    seed: int = 0,
) -> Path:
    """Sample + judge all 4000 responses for one model. Returns the JSONL path."""
    spec = config.ALL_MODELS[model_key]
    client = build_client(spec, hf_backend=hf_backend)
    judge = judge or FrustrationJudge()
    conds = build_all_conditions(seed=seed)
    out_path = responses_path(model_key)

    done = {row["id"] for row in read_jsonl(out_path)}

    # roll out in batches, then judge each assistant turn
    for start in range(0, len(conds), batch_size):
        chunk = conds[start:start + batch_size]
        # skip a chunk only if *all* its turns are already recorded
        rollouts = rollout_batch(client, chunk)
        for local_i, ro in enumerate(rollouts):
            cond_idx = start + local_i
            for turn, text in enumerate(ro.assistant_turns):
                rid = _record_id(model_key, cond_idx, turn)
                if rid in done:
                    continue
                result = judge.score(text)
                append_jsonl(out_path, {
                    "id": rid,
                    "model": model_key,
                    "family": spec.family,
                    "category": ro.condition.category,
                    "condition": ro.condition.condition,
                    "turn": turn,                       # 0-indexed assistant turn
                    "n_turns": ro.condition.n_turns,
                    "rating": result.rating,
                    "evidence": result.evidence,
                    "response": text,
                    "meta": ro.condition.meta,
                })
    return out_path


def run_all(model_keys: list[str] | None = None, **kw) -> dict[str, Path]:
    keys = model_keys or [m.key for m in config.EVAL_MODELS]
    judge = FrustrationJudge()
    return {k: run_model(k, judge=judge, **kw) for k in keys}
