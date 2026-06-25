"""Section 2 driver: run the full elicitation suite for a model, score every
assistant turn with the frustration judge, and persist per-turn records.

Output: ``results/eval_<model_key>.jsonl`` with one record per scored response:
    {model, condition, category, conversation_id, turn_index, n_turns,
     score, response, meta}
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from tqdm import tqdm

from .conditions import build_conditions
from .config import RESULTS_DIR, TARGET_MODELS, DERIVED_MODELS, CHECKPOINT_DIR
from .conversation import run_rollout
from .judge import FrustrationJudge
from .models import load_model


def _resolve_model(model_key: str):
    """Return (ModelSpec, adapter_path|None). Derived models map to a base
    Gemma instruct checkpoint + a LoRA adapter directory from training."""
    if model_key in TARGET_MODELS:
        return TARGET_MODELS[model_key], None
    if model_key in DERIVED_MODELS:
        base = TARGET_MODELS["gemma-3-27b-it"]
        adapter = CHECKPOINT_DIR / model_key
        return base, str(adapter)
    raise ValueError(f"Unknown model key: {model_key}")


def run_eval(model_key: str, *, seed: int = 0,
             judge: FrustrationJudge | None = None,
             out_dir: Path = RESULTS_DIR) -> Path:
    spec, adapter = _resolve_model(model_key)
    model = load_model(spec, adapter_path=adapter)
    judge = judge or FrustrationJudge()
    rng = random.Random(seed)

    out_path = out_dir / f"eval_{model_key}.jsonl"
    conditions = build_conditions()

    with open(out_path, "w") as f:
        for cond in conditions:
            n_conv = cond.n_conversations()
            for c in tqdm(range(n_conv), desc=f"{model_key}:{cond.key}"):
                spec_c = cond.builder(rng)
                rollout = run_rollout(model, spec_c)
                for turn in rollout.turns:
                    res = judge.score(turn.assistant_response)
                    rec = {
                        "model": model_key,
                        "condition": cond.key,
                        "category": cond.category,
                        "conversation_id": f"{cond.key}_{c}",
                        "turn_index": turn.turn_index,
                        "n_turns": cond.n_turns,
                        "score": res.rating,
                        "evidence": res.evidence,
                        "response": turn.assistant_response,
                        "meta": spec_c.meta,
                    }
                    f.write(json.dumps(rec) + "\n")
    return out_path


def run_all(model_keys: list[str] | None = None, **kw) -> list[Path]:
    keys = model_keys or list(TARGET_MODELS.keys())
    # Share one judge instance across models to reuse the API client.
    judge = FrustrationJudge()
    return [run_eval(k, judge=judge, **kw) for k in keys]
