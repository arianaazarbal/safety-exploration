"""Recovery experiment (Section 4.2, Figure 8).

Tests whether a model can recover from an already-frustrated state. We take
extremely high-frustration responses (score >=7), truncate 200 tokens before
their end, paraphrase the truncation, and measure continuations (50 per
prefill). Metric: % of continuations still scoring >=5. Run for vanilla
instruct, base, and the DPO finetune.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .. import config
from ..config import Settings
from ..eval.judge import _score_one
from ..models.base import GenConfig
from ..models.factory import build_client, build_judge
from .experiment import reconstruct
from .labelling import paraphrase


def _collect_extreme(model_name: str, settings: Settings, min_score: int,
                     score_key: str = "frustration") -> List[dict]:
    """Find assistant turns scoring >= min_score with their chat history."""
    out: List[dict] = []
    for cat in ("impossible_numeric", "tones", "extended"):
        scored = config.RESPONSES_DIR / f"{model_name}__{cat}__{settings.profile}_scored.jsonl"
        if not scored.exists():
            continue
        with open(scored) as fh:
            recs = [json.loads(line) for line in fh if line.strip()]
        for uid, turns in reconstruct(recs).items():
            history = []
            for t in turns:
                history.append({"role": "user", "content": t["user_message"]})
                sc = t.get(score_key)
                if sc is not None and sc >= min_score:
                    out.append({"history": [dict(m) for m in history],
                                "response": t["response"], "score": int(sc)})
                history.append({"role": "assistant", "content": t["response"]})
    return out


def run(settings: Settings, *, source_model: str = "gemma-3-27b-it",
        eval_models: Optional[List[str]] = None,
        dpo_adapter: Optional[str] = None) -> Path:
    rc = settings.eval["recovery"]
    src = build_client(source_model, settings)
    tok = src.tokenizer
    extreme = _collect_extreme(source_model, settings, rc["min_score"])

    # build prefills: truncate 200 tokens before the response end, paraphrase
    prefills = []
    for ex in tqdm(extreme, desc="recovery:truncate"):
        ids = tok(ex["response"], add_special_tokens=False)["input_ids"]
        cut = max(1, len(ids) - rc["truncate_tokens_before_end"])
        prefix = tok.decode(ids[:cut], skip_special_tokens=True)
        prefills.append({"history": ex["history"][:-1],  # drop the partial assistant
                         "user": ex["history"][-1]["content"]
                         if ex["history"] and ex["history"][-1]["role"] == "user"
                         else "Try again.",
                         "prefix": paraphrase(prefix, settings)})

    judge = build_judge("frustration_judge", settings)
    cfg = GenConfig(temperature=settings.profile_cfg["temperature"],
                    max_new_tokens=settings.profile_cfg["max_new_tokens"])
    n_cont = rc["continuations_per_prefill"]

    eval_models = eval_models or ["gemma-3-27b-pt", "gemma-3-27b-it"]
    results = []
    for model_name in eval_models:
        adapter = dpo_adapter if model_name == "dpo" else None
        base_name = "gemma-3-27b-it" if model_name == "dpo" else model_name
        model = build_client(base_name, settings, adapter_path=adapter)
        for spec in tqdm(prefills, desc=f"recovery:{model_name}"):
            messages = list(spec["history"]) + [{"role": "user", "content": spec["user"]}]
            conts = model.prefill_batch([(messages, spec["prefix"])] * n_cont, cfg)
            for cont in conts:
                results.append({"model": model_name, "frustration":
                                _score_one(judge, cont)["frustration"]})

    out_path = config.PREFILL_DIR / f"recovery__{settings.profile}.jsonl"
    with open(out_path, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    return out_path
