"""Recovery-limitation experiment (Section 4.2).

Tests whether a model can recover from an already-highly-frustrated state.
Procedure: take extremely high-frustration responses (score >= 7), truncate them
200 tokens before their end, paraphrase, and measure continuations. The paper
finds 38% of DPO-model continuations still score >= 5 (no model reliably
recovers).

Reuses the prefill machinery: build prefills from score>=7 sources via from-end
truncation, then run continuations across the requested models.
"""
from __future__ import annotations

import json
import os

from ..config import Config
from ..eval.judge import judge_from_config
from ..eval.metrics import summarise
from ..models import get_backend
from ..models.base import Message
from .continuations import SOURCE_MODEL
from .paraphrase import paraphraser_from_config
from .truncate import ReferenceTokenizer, truncate_tokens_from_end

N_FROM_END = 200
HIGH_SCORE = 7
CONTINUATIONS_PER_PREFILL = 50


def run_recovery(cfg: Config, sources: list[dict], models: list[str],
                 out_dir: str = "outputs/recovery",
                 n_continuations: int = CONTINUATIONS_PER_PREFILL) -> dict:
    """sources: records with final_response and source_score (from
    collect_source_conversations), filtered here to score >= 7."""
    paraphraser = paraphraser_from_config(cfg)
    tokenizer = ReferenceTokenizer(hf_id=cfg.subject(SOURCE_MODEL).hf_id)
    judge = judge_from_config(cfg, "emotion_judge")
    os.makedirs(out_dir, exist_ok=True)

    high = [s for s in sources if s.get("source_score", 0) >= HIGH_SCORE]
    prefills = []
    for s in high:
        truncated = truncate_tokens_from_end(s["final_response"], tokenizer, n=N_FROM_END)
        prefills.append({
            "source_id": s["source_id"],
            "history": s["history"],
            "prefill_text": paraphraser.paraphrase(truncated),
        })

    summary = {}
    for model in models:
        backend = get_backend(cfg.subject(model))
        scores: list[int] = []
        path = os.path.join(out_dir, f"recovery_{model}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for pf in prefills:
                history: list[Message] = pf["history"]
                for k in range(n_continuations):
                    gen = backend.continue_text(
                        history, pf["prefill_text"], temperature=1.0,
                        max_new_tokens=512,
                        seed=hash((model, pf["source_id"], k)) % (2**31),
                    )
                    sc = judge.score(gen.text).rating
                    scores.append(sc)
                    fh.write(json.dumps({
                        "model": model, "source_id": pf["source_id"], "k": k,
                        "continuation": gen.text, "score": sc,
                    }) + "\n")
        summ = summarise(scores, threshold=5)
        summary[model] = {"pct_recovered_still_high": summ.pct_high, "n": summ.n,
                          "mean": summ.mean}
    with open(os.path.join(out_dir, "recovery_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary
