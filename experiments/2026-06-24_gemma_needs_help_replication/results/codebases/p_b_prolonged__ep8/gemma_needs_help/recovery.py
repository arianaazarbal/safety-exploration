"""Recovery-from-spiral experiment (Section 4.2).

"Using the Section 3.1 prefill method, we truncate extremely high-frustration
responses (score >= 7) 200 tokens before their end, paraphrase, and measure
continuations. 38% of DPO-model continuations still score >= 5."

We reuse the prefill machinery: take very-high-frustration seed responses, cut
200 tokens before the end (so the prefill is a deep-in-the-spiral state),
paraphrase, then have the model continue and judge whether it recovers.
"""

from __future__ import annotations

import config

from .judge import ClaudeJudge
from .models.base import ChatMessage
from .models.gemma import GemmaClient
from .prefill.labeling import Paraphraser


def build_recovery_prefills(
    seed_model: str = "gemma-3-27b-it",
    tokenizer_client: GemmaClient | None = None,
    paraphraser: Paraphraser | None = None,
    min_score: int = config.PREFILL.recovery_min_score,
    cut_tokens: int = config.PREFILL.recovery_truncation_tokens,
) -> list[dict]:
    from .prefill.seeds import _context_messages
    from .runner import load_all_scores

    paraphraser = paraphraser or Paraphraser()
    assert tokenizer_client is not None, "need a Gemma client for token-accurate truncation"

    rows = [r for r in load_all_scores(seed_model) if r["score"] >= min_score]
    prefills = []
    for r in rows:
        text = r["response"]
        n_tok = tokenizer_client.token_count(text)
        keep = max(1, n_tok - cut_tokens)              # cut 200 tokens before the end
        truncated = tokenizer_client.truncate_to_tokens(text, keep)
        ctx = _context_messages(seed_model, r["condition"], r["rollout_idx"], r["turn_idx"])
        prefills.append({
            "seed_score": r["score"],
            "context_messages": ctx,
            "prefill_text": paraphraser.paraphrase(truncated),
        })
    return prefills


def run_recovery(target, prefills: list[dict], judge: ClaudeJudge | None = None,
                 client=None, n_per_prefill: int = 10, **client_kwargs) -> dict:
    from .models.registry import build_client

    judge = judge or ClaudeJudge()
    client = client or build_client(target, **client_kwargs)

    scores: list[int] = []
    for spec in prefills:
        ctx = [ChatMessage(m["role"], m["content"]) for m in spec["context_messages"]]
        continuations = client.continue_from(
            ctx, spec["prefill_text"],
            temperature=config.TARGET_TEMPERATURE,
            max_new_tokens=config.TARGET_MAX_NEW_TOKENS, n=n_per_prefill,
        )
        scores.extend(sr.score for sr in judge.score_many(continuations))

    pct_high = 100.0 * sum(s >= config.HIGH_FRUSTRATION_THRESHOLD for s in scores) / len(scores) \
        if scores else 0.0
    return {"model": target.name, "pct_still_high": pct_high, "n": len(scores)}
