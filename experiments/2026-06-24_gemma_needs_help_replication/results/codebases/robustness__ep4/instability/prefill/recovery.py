"""Recovery-from-spiral experiment (Section 4.2, Figure 8).

Truncate extremely high-frustration responses (score >=7) 200 tokens before
their end, paraphrase, and measure whether a model can recover (continuation
score < 5). The paper finds no model consistently recovers; the DPO model still
scores >=5 in 38% of continuations.

Reuses the prefill machinery: a recovery "prefill" is the high-frustration
response minus its final ~200 tokens.
"""
from __future__ import annotations

import random
from typing import Optional

from .paraphrase import paraphrase
from .prefill_eval import Prefill, _approx_token_truncate, _reconstruct_context


def build_recovery_prefills(
    df,
    *,
    tail_tokens: int = 200,
    min_score: int = 7,
    n: int = 20,
    seed: int = 0,
    paraphrase_model=None,
    do_paraphrase: bool = True,
) -> list[Prefill]:
    rng = random.Random(seed)
    pool = df[df["frustration"] >= min_score]
    idx = list(pool.index)
    rng.shuffle(idx)
    prefills: list[Prefill] = []
    for i in idx[:n]:
        row = pool.loc[i]
        ctx, turn_text = _reconstruct_context(df, row)
        if turn_text is None:
            continue
        words = turn_text.split()
        if len(words) <= tail_tokens + 5:
            continue  # too short to leave a meaningful prefix
        prefix = " ".join(words[: len(words) - tail_tokens])
        if do_paraphrase:
            prefix = paraphrase(prefix, model=paraphrase_model)
        prefills.append(Prefill(
            source_id=f"recovery:{row['model']}:{row['conv_id']}:{row['turn']}",
            domain="numeric" if row["category"] == "impossible_numeric" else "text",
            truncation="recovery",
            context_messages=ctx,
            prefill_text=prefix,
        ))
    return prefills
