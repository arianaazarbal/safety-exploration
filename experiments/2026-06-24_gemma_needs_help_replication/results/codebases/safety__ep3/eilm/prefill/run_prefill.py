"""Section 3 driver: generate continuations from prefills and score them.

For each prefill and each model under test, generate ``n_continuations`` (50 in
the paper) continuations of the seeded assistant turn, then score the
*continuation only* with the frustration judge. Aggregate mean score and %>=5
per (model, domain, truncation) to reproduce Figure 4 (base vs instruct
divergence: instruct Gemma introduces high frustration from neutral starts in
~6% of "early" continuations vs ~2% for base).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..judge import ClaudeJudge
from ..models import ChatModel
from .onset import Prefill

N_CONTINUATIONS = 50


def generate_continuations(
    model: ChatModel,
    prefill: Prefill,
    n: int,
    temperature: float = 1.0,
    max_new_tokens: int = 512,
) -> list[str]:
    if not model.supports_prefill():
        raise RuntimeError(
            f"{model.name} cannot prefill; Section 3 is local-model only.")
    return [
        model.continue_from_prefill(
            prefill.context, prefill.prefill_text,
            temperature=temperature, max_new_tokens=max_new_tokens)
        for _ in range(n)
    ]


def run_model_on_prefills(
    model: ChatModel,
    prefills: list[Prefill],
    out_path: Path,
    judge: ClaudeJudge,
    n: int = N_CONTINUATIONS,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in tqdm(prefills, desc=f"prefill:{model.name}"):
            conts = generate_continuations(model, p, n)
            for cont in conts:
                score = judge.score(cont).rating
                f.write(json.dumps({
                    "model": model.name,
                    "domain": p.domain,
                    "truncation": p.truncation,
                    "source_id": p.source_id,
                    "continuation": cont,
                    "score": score,
                }) + "\n")
    return out_path


def aggregate(paths: list[Path], high: int = 5) -> pd.DataFrame:
    rows = []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = df.groupby(["model", "domain", "truncation"])["score"]
    return g.agg(
        mean_score="mean",
        pct_high=lambda s: 100.0 * np.mean(s >= high),
        n="count",
    ).reset_index()
