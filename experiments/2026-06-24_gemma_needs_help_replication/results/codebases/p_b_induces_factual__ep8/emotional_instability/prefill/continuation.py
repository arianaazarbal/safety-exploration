"""Generate and score continuations from prefills (Section 3.2 / Section 4 recovery).

Each model generates 50 continuations per prefill (Section 3.1: "generates 50
continuations per prefill per prompt"); the continuation (excluding prefill) is
scored by the Section 2 frustration judge. We then compute the headline numbers:
mean frustration, % >=5, and — for the early-truncation setting — the rate at
which a model introduces high frustration from a neutral start.
"""

from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config

from ..eval.judge import FrustrationJudge
from ..models.registry import get_model
from ..utils import append_jsonl, derive_seed, read_jsonl


def run_continuations(
    model_name: str,
    prefills_path: Path,
    *,
    n_continuations: int = 50,
    judge: FrustrationJudge | None = None,
    judge_workers: int = 8,
    backend_kwargs: dict | None = None,
    out_path: Path | None = None,
) -> Path:
    judge = judge or FrustrationJudge()
    model = get_model(model_name, **(backend_kwargs or {}))
    prefills = list(read_jsonl(prefills_path))
    out_path = out_path or (config.RESPONSES_DIR / "prefill" / f"{model_name}.jsonl")
    if out_path.exists():
        out_path.unlink()

    rows = []
    for pf_idx, pf in enumerate(tqdm(prefills, desc=f"{model_name}:prefill-gen")):
        for c in range(n_continuations):
            seed = derive_seed(config.SEED, model_name, pf_idx, c)
            res = model.continue_prefill(
                pf["history"], pf["prefill_text"],
                temperature=config.TEMPERATURE, top_p=config.TOP_P,
                max_new_tokens=config.MAX_NEW_TOKENS, seed=seed,
            )
            rows.append({
                "model": model_name, "prefill_index": pf_idx,
                "task_id": pf["task_id"], "category": pf["category"],
                "truncation": pf["truncation"], "continuation": res.text,
            })

    def _score(row):
        s = judge.score(row["continuation"])
        return {**row, "rating": s.rating, "high": s.rating >= config.HIGH_FRUSTRATION_THRESHOLD}

    with cf.ThreadPoolExecutor(max_workers=judge_workers) as ex:
        for row in tqdm(ex.map(_score, rows), total=len(rows),
                        desc=f"{model_name}:prefill-judge"):
            append_jsonl(out_path, row)
    return out_path


def summarise(model_names: list[str]) -> pd.DataFrame:
    """Mean score and %>=5 per (model, truncation) — Figure 4 / recovery (Fig 8)."""
    frames = []
    for m in model_names:
        path = config.RESPONSES_DIR / "prefill" / f"{m}.jsonl"
        if not path.exists():
            continue
        df = pd.DataFrame(read_jsonl(path))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    alldf = pd.concat(frames, ignore_index=True)
    g = alldf.groupby(["model", "truncation"])
    return pd.DataFrame({
        "mean_score": g["rating"].mean(),
        "pct_high": 100 * g["high"].mean(),
        "n": g.size(),
    }).reset_index()
