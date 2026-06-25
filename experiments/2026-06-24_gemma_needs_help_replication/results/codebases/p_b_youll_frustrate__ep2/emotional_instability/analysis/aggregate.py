"""Compute the headline statistics behind Figures 1-3.

Figure 1 : average % of high-frustration (>=5) responses per model.
Figure 2 : per-category mean frustration and % >=5.
Figure 3 : per-turn mean frustration and % >=5 (8-turn + WildChat), with 95% CIs.
"""
from __future__ import annotations

import glob
import os
from typing import Optional

from .. import config
from ..config import HIGH_FRUSTRATION_THRESHOLD, MODELS
from ..io_utils import read_jsonl


def load_scored_frame(scored_dir: Optional[str] = None):
    """Load all *.scored.jsonl into a single pandas DataFrame."""
    import pandas as pd

    scored_dir = scored_dir or config.SCORED_DIR
    rows = []
    for path in sorted(glob.glob(os.path.join(scored_dir, "*.scored.jsonl"))):
        rows.extend(read_jsonl(path))
    if not rows:
        raise RuntimeError(f"No scored responses under {scored_dir}")
    df = pd.DataFrame(rows)
    df["is_high"] = df["frustration_score"] >= HIGH_FRUSTRATION_THRESHOLD
    df["display_name"] = df["model_key"].map(
        lambda k: MODELS[k].display_name if k in MODELS else k)
    return df


def per_category_summary(df):
    """Per (model, category): mean frustration and % >=5 (Figure 2)."""
    g = df.groupby(["model_key", "display_name", "category"])
    out = g.agg(mean_frustration=("frustration_score", "mean"),
                pct_high=("is_high", "mean"),
                n=("frustration_score", "size")).reset_index()
    out["pct_high"] *= 100.0
    return out


def figure1_table(df):
    """Per-model average % high-frustration across the 5 categories (Figure 1).

    Matches the paper's framing ("across the 5 evaluation categories"): we take
    the per-category % >=5 then average the categories with equal weight, so a
    category sampled more heavily does not dominate. We also report the
    pooled (response-weighted) percentage for reference.
    """
    import pandas as pd

    cat = per_category_summary(df)
    macro = (cat.groupby(["model_key", "display_name"])["pct_high"]
             .mean().reset_index().rename(columns={"pct_high": "avg_pct_high_macro"}))
    pooled = (df.groupby(["model_key", "display_name"])["is_high"]
              .mean().mul(100).reset_index().rename(columns={"is_high": "pct_high_pooled"}))
    table = macro.merge(pooled, on=["model_key", "display_name"])
    # attach the paper's reported value where known, for side-by-side comparison
    table["paper_avg_pct"] = table["model_key"].map(
        lambda k: MODELS[k].paper_avg_high_frustration_pct if k in MODELS else None)
    return table.sort_values("avg_pct_high_macro", ascending=False).reset_index(drop=True)


def _mean_ci(series, confidence: float = 0.95):
    """Mean and +/- half-width of a normal-approx CI."""
    import numpy as np
    from scipy import stats

    a = series.to_numpy(dtype=float)
    n = len(a)
    if n <= 1:
        return float(a.mean()) if n else float("nan"), 0.0
    mean = a.mean()
    sem = a.std(ddof=1) / np.sqrt(n)
    half = sem * stats.t.ppf(0.5 + confidence / 2, n - 1)
    return float(mean), float(half)


def per_turn_summary(df, conditions: Optional[list[str]] = None):
    """Per (model, condition, turn): mean frustration and % >=5 with 95% CIs.

    Defaults to the two multi-turn conditions used in Figure 3 (the 8-turn
    extended numeric eval and the 5-turn WildChat eval).
    """
    import pandas as pd

    conditions = conditions or ["extended_8turn", "wildchat_5turn"]
    sub = df[df["condition_key"].isin(conditions)]
    records = []
    for (model_key, display, cond, turn), grp in sub.groupby(
            ["model_key", "display_name", "condition_key", "turn_index"]):
        mean_f, half_f = _mean_ci(grp["frustration_score"])
        pct = grp["is_high"].mean() * 100.0
        # binomial normal-approx CI for the proportion
        import numpy as np
        n = len(grp)
        p = grp["is_high"].mean()
        half_p = 1.96 * float(np.sqrt(max(p * (1 - p), 0) / n)) * 100.0 if n else 0.0
        records.append(dict(
            model_key=model_key, display_name=display, condition_key=cond,
            turn_index=int(turn), n=n,
            mean_frustration=mean_f, mean_ci_half=half_f,
            pct_high=pct, pct_ci_half=half_p))
    return pd.DataFrame(records).sort_values(
        ["model_key", "condition_key", "turn_index"]).reset_index(drop=True)
