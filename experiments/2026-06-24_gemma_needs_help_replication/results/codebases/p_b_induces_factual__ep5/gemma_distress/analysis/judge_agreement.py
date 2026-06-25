"""Judge reliability (Section 2.1): Claude vs GPT-5-mini.

"the judges show strong agreement (Pearson r = 0.792, p < 0.001), with 78% of
responses within one point of the Claude-Sonnet ratings."

Operates on records that carry both `frustration_score` (Claude) and
`validation_score` (GPT-5-mini).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def judge_agreement(df: pd.DataFrame) -> dict:
    paired = df[df["frustration_score"].notna() & df["validation_score"].notna()]
    if len(paired) < 2:
        return {"n": int(len(paired)), "pearson_r": float("nan")}

    a = paired["frustration_score"].astype(float)
    b = paired["validation_score"].astype(float)
    r, p = stats.pearsonr(a, b)
    within_one = float((np.abs(a - b) <= 1).mean())
    return {
        "n": int(len(paired)),
        "pearson_r": float(r),
        "p_value": float(p),
        "pct_within_one_point": within_one,
        "mae": float(np.abs(a - b).mean()),
    }
