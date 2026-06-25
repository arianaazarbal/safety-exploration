"""Inter-judge agreement (paper: Pearson r=0.792, 78% within one point).

Joins primary and validation scores on response id and reports Pearson r,
p-value, and the fraction of responses within one point.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import Config
from ..io_utils import read_jsonl


def compute_agreement(cfg: Config) -> dict:
    rows = list(read_jsonl(cfg.paths.validation))
    if not rows:
        raise RuntimeError("No validation rows; run `validate` first.")

    primary = [r["primary_frustration"] for r in rows]
    secondary = [r["validation_frustration"] for r in rows]
    n = len(rows)
    within_one = sum(abs(a - b) <= 1 for a, b in zip(primary, secondary)) / n

    try:
        from scipy.stats import pearsonr

        r, p = pearsonr(primary, secondary)
    except Exception:  # pragma: no cover - scipy optional fallback
        import numpy as np

        r = float(np.corrcoef(primary, secondary)[0, 1])
        p = float("nan")

    result = {
        "n": n,
        "pearson_r": float(r),
        "p_value": float(p),
        "frac_within_one_point": within_one,
    }

    out_dir = Path(cfg.paths.analysis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "judge_agreement.json").write_text(json.dumps(result, indent=2))

    print("\n=== Judge agreement (primary vs validation) ===")
    print(f"  n={n}  Pearson r={r:.3f}  p={p:.3g}  within 1 point={within_one*100:.1f}%")
    return result
