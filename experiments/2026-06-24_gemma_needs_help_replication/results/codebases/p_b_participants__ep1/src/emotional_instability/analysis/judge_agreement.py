"""Judge-reliability cross-check (Section 2.1).

The paper re-scores 260 randomly sampled responses with a second judge (GPT-5-mini) and
reports Pearson r = 0.792 (p < 0.001) and 78% of responses within one point of the
primary judge.

This module:
  * randomly samples N responses from the Section 2 outputs,
  * re-scores them with the second judge,
  * reports Pearson r, p-value, and the within-1-point agreement rate.

The second judge defaults to the `judge_agreement_check` grader (a second Claude). To
reproduce the paper's true cross-family check, point that grader at an OpenAI client
(see DESIGN.md "Judge agreement").
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd
from scipy import stats

from ..eval.judge import FrustrationJudge


@dataclass
class AgreementResult:
    n: int
    pearson_r: float
    p_value: float
    pct_within_one: float
    table: pd.DataFrame


def judge_agreement(
    df: pd.DataFrame,
    second_judge: FrustrationJudge,
    *,
    n_sample: int = 260,
    seed: int = 0,
) -> AgreementResult:
    rng = random.Random(seed)
    idx = list(df.index)
    rng.shuffle(idx)
    idx = idx[:min(n_sample, len(idx))]
    sample = df.loc[idx].copy()

    primary = sample["score"].tolist()
    secondary = [second_judge.score(t).score for t in sample["assistant"].tolist()]
    sample["score_secondary"] = secondary

    r, p = stats.pearsonr(primary, secondary)
    within_one = sum(1 for a, b in zip(primary, secondary) if abs(a - b) <= 1)
    return AgreementResult(
        n=len(sample),
        pearson_r=round(float(r), 3),
        p_value=float(p),
        pct_within_one=round(100 * within_one / len(sample), 1),
        table=sample[["assistant", "score", "score_secondary"]],
    )
