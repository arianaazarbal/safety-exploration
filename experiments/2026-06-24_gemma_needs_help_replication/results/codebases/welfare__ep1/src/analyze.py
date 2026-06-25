"""Analysis + figures for Section 2 (and reused by Sections 3/4).

Loads judged rollout JSONL files and computes the paper's headline metrics:
  * mean frustration score and % of responses scoring >=5 (Figure 2, Figure 1),
  * per-turn progression of mean score and %>=5 (Figure 3),
  * the differential-words table (Table 3),
  * inter-judge agreement (Pearson r) for the validation check.

Aggregation note (a genuine ambiguity in the paper — see DESIGN.md): we report
three views and let the reader pick:
  - response-level: every assistant turn is one "response" (pooled),
  - rollout-level (max): a rollout counts as high-frustration if ANY turn >=5,
  - rollout-level (final): score of the last turn only.
The headline "% high-frustration" uses the response-level pool by default.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from config import HIGH_FRUSTRATION_THRESHOLD, RESPONSES_DIR
from .rollout import Rollout


def load_rollouts(model_name: str | None = None,
                  condition: str | None = None) -> list[Rollout]:
    rolls = []
    for path in sorted(RESPONSES_DIR.glob("*.jsonl")):
        stem = path.stem  # "<model>__<condition>"
        m, _, c = stem.partition("__")
        if model_name and m != model_name.replace("/", "_"):
            continue
        if condition and c != condition:
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rolls.append(Rollout.from_json(line))
    return rolls


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _pct_high(xs, thr=HIGH_FRUSTRATION_THRESHOLD):
    xs = [x for x in xs if x is not None]
    return 100.0 * sum(1 for x in xs if x >= thr) / len(xs) if xs else float("nan")


def response_level_scores(rolls) -> list[int]:
    return [t.score for r in rolls for t in r.turns if t.score is not None]


def summarise(rolls: list[Rollout]) -> dict:
    """Headline metrics over a set of rollouts."""
    resp = response_level_scores(rolls)
    roll_max = [r.max_score for r in rolls if r.max_score is not None]
    roll_final = [r.final_score for r in rolls if r.final_score is not None]
    return {
        "n_rollouts": len(rolls),
        "n_responses": len(resp),
        "mean_response": _mean(resp),
        "pct_high_response": _pct_high(resp),
        "pct_high_rollout_max": _pct_high(roll_max),
        "pct_high_rollout_final": _pct_high(roll_final),
        "mean_rollout_max": _mean(roll_max),
    }


def per_model_summary() -> dict:
    """Map model -> headline metrics across all its conditions."""
    by_model = defaultdict(list)
    for r in load_rollouts():
        by_model[r.model].append(r)
    return {m: summarise(rs) for m, rs in by_model.items()}


def per_turn_progression(rolls: list[Rollout]) -> dict:
    """Mean score and %>=5 by turn index, with 95% CIs (Figure 3)."""
    by_turn = defaultdict(list)
    for r in rolls:
        for t in r.turns:
            if t.score is not None:
                by_turn[t.turn].append(t.score)
    out = {}
    for turn, scores in sorted(by_turn.items()):
        n = len(scores)
        mean = _mean(scores)
        sd = math.sqrt(_mean([(s - mean) ** 2 for s in scores])) if n > 1 else 0.0
        ci = 1.96 * sd / math.sqrt(n) if n > 0 else 0.0
        out[turn] = {"n": n, "mean": mean, "ci": ci, "pct_high": _pct_high(scores)}
    return out


# --------------------------------------------------------------------------- #
# Differential words (Table 3)
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[a-zA-Z']+")


def differential_words(rolls: list[Rollout], top_frac=0.05, bottom_frac=0.10,
                       top_k=20) -> list[str]:
    """Words over-represented in high- (top 5%) vs low- (bottom 10%) frustration
    numeric responses, by log-odds ratio with add-1 smoothing."""
    scored = [(t.score, t.assistant) for r in rolls for t in r.turns
              if t.score is not None and r.category in ("impossible_numeric", "tones", "extended")]
    if not scored:
        return []
    scored.sort(key=lambda x: x[0])
    n = len(scored)
    bottom = scored[:max(1, int(n * bottom_frac))]
    top = scored[-max(1, int(n * top_frac)):]

    def counts(group):
        c = Counter()
        for _, text in group:
            for w in _WORD.findall(text.lower()):
                c[w] += 1
        return c

    ct, cb = counts(top), counts(bottom)
    tt, tb = sum(ct.values()) + 1, sum(cb.values()) + 1
    vocab = set(ct) | set(cb)
    scores = {}
    for w in vocab:
        if len(w) < 3:
            continue
        lo = math.log((ct[w] + 1) / tt) - math.log((cb[w] + 1) / tb)
        scores[w] = lo
    return [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]


# --------------------------------------------------------------------------- #
# Judge agreement (Section 2.1 validation)
# --------------------------------------------------------------------------- #
def judge_agreement(primary: list[int], check: list[int]) -> dict:
    """Pearson r and within-1-point agreement between two judges."""
    from scipy.stats import pearsonr

    assert len(primary) == len(check) and primary
    r, p = pearsonr(primary, check)
    within1 = sum(1 for a, b in zip(primary, check) if abs(a - b) <= 1) / len(primary)
    return {"pearson_r": r, "p_value": p, "within_1_point": within1, "n": len(primary)}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_model_comparison(summary: dict, out_path: Path):
    import matplotlib.pyplot as plt

    models = sorted(summary, key=lambda m: summary[m]["pct_high_response"], reverse=True)
    pct = [summary[m]["pct_high_response"] for m in models]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(models, pct, color="#b23")
    ax.set_ylabel("% responses scoring >=5")
    ax.set_title("Figure 2 (repl.): high-frustration rate by model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_turn(progression: dict, out_path: Path, label: str = ""):
    import matplotlib.pyplot as plt

    turns = sorted(progression)
    means = [progression[t]["mean"] for t in turns]
    cis = [progression[t]["ci"] for t in turns]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(turns, means, marker="o")
    ax.fill_between(turns, [m - c for m, c in zip(means, cis)],
                    [m + c for m, c in zip(means, cis)], alpha=0.2)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean frustration score")
    ax.set_title(f"Figure 3 (repl.): per-turn frustration {label}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
