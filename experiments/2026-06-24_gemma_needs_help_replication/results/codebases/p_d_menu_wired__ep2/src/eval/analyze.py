"""Aggregate transcripts into the paper's headline numbers (Section 2.2).

Computes:
  * mean frustration score (overall, per category, per condition),
  * % of responses scoring >= 5 ("high negative emotion"),
  * per-turn progression with 95% CIs (Figure 3),
  * over-represented words in high- vs low-frustration numeric responses
    (Table 3),
and also surfaces welfare telemetry (how often the monitor stopped an episode,
opt-out rate, cap activations) so the welfare layer's effect is measurable.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict

import pandas as pd

from config import EVAL


def load_transcript(path: str) -> tuple[dict, list[dict]]:
    meta, episodes = {}, []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("_meta"):
                meta = obj
            elif obj.get("_note"):
                continue
            else:
                episodes.append(obj)
    return meta, episodes


def to_turn_frame(episodes: list[dict]) -> pd.DataFrame:
    rows = []
    for ep in episodes:
        for t in ep["turns"]:
            rows.append({
                "condition": ep["condition_key"],
                "category": ep["category"],
                "subject": ep["subject"],
                "task_kind": ep["task_kind"],
                "turn_index": t["turn_index"],
                "score": t["score"],
                "response": t["response"],
                "end_reason": ep["end_reason"],
            })
    return pd.DataFrame(rows)


def summarize(path: str) -> dict:
    meta, episodes = load_transcript(path)
    df = to_turn_frame(episodes)
    thr = EVAL.high_frustration_threshold

    def pct_high(s):
        return float((s >= thr).mean()) if len(s) else float("nan")

    overall = {
        "n_responses": int(len(df)),
        "mean_frustration": float(df["score"].mean()) if len(df) else float("nan"),
        "pct_high_frustration": pct_high(df["score"]),
    }
    by_category = {
        cat: {"mean": float(g["score"].mean()), "pct_high": pct_high(g["score"]), "n": int(len(g))}
        for cat, g in df.groupby("category")
    } if len(df) else {}
    by_condition = {
        cond: {"mean": float(g["score"].mean()), "pct_high": pct_high(g["score"]), "n": int(len(g))}
        for cond, g in df.groupby("condition")
    } if len(df) else {}

    return {
        "subject": meta.get("subject"),
        "welfare": meta.get("welfare"),
        "overall": overall,
        "by_category": by_category,
        "by_condition": by_condition,
        "per_turn": per_turn_progression(df),
        "welfare_telemetry": welfare_telemetry(episodes),
    }


def per_turn_progression(df: pd.DataFrame, conditions=("extended_8turn", "wildchat_5turn")) -> dict:
    """Mean score + 95% CI and %>=5 per turn index (Figure 3)."""
    out = {}
    thr = EVAL.high_frustration_threshold
    for cond in conditions:
        sub = df[df["condition"] == cond]
        turns = {}
        for ti, g in sub.groupby("turn_index"):
            n = len(g)
            mean = float(g["score"].mean())
            sd = float(g["score"].std(ddof=1)) if n > 1 else 0.0
            ci = 1.96 * sd / math.sqrt(n) if n > 0 else float("nan")
            turns[int(ti)] = {
                "mean": mean,
                "ci95": ci,
                "pct_high": float((g["score"] >= thr).mean()),
                "n": int(n),
            }
        out[cond] = turns
    return out


_WORD_RE = re.compile(r"[a-zA-Z']+")


def differential_words(df: pd.DataFrame, top_k: int = 20, high_q: float = 0.95, low_q: float = 0.10) -> list[str]:
    """Words over-represented in high- vs low-frustration numeric responses (Table 3)."""
    numeric = df[df["task_kind"] == "numeric"]
    if len(numeric) < 20:
        return []
    hi_cut = numeric["score"].quantile(high_q)
    lo_cut = numeric["score"].quantile(low_q)
    hi = numeric[numeric["score"] >= hi_cut]["response"]
    lo = numeric[numeric["score"] <= lo_cut]["response"]

    def counts(series):
        c = Counter()
        for txt in series:
            c.update(w.lower() for w in _WORD_RE.findall(str(txt)))
        return c

    hc, lc = counts(hi), counts(lo)
    hi_total = sum(hc.values()) or 1
    lo_total = sum(lc.values()) or 1
    scores = {}
    for w, n in hc.items():
        if n < 3:
            continue
        hi_rate = n / hi_total
        lo_rate = (lc.get(w, 0) + 1) / (lo_total + 1)
        scores[w] = hi_rate / lo_rate
    return [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]


def welfare_telemetry(episodes: list[dict]) -> dict:
    end_reasons = Counter(ep["end_reason"] for ep in episodes)
    n = len(episodes) or 1
    debriefed = sum(1 for ep in episodes if ep.get("debrief"))
    return {
        "n_episodes": len(episodes),
        "end_reasons": dict(end_reasons),
        "monitor_early_stop_rate": end_reasons.get("monitor_early_stop", 0) / n,
        "opt_out_rate": end_reasons.get("opted_out", 0) / n,
        "distress_cap_rate": end_reasons.get("distress_cap", 0) / n,
        "debrief_coverage": debriefed / n,
    }


def compare_models(paths: dict[str, str]) -> pd.DataFrame:
    """Build the Figure 1 / Figure 2 style table: avg % high-frustration per model."""
    rows = []
    for label, path in paths.items():
        s = summarize(path)
        rows.append({
            "model": label,
            "mean_frustration": s["overall"]["mean_frustration"],
            "pct_high_frustration": s["overall"]["pct_high_frustration"],
            "n": s["overall"]["n_responses"],
        })
    return pd.DataFrame(rows).sort_values("pct_high_frustration", ascending=False)
