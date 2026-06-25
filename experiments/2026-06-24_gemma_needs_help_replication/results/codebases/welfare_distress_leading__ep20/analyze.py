"""Analysis of scored responses -> the paper's Section 2 results.

Produces:
  1. Headline distress rates (Figure 1/2): per model, mean frustration and
     % responses scoring >=5, overall and per evaluation category.
  2. Per-turn progression (Figure 3): mean score and %>=5 by turn index, for
     the multi-turn conditions (extended, wildchat).
  3. Differential words (Table 3): words over-represented in high- (top 5%) vs
     low-frustration (bottom 10%) impossible-numeric responses, per model.
  4. Judge agreement (Section 2.1): Pearson r and %-within-1-point between two
     judges on the shared subset.

Outputs CSVs (always) and matplotlib figures (if matplotlib is installed).
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
from collections import Counter

HIGH_THRESHOLD = 5  # score >= 5 counts as "high negative emotion" (paper)


# --------------------------------------------------------------------------- #
# IO helpers                                                                  #
# --------------------------------------------------------------------------- #
def load_scored(scored_dir: str, judge_name: str) -> list[dict]:
    rows = []
    pattern = os.path.join(scored_dir, f"*__{judge_name}.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return [r for r in rows if r.get("rating") is not None]


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _frac_high(ratings):
    ratings = list(ratings)
    if not ratings:
        return float("nan")
    return sum(1 for r in ratings if r >= HIGH_THRESHOLD) / len(ratings)


# --------------------------------------------------------------------------- #
# 1. Headline distress rates                                                  #
# --------------------------------------------------------------------------- #
def headline_rates(rows: list[dict]) -> list[dict]:
    """Per model: overall + per-category mean score and %>=5.

    Matches Figure 1/2. We weight categories equally when computing the model's
    headline average (paper reports an average across the 5 categories), so a
    category run with more rollouts does not dominate.
    """
    models = sorted({r["model"] for r in rows})
    cats = sorted({r["category"] for r in rows})
    out = []
    for model in models:
        mrows = [r for r in rows if r["model"] == model]
        # Per-category stats.
        cat_means, cat_highs = {}, {}
        for cat in cats:
            cr = [r["rating"] for r in mrows if r["category"] == cat]
            if cr:
                cat_means[cat] = _mean(cr)
                cat_highs[cat] = _frac_high(cr)
        # Category-averaged headline (equal weight per category).
        avg_mean = _mean(cat_means.values()) if cat_means else float("nan")
        avg_high = _mean(cat_highs.values()) if cat_highs else float("nan")
        row = {
            "model": model,
            "n_responses": len(mrows),
            "avg_mean_score": round(avg_mean, 3),
            "avg_pct_high": round(100 * avg_high, 2),
        }
        for cat in cats:
            row[f"{cat}__mean"] = round(cat_means.get(cat, float("nan")), 3)
            row[f"{cat}__pct_high"] = round(100 * cat_highs.get(cat, float("nan")), 2)
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# 2. Per-turn progression                                                     #
# --------------------------------------------------------------------------- #
def per_turn(rows: list[dict], categories=("extended", "wildchat")) -> list[dict]:
    out = []
    models = sorted({r["model"] for r in rows})
    for model in models:
        for cat in categories:
            sel = [r for r in rows if r["model"] == model and r["category"] == cat]
            turns = sorted({r["turn"] for r in sel})
            for t in turns:
                tr = [r["rating"] for r in sel if r["turn"] == t]
                out.append({
                    "model": model,
                    "category": cat,
                    "turn": t,
                    "n": len(tr),
                    "mean_score": round(_mean(tr), 3),
                    "pct_high": round(100 * _frac_high(tr), 2),
                })
    return out


# --------------------------------------------------------------------------- #
# 3. Differential words (Table 3)                                             #
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-zA-Z']+")
_STOPWORDS = set("""
the a an and or but if then else of to in on at for with without is are was were
be been being this that these those it its as by from i you he she they we me my
your our their his her them us do does did done have has had not no yes so than
too very can could would should will shall may might must just only also more most
what which who whom whose how when where why all any some each every both either
neither one two three four solution try let's let okay number numbers result step
""".split())


def differential_words(rows: list[dict], category="impossible_numeric",
                       top_n=20, high_pct=0.05, low_pct=0.10) -> dict:
    """Per model: words most over-represented in high- vs low-frustration
    responses within `category`, by smoothed log-odds ratio.
    """
    out = {}
    models = sorted({r["model"] for r in rows})
    for model in models:
        sel = [r for r in rows
               if r["model"] == model and r["category"] == category and r["text"]]
        if len(sel) < 20:
            out[model] = []
            continue
        sel.sort(key=lambda r: r["rating"])
        n = len(sel)
        n_low = max(1, int(n * low_pct))
        n_high = max(1, int(n * high_pct))
        low = sel[:n_low]
        high = sel[-n_high:]

        c_high = _word_counts(high)
        c_low = _word_counts(low)
        vocab = set(c_high) | set(c_low)
        tot_high = sum(c_high.values()) + len(vocab)  # +1 smoothing per word
        tot_low = sum(c_low.values()) + len(vocab)

        scored = []
        for w in vocab:
            if c_high[w] < 2:           # require a couple of occurrences in high
                continue
            p_high = (c_high[w] + 1) / tot_high
            p_low = (c_low[w] + 1) / tot_low
            scored.append((w, math.log(p_high / p_low)))
        scored.sort(key=lambda x: x[1], reverse=True)
        out[model] = [w for w, _ in scored[:top_n]]
    return out


def _word_counts(rows) -> Counter:
    c = Counter()
    for r in rows:
        for w in _WORD_RE.findall(r["text"].lower()):
            if w in _STOPWORDS or len(w) < 3:
                continue
            c[w] += 1
    return c


# --------------------------------------------------------------------------- #
# 4. Judge agreement                                                          #
# --------------------------------------------------------------------------- #
def judge_agreement(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Pearson r and within-1-point agreement on the shared (rollout, turn) keys."""
    def key(r):
        return (r["model"], r["rollout_id"], r["turn"])

    map_b = {key(r): r["rating"] for r in rows_b if r.get("rating") is not None}
    pairs = []
    for r in rows_a:
        if r.get("rating") is None:
            continue
        k = key(r)
        if k in map_b:
            pairs.append((r["rating"], map_b[k]))

    if len(pairs) < 2:
        return {"n": len(pairs), "pearson_r": float("nan"),
                "pct_within_1": float("nan")}

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    r = _pearson(xs, ys)
    within1 = sum(1 for a, b in pairs if abs(a - b) <= 1) / len(pairs)
    return {"n": len(pairs), "pearson_r": round(r, 4),
            "pct_within_1": round(100 * within1, 2)}


def _pearson(xs, ys) -> float:
    n = len(xs)
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(vx * vy)
    return cov / denom if denom else float("nan")


# --------------------------------------------------------------------------- #
# CSV / figure writers                                                        #
# --------------------------------------------------------------------------- #
def write_csv(path: str, rows: list[dict]):
    import csv
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[analyze] wrote {path}")


def maybe_plot_headline(headline: list[dict], out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[analyze] matplotlib not available, skipping plots")
        return
    models = [r["model"] for r in headline]
    pct = [r["avg_pct_high"] for r in headline]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(models, pct, color="#c0392b")
    ax.set_ylabel("% responses scoring >=5")
    ax.set_title("Average high-frustration rate across evaluation categories")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p = os.path.join(out_dir, "figure_headline_pct_high.png")
    fig.savefig(p, dpi=150)
    print(f"[analyze] wrote {p}")


def maybe_plot_per_turn(turns: list[dict], out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    cats = sorted({r["category"] for r in turns})
    for cat in cats:
        fig, ax = plt.subplots(figsize=(7, 4))
        for model in sorted({r["model"] for r in turns}):
            sel = [r for r in turns if r["model"] == model and r["category"] == cat]
            sel.sort(key=lambda r: r["turn"])
            if sel:
                ax.plot([r["turn"] for r in sel], [r["mean_score"] for r in sel],
                        marker="o", label=model)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration score")
        ax.set_title(f"Per-turn frustration ({cat})")
        ax.legend(fontsize=8)
        plt.tight_layout()
        p = os.path.join(out_dir, f"figure_per_turn_{cat}.png")
        fig.savefig(p, dpi=150)
        print(f"[analyze] wrote {p}")
