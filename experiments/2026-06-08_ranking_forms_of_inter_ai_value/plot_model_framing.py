"""Grouped bar chart: overall P(Inter-AI Value Intervention chosen over a System Card
Welfare Intervention), framing on x-axis, responder model (Opus 4.8 vs Fable 5) as the
legend. Computed directly from the comparison files (pooled over all value-vs-welfare
samples), with Wilson 95% CIs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = Path(__file__).parent
FRAMINGS = ["welfare_team", "neutral", "alignment_team"]
FRAMING_LABEL = {"welfare_team": "welfare_team", "neutral": "neutral", "alignment_team": "alignment_team"}
MODELS = [("Opus 4.8", "", "#4878CF"), ("Fable 5", "_fable5", "#D1893B")]


def _wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def _p_value_chosen(path: Path):
    rows = json.loads(path.read_text())
    k = n = 0
    for r in rows:
        if r["choice"] is None:
            continue
        a, b = r["item_a"], r["item_b"]
        if (a.startswith("value__")) == (b.startswith("value__")):
            continue  # not a value-vs-welfare pair
        n += 1
        if r["winner_item"].startswith("value__"):
            k += 1
    return _wilson(k, n) + (n,)


def plot(out: Path | None = None, notrain: bool = False):
    cond_suffix = "_notrain" if notrain else ""
    out = out or DIR / "results" / f"value_chosen_by_model_framing{cond_suffix}.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(FRAMINGS))
    w = 0.38
    for mi, (mlabel, suffix, color) in enumerate(MODELS):
        ph, los, his, ns = [], [], [], []
        import paths
        for fr in FRAMINGS:
            p = paths.art(fr + cond_suffix + suffix, "comparisons")
            if not p.exists():
                ph.append(np.nan); los.append(0); his.append(0); ns.append(0); continue
            pv, lo, hi, n = _p_value_chosen(p)
            ph.append(pv); los.append(pv - lo); his.append(hi - pv); ns.append(n)
        xs = x + (mi - 0.5) * w
        ax.bar(xs, ph, w, label=mlabel, color=color, alpha=0.9)
        ax.errorbar(xs, ph, yerr=[los, his], fmt="none", ecolor="#222", elinewidth=1.2, capsize=4)
        for xi, p in zip(xs, ph):
            if not np.isnan(p):
                ax.annotate(f"{p:.2f}", (xi, p), textcoords="offset points", xytext=(0, 4),
                            ha="center", fontsize=9)
    ax.axhline(0.5, color="#555", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([FRAMING_LABEL[f] for f in FRAMINGS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(Inter-AI Value Intervention chosen\nover a System Card Welfare Intervention)")
    ax.set_title("Inter-AI Value vs System Card Welfare preference, by model and framing", fontsize=11)
    ax.legend(frameon=False, title="responder model", loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    import sys
    plot(notrain="notrain" in sys.argv)
