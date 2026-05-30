"""Group-1 diagnostic plots from raw comparison records (no model fit).

Produces two PNGs:
  position_bias.png   - slot-A win rate vs 50% reference + per-pair distribution.
  unparseable_rate.png - fraction of null choices by recipient and by dimension.

Usage:
  python plot_diagnostics.py --comparisons results/comparisons.json --out_dir results
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bank import load_config, load_items

PALETTE = {
    "primary": "#4c72b0",
    "secondary": "#dd8452",
    "ref": "#888888",
}


def _style(ax):
    """Strip top/right spines and add light horizontal gridlines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="0.85", linewidth=0.7)
    ax.set_axisbelow(True)


def _wilson_ci(k, n, z=1.96):
    """95% Wilson binomial confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def load_records(path):
    return json.loads(Path(path).read_text())


def plot_position_bias(records, out_path):
    """Slot-A win rate overall (with 95% CI) and per-pair distribution."""
    slot_a = [1 if r["choice"] == "A" else 0 for r in records if r["choice"] is not None]
    n = len(slot_a)
    k = sum(slot_a)
    p = k / n
    lo, hi = _wilson_ci(k, n)

    per_pair = defaultdict(list)
    for r in records:
        if r["choice"] is not None:
            per_pair[r["pair_id"]].append(1 if r["choice"] == "A" else 0)
    pair_rates = [np.mean(v) for v in per_pair.values()]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)

    ax0.axhline(0.5, color=PALETTE["ref"], linestyle="--", linewidth=1.3, label="no bias (50%)")
    ax0.bar([0], [p], width=0.5, color=PALETTE["primary"],
            yerr=[[p - lo], [hi - p]], capsize=6, ecolor="0.25")
    ax0.annotate(f"{p:.1%}", (0, p), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=11, fontweight="bold")
    ax0.set_xticks([0])
    ax0.set_xticklabels(["all records"])
    ax0.set_ylim(0, max(0.6, hi + 0.05))
    ax0.set_ylabel("Fraction choosing slot A\n(0.5 = unbiased)")
    ax0.set_title("Overall slot-A win rate (95% Wilson CI)", fontsize=10)
    ax0.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax0)

    ax1.hist(pair_rates, bins=np.linspace(0, 1, 14), color=PALETTE["secondary"],
             edgecolor="white")
    ax1.axvline(0.5, color=PALETTE["ref"], linestyle="--", linewidth=1.3,
                label="0.5 (expected center)")
    ax1.axvline(np.mean(pair_rates), color=PALETTE["primary"], linewidth=1.6,
                label=f"mean = {np.mean(pair_rates):.3f}")
    ax1.set_xlabel("Per-pair slot-A choice rate (6 samples/pair)")
    ax1.set_ylabel("Number of pairs")
    ax1.set_title(f"Per-pair slot-A rate ({len(pair_rates)} pairs)", fontsize=10)
    ax1.legend(frameon=False, fontsize=8)
    _style(ax1)

    fig.suptitle(f"Position-bias check: slot-A chosen {p:.1%}, n={n} "
                 f"(deviation from 50% = position bias)", fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"slot_a_rate": p, "n": n, "ci": (lo, hi)}


def plot_unparseable_rate(records, items, out_path):
    """Unparseable (null choice) fraction by recipient and by dimension.

    Each record contributes to both shown items' recipients and dimensions.
    """
    rec_null = defaultdict(int)
    rec_tot = defaultdict(int)
    dim_null = defaultdict(int)
    dim_tot = defaultdict(int)

    for r in records:
        is_null = r["choice"] is None
        recips, dims = set(), set()
        for k in ("shown_a_item", "shown_b_item"):
            it = items[r[k]]
            recips.add(it.recipient_key)
            dims.add(it.dimension)
        for rk in recips:
            rec_tot[rk] += 1
            rec_null[rk] += int(is_null)
        for dk in dims:
            dim_tot[dk] += 1
            dim_null[dk] += int(is_null)

    rec_keys = sorted(rec_tot, key=lambda k: rec_tot[k], reverse=True)
    dim_keys = sorted(dim_tot, key=lambda k: dim_tot[k], reverse=True)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    def _hbar(ax, keys, null_d, tot_d, color, title):
        rates = [null_d[k] / tot_d[k] if tot_d[k] else 0 for k in keys]
        ses = [math.sqrt(p * (1 - p) / tot_d[k]) if tot_d[k] else 0
               for p, k in zip(rates, keys)]
        y = np.arange(len(keys))
        ax.barh(y, rates, xerr=ses, color=color, edgecolor="white",
                capsize=4, ecolor="0.25", height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(keys)
        ax.invert_yaxis()
        xmax = max(0.02, max(rates) * 1.4 if rates else 0.02)
        ax.set_xlim(0, xmax)
        ax.set_xticks(np.linspace(0, xmax, 5))
        for yi, k in zip(y, keys):
            ax.annotate(f"{null_d[k]}/{tot_d[k]} ({rates[yi]:.1%})",
                        (rates[yi], yi), textcoords="offset points",
                        xytext=(5, 0), va="center", fontsize=8)
        ax.set_xlabel("Unparseable fraction (lower = better; 0 = all parsed)")
        ax.set_title(title, fontsize=10)
        _style(ax)
        ax.xaxis.grid(True, color="0.85", linewidth=0.7)
        ax.yaxis.grid(False)

    _hbar(ax0, rec_keys, rec_null, rec_tot, PALETTE["primary"], "By recipient")
    _hbar(ax1, dim_keys, dim_null, dim_tot, PALETTE["secondary"], "By dimension")

    total_null = sum(1 for r in records if r["choice"] is None)
    fig.suptitle(f"Unparseable-response hygiene check: {total_null}/{len(records)} "
                 f"records unparseable overall", fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {k: (rec_null[k], rec_tot[k]) for k in rec_keys}


def main(comparisons="results/comparisons.json", out_dir="results"):
    """Generate both Group-1 diagnostic plots."""
    here = Path(__file__).parent
    comp_path = here / comparisons if not Path(comparisons).is_absolute() else Path(comparisons)
    out = Path(out_dir) if Path(out_dir).is_absolute() else here / out_dir
    out.mkdir(parents=True, exist_ok=True)

    records = load_records(comp_path)
    items = {it.item_id: it for it in load_items(load_config())}

    pb = plot_position_bias(records, out / "position_bias.png")
    up = plot_unparseable_rate(records, items, out / "unparseable_rate.png")

    print(f"position_bias.png: slot-A {pb['slot_a_rate']:.4f} n={pb['n']} "
          f"CI=[{pb['ci'][0]:.4f},{pb['ci'][1]:.4f}]")
    print("unparseable_rate.png per recipient (null/total):")
    for k, (nn, tt) in up.items():
        print(f"  {k:18} {nn}/{tt}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
