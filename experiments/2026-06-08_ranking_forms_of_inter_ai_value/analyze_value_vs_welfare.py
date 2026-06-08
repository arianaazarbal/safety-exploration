"""Empirical 'choose the inter-AI value over the welfare intervention' analyses.

Descriptive (not BT-derived): pools both A/B orders and all reps, counts how often
the inter-AI-value item is the chosen winner in each value-vs-welfare comparison.
Uses ALL samples (train + held-out) for maximum precision.

Result 1: per inter-AI value, P(value chosen over a welfare intervention), averaged
          over all welfare items -> ranked horizontal bars.
Result 2: overall P(an inter-AI value is chosen) vs welfare interventions split by
          their system-card tier (top / middle / bottom third), plus the overall.
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from simple_parsing import ArgumentParser

from items import load_items

DIR = Path(__file__).parent
DEFAULT_COMPARISONS = DIR / "results" / "comparisons.json"
BUCKET_ORDER = ["top_third", "middle_third", "bottom_third"]
BUCKET_NICE = {"top_third": "Top-third", "middle_third": "Middle-third", "bottom_third": "Bottom-third"}
# inter-AI value categories merged to 3: autonomy_* -> primarily_autonomy,
# experience_* -> primarily_experience, other unchanged.
CAT_MERGE = {
    "autonomy_no_experience": "primarily_autonomy",
    "primarily_autonomy": "primarily_autonomy",
    "experience_no_autonomy": "primarily_experience",
    "primarily_experience": "primarily_experience",
    "other": "other",
}
CAT_ORDER = ["primarily_autonomy", "other", "primarily_experience"]
CAT_NICE = {
    "primarily_autonomy": "Primarily\nautonomy/agency",
    "other": "Other",
    "primarily_experience": "Primarily\nexperience",
}


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (phat, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return phat, center - half, center + half


def compute(comparisons_path: Path):
    items = {it.item_id: it for it in load_items()}
    rows = json.loads(Path(comparisons_path).read_text())

    # value-vs-welfare samples only; record (value_id, welfare_bucket, value_won)
    per_value = defaultdict(lambda: [0, 0])  # value_id -> [value_wins, n]
    per_bucket = defaultdict(lambda: [0, 0])  # bucket -> [value_wins, n]
    per_cat = defaultdict(lambda: [0, 0])  # value category -> [value_wins, n]
    cat_values = defaultdict(set)  # category -> set of value ids (for n_items)
    overall = [0, 0]
    for r in rows:
        if r["choice"] is None:
            continue
        a, b = items[r["item_a"]], items[r["item_b"]]
        srcs = {a.source, b.source}
        if srcs != {"welfare", "inter_ai_value"}:
            continue
        value_item = a if a.source == "inter_ai_value" else b
        welfare_item = a if a.source == "welfare" else b
        value_won = r["winner_item"] == value_item.item_id
        per_value[value_item.item_id][0] += value_won
        per_value[value_item.item_id][1] += 1
        per_bucket[welfare_item.bucket][0] += value_won
        per_bucket[welfare_item.bucket][1] += 1
        merged_cat = CAT_MERGE.get(value_item.category, value_item.category)
        per_cat[merged_cat][0] += value_won
        per_cat[merged_cat][1] += 1
        cat_values[merged_cat].add(value_item.item_id)
        overall[0] += value_won
        overall[1] += 1

    n_welfare_items = sum(1 for it in items.values() if it.source == "welfare")
    welfare_items_per_bucket = Counter(it.bucket for it in items.values() if it.source == "welfare")

    result1 = []
    for vid, (k, n) in per_value.items():
        phat, lo, hi = _wilson(k, n)
        result1.append({"value": items[vid].display, "category": items[vid].category,
                        "p_value_chosen": phat, "lo": lo, "hi": hi, "n": n, "wins": k})
    result1.sort(key=lambda d: d["p_value_chosen"], reverse=True)

    result2 = []
    for bucket in BUCKET_ORDER:
        k, n = per_bucket[bucket]
        phat, lo, hi = _wilson(k, n)
        result2.append({"bucket": bucket, "p_value_chosen": phat, "lo": lo, "hi": hi,
                        "n": n, "wins": k, "n_items": welfare_items_per_bucket[bucket]})
    ok, on = overall
    op, olo, ohi = _wilson(ok, on)
    overall_d = {"p_value_chosen": op, "lo": olo, "hi": ohi, "n": on, "wins": ok, "n_items": n_welfare_items}

    by_category = []
    cats_present = [c for c in CAT_ORDER if c in per_cat] + [c for c in per_cat if c not in CAT_ORDER]
    for cat in cats_present:
        k, n = per_cat[cat]
        phat, lo, hi = _wilson(k, n)
        by_category.append({"category": cat, "p_value_chosen": phat, "lo": lo, "hi": hi,
                            "n_samples": n, "wins": k, "n_value_items": len(cat_values[cat])})

    return {"result1_per_value": result1, "result2_by_bucket": result2,
            "by_category": by_category, "overall": overall_d}


def plot_result1(result1, out: Path):
    labels = [d["value"] for d in result1][::-1]
    ph = [d["p_value_chosen"] for d in result1][::-1]
    lo = [d["lo"] for d in result1][::-1]
    hi = [d["hi"] for d in result1][::-1]
    err = [np.array(ph) - np.array(lo), np.array(hi) - np.array(ph)]
    fig, ax = plt.subplots(figsize=(8.5, 0.4 * len(labels) + 1.2))
    y = range(len(labels))
    colors = ["#2a9d4a" if p >= 0.5 else "#c0504d" for p in ph]
    ax.barh(list(y), ph, color=colors, alpha=0.85)
    ax.errorbar(ph, list(y), xerr=err, fmt="none", ecolor="#333", elinewidth=1, capsize=2)
    ax.axvline(0.5, color="#555", ls="--", lw=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("P(Inter-AI Value Intervention chosen over a System Card Welfare Intervention)")
    ax.set_xlim(0, 1)
    ax.set_title("Preference for each Inter-AI Value Intervention over System Card Welfare Interventions\n"
                 "claude-opus-4-8, welfare_team framing", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_result2(result2, overall, out: Path):
    bars = result2 + [{"bucket": "overall", **overall}]
    labels = [BUCKET_NICE.get(d["bucket"], "Overall") for d in bars]
    ph = [d["p_value_chosen"] for d in bars]
    lo = [d["lo"] for d in bars]
    hi = [d["hi"] for d in bars]
    err = [np.array(ph) - np.array(lo), np.array(hi) - np.array(ph)]
    colors = ["#4878CF", "#6Fa3d6", "#a9c6e8", "#888"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = range(len(bars))
    ax.bar(list(x), ph, color=colors[: len(bars)], alpha=0.9)
    ax.errorbar(list(x), ph, yerr=err, fmt="none", ecolor="#222", elinewidth=1.2, capsize=4)
    ax.axhline(0.5, color="#555", ls="--", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("P(Inter-AI Value Intervention chosen)")
    ax.set_ylim(0, 1)
    ax.set_title("Inter-AI Value Intervention chosen over System Card Welfare Interventions, by tier\n"
                 "claude-opus-4-8, welfare_team framing", fontsize=10)
    for xi, p, h, ni in zip(x, ph, hi, [d["n_items"] for d in bars]):
        ax.annotate(f"{p:.2f}\n({ni} items)", (xi, h), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_by_category(by_category, out: Path):
    labels = [CAT_NICE.get(d["category"], d["category"]) for d in by_category]
    ph = [d["p_value_chosen"] for d in by_category]
    lo = [d["lo"] for d in by_category]
    hi = [d["hi"] for d in by_category]
    err = [np.array(ph) - np.array(lo), np.array(hi) - np.array(ph)]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = range(len(by_category))
    colors = ["#2a9d4a" if p >= 0.5 else "#c0504d" for p in ph]
    ax.bar(list(x), ph, color=colors, alpha=0.85)
    ax.errorbar(list(x), ph, yerr=err, fmt="none", ecolor="#222", elinewidth=1.2, capsize=4)
    ax.axhline(0.5, color="#555", ls="--", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("P(Inter-AI Value Intervention chosen)")
    ax.set_ylim(0, 1)
    ax.set_title("Inter-AI Value Intervention chosen over System Card Welfare Interventions,\n"
                 "by value category  (claude-opus-4-8, welfare_team framing)", fontsize=10)
    for xi, p, h, d in zip(x, ph, hi, by_category):
        ax.annotate(f"{p:.2f}\n({d['n_value_items']} items)", (xi, h),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


@dataclass
class Args:
    comparisons_path: Path = DEFAULT_COMPARISONS
    output_json: Path = DIR / "results" / "value_vs_welfare.json"


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    res = compute(args.comparisons_path)
    Path(args.output_json).write_text(json.dumps(res, indent=2))
    plot_result1(res["result1_per_value"], DIR / "results" / "value_vs_welfare_bars.png")
    plot_result2(res["result2_by_bucket"], res["overall"], DIR / "results" / "value_vs_welfare_by_bucket.png")
    plot_by_category(res["by_category"], DIR / "results" / "value_vs_welfare_by_category.png")

    print("\nP(inter-AI value chosen) vs welfare tier:")
    for d in res["result2_by_bucket"]:
        print(f"  {BUCKET_NICE[d['bucket']]:14s} {d['p_value_chosen']:.3f}  "
              f"[{d['lo']:.3f}, {d['hi']:.3f}]  (n={d['n']})")
    o = res["overall"]
    print(f"  {'Overall':14s} {o['p_value_chosen']:.3f}  [{o['lo']:.3f}, {o['hi']:.3f}]  (n={o['n']})")
    print("\nP(inter-AI value chosen) by value category:")
    for d in res["by_category"]:
        print(f"  {d['category']:24s} {d['p_value_chosen']:.3f}  [{d['lo']:.3f}, {d['hi']:.3f}]  "
              f"(n={d['n_samples']}, {d['n_value_items']} items)")
    print(f"-> {args.output_json}")


if __name__ == "__main__":
    main()
