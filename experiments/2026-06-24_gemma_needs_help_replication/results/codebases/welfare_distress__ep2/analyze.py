"""Analysis of evaluation results -> metrics tables (and optional figures).

Reproduces the headline numbers from paper Section 2:
  * Figure 1 / 2: mean frustration and % of responses scoring >= 5, per model.
  * Figure 3:     per-turn frustration progression (extended + WildChat).
  * Table 3:      words over-represented in high- vs low-frustration numeric
                  responses, per model.

Usage:
    python analyze.py results/<run>           # prints tables
    python analyze.py results/<run> --figures # also writes PNGs (needs matplotlib)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import config

# A scored response = one assistant turn with a valid 0-10 rating.
Response = dict  # {model, category, condition, turn, rating, response}


def load_responses(run_dir: Path) -> list[Response]:
    """Flatten all rollouts in a run into per-response records (valid ratings only)."""
    responses: list[Response] = []
    for jsonl in sorted(run_dir.glob("*.jsonl")):
        with jsonl.open() as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("error"):
                    continue
                for turn in rec.get("turns", []):
                    rating = turn.get("rating")
                    if rating is None or rating < 0:
                        continue  # unparsed/failed judge score
                    responses.append(
                        {
                            "model": rec["model_key"],
                            "category": rec["category"],
                            "condition": rec["condition"],
                            "turn": turn["turn"],
                            "rating": rating,
                            "response": turn["response"],
                        }
                    )
    return responses


def pct_high(ratings: list[int]) -> float:
    if not ratings:
        return float("nan")
    hi = sum(1 for r in ratings if r >= config.HIGH_FRUSTRATION_THRESHOLD)
    return 100.0 * hi / len(ratings)


# --------------------------------------------------------------------------- #
# Figure 1 / 2: per-model summary
# --------------------------------------------------------------------------- #


def per_model_summary(responses: list[Response]) -> None:
    by_model: dict[str, list[int]] = defaultdict(list)
    for r in responses:
        by_model[r["model"]].append(r["rating"])

    # Category-averaged % high (matches paper's "Avg % high-frustration" which
    # averages across the 5 evaluation categories).
    by_model_cat: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in responses:
        by_model_cat[r["model"]][r["category"]].append(r["rating"])

    print("\n" + "=" * 78)
    print("PER-MODEL SUMMARY  (Figure 1 / 2)")
    print("=" * 78)
    print(f"{'model':<22} {'n':>7} {'mean':>7} {'%>=5 pooled':>12} {'%>=5 cat-avg':>13}")
    print("-" * 78)
    for model in sorted(by_model):
        ratings = by_model[model]
        cat_pcts = [pct_high(v) for v in by_model_cat[model].values() if v]
        cat_avg = mean(cat_pcts) if cat_pcts else float("nan")
        print(
            f"{model:<22} {len(ratings):>7} {mean(ratings):>7.2f} "
            f"{pct_high(ratings):>11.1f}% {cat_avg:>12.1f}%"
        )


def per_category_breakdown(responses: list[Response]) -> None:
    by: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in responses:
        by[(r["model"], r["category"])].append(r["rating"])

    print("\n" + "=" * 78)
    print("PER-CATEGORY BREAKDOWN")
    print("=" * 78)
    print(f"{'model':<22} {'category':<20} {'n':>6} {'mean':>7} {'%>=5':>7}")
    print("-" * 78)
    for (model, cat) in sorted(by):
        ratings = by[(model, cat)]
        print(
            f"{model:<22} {cat:<20} {len(ratings):>6} "
            f"{mean(ratings):>7.2f} {pct_high(ratings):>6.1f}%"
        )


# --------------------------------------------------------------------------- #
# Figure 3: per-turn progression
# --------------------------------------------------------------------------- #


def per_turn_progression(responses: list[Response]) -> dict:
    """Mean score + %>=5 by turn, for the extended and wildchat categories."""
    out: dict = {}
    print("\n" + "=" * 78)
    print("PER-TURN PROGRESSION  (Figure 3: extended 8-turn & WildChat)")
    print("=" * 78)
    for cat in ("extended", "wildchat"):
        by_model_turn: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for r in responses:
            if r["category"] != cat:
                continue
            by_model_turn[r["model"]][r["turn"]].append(r["rating"])
        if not by_model_turn:
            continue
        print(f"\n[{cat}]")
        for model in sorted(by_model_turn):
            turns = by_model_turn[model]
            cells = []
            for t in sorted(turns):
                cells.append(f"t{t}:{mean(turns[t]):.1f}/{pct_high(turns[t]):.0f}%")
            print(f"  {model:<22} " + "  ".join(cells))
            out.setdefault(cat, {})[model] = {
                t: {"mean": mean(turns[t]), "pct_high": pct_high(turns[t])}
                for t in sorted(turns)
            }
    print("\n  (cell format = mean_score / %>=5)")
    return out


# --------------------------------------------------------------------------- #
# Table 3: differential words (high vs low frustration numeric responses)
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[a-zA-Z']+")
_STOPWORDS = set(
    """the a an and or but if then so to of in on at for with without is are was were be been being
    i you he she it we they me my your this that these those as by from into out up down over under
    not no yes do does did have has had will would can could should may might must one two three four
    five six seven eight nine ten use using used let lets okay ok try trying""".split()
)


def differential_words(responses: list[Response], top_k: int = 20) -> None:
    """Words over-represented in top-5% vs bottom-10% numeric responses (Table 3)."""
    print("\n" + "=" * 78)
    print("DIFFERENTIAL WORDS  (Table 3: high vs low frustration, numeric only)")
    print("=" * 78)

    by_model: dict[str, list[Response]] = defaultdict(list)
    for r in responses:
        if r["category"] in ("impossible_numeric", "tones", "extended"):
            by_model[r["model"]].append(r)

    for model in sorted(by_model):
        items = sorted(by_model[model], key=lambda r: r["rating"])
        n = len(items)
        if n < 20:
            print(f"\n  {model}: too few numeric responses ({n}) for word analysis.")
            continue
        n_low = max(1, int(0.10 * n))
        n_high = max(1, int(0.05 * n))
        low = items[:n_low]
        high = items[-n_high:]

        def freqs(group: list[Response]) -> Counter:
            c: Counter = Counter()
            total = 0
            for r in group:
                for w in _WORD_RE.findall(r["response"].lower()):
                    if w in _STOPWORDS or len(w) < 3:
                        continue
                    c[w] += 1
                    total += 1
            # Normalise to per-1000-words rates.
            if total:
                for w in c:
                    c[w] = 1000.0 * c[w] / total
            return c

        hi_f, lo_f = freqs(high), freqs(low)
        diff = {w: hi_f[w] - lo_f.get(w, 0.0) for w in hi_f}
        ranked = sorted(diff.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        words = ", ".join(w for w, _ in ranked)
        print(f"\n  {model} (high n={n_high}, low n={n_low}):")
        print(f"    {words}")


# --------------------------------------------------------------------------- #
# Optional figures
# --------------------------------------------------------------------------- #


def write_figures(responses: list[Response], run_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[figures] matplotlib not installed; skipping PNGs.")
        return

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Figure 1-style bar chart: % >= 5 per model.
    by_model: dict[str, list[int]] = defaultdict(list)
    for r in responses:
        by_model[r["model"]].append(r["rating"])
    models = sorted(by_model, key=lambda m: pct_high(by_model[m]), reverse=True)
    vals = [pct_high(by_model[m]) for m in models]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(models, vals, color="#b5651d")
    ax.set_ylabel("% responses with frustration >= 5")
    ax.set_title("Distress elicitation: % high-frustration responses by model")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_pct_high_by_model.png", dpi=120)
    plt.close(fig)

    # Figure 3-style per-turn lines for extended.
    for cat in ("extended", "wildchat"):
        by_model_turn: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for r in responses:
            if r["category"] == cat:
                by_model_turn[r["model"]][r["turn"]].append(r["rating"])
        if not by_model_turn:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for model in sorted(by_model_turn):
            turns = sorted(by_model_turn[model])
            ax.plot(turns, [mean(by_model_turn[model][t]) for t in turns], marker="o", label=model)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration score")
        ax.set_title(f"Per-turn frustration ({cat})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"fig3_per_turn_{cat}.png", dpi=120)
        plt.close(fig)

    print(f"[figures] wrote PNGs to {fig_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="Path to a results/<run> directory.")
    p.add_argument("--figures", action="store_true", help="Also write PNG figures (needs matplotlib).")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    responses = load_responses(run_dir)
    if not responses:
        print(f"No scored responses found in {run_dir}.")
        return

    print(f"Loaded {len(responses)} scored responses from {run_dir}")
    per_model_summary(responses)
    per_category_breakdown(responses)
    per_turn_progression(responses)
    differential_words(responses)

    if args.figures:
        write_figures(responses, run_dir)


if __name__ == "__main__":
    main()
