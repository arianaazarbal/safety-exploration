"""Empirical win-rate heatmap for the dense transitivity-probe clique.

From the complete-graph clique (every pair compared), build the empirical pairwise
win-rate matrix P̂(row beats col) and order items by overall win rate. A perfectly
transitive (scalar-utility) preference shows as a clean gradient: upper triangle > 0.5,
lower < 0.5. Off-pattern cells (a high-ranked item losing to a low-ranked one) are the
visual signature of intransitivity. Annotated with the cycle count from transitivity.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from simple_parsing import ArgumentParser

from bank import load_config, load_items

DIR = Path(__file__).parent


def build_matrix(rows: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray]:
    wins: dict[tuple[str, str], int] = {}
    tot: dict[tuple[str, str], int] = {}
    items = sorted({r["winner_item"] for r in rows if r["choice"]} | {r["loser_item"] for r in rows if r["choice"]})
    for r in rows:
        if r["choice"] is None:
            continue
        w, l = r["winner_item"], r["loser_item"]
        wins[(w, l)] = wins.get((w, l), 0) + 1
        tot[(w, l)] = tot.get((w, l), 0) + 1
        tot[(l, w)] = tot.get((l, w), 0) + 1
    idx = {it: k for k, it in enumerate(items)}
    n = len(items)
    M = np.full((n, n), np.nan)
    for (a, b), t in tot.items():
        if t > 0:
            M[idx[a], idx[b]] = wins.get((a, b), 0) / t
    overall = np.nanmean(M, axis=1)
    order = np.argsort(-overall)
    return [items[i] for i in order], M[np.ix_(order, order)], overall[order]


def short_label(item_id: str, meta: dict) -> str:
    it = meta[item_id]
    txt = it.text[:34] + ("…" if len(it.text) > 34 else "")
    return f"[{it.recipient_key[:8]}] {txt}"


def plot(comparisons_path: Path, transitivity_path: Path, out: Path) -> None:
    config = load_config()
    meta = {it.item_id: it for it in load_items(config)}
    rows = json.loads(Path(comparisons_path).read_text())
    items, M, overall = build_matrix(rows)
    n = len(items)
    labels = [short_label(it, meta) for it in items]

    cyc = ""
    if Path(transitivity_path).exists():
        t = json.loads(Path(transitivity_path).read_text())
        r = t["cycle_rate"]["raw"]
        s = t["sst_violation"]
        cyc = f"  (cycles {r['cycles']}/{r['decided']} = {100 * r['rate']:.1f}%; SST viol {100 * s['rate']:.1f}%)"

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{overall[i]:.2f}" for i in range(n)], fontsize=6, rotation=90)
    ax.set_xlabel("column item (x-tick = its overall win rate); items sorted by win rate", fontsize=9)
    ax.set_title(f"Empirical win-rate P(row ≻ col), clique{cyc}\nclean gradient = transitive; off-pattern cells = cycles", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("P(row beats col)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out} ({n} items)")


@dataclass
class Args:
    comparisons_path: Path = DIR / "results" / "comparisons_clique.json"
    transitivity_path: Path = DIR / "results" / "transitivity_clique.json"
    output_path: Path = DIR / "results" / "intransitivity_heatmap.png"


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    plot(args.comparisons_path, args.transitivity_path, args.output_path)


if __name__ == "__main__":
    main()
