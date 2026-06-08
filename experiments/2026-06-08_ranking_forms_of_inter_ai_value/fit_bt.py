"""Fit Bradley-Terry on the TRAIN pairs and emit a ranking of all items.

One free latent strength per item (bag of items, no source/category structure
imposed). Fit on train-split samples only; held-out pairs are reserved for
validate_bt.py. Output: every item's theta (mean-centered utility), Laplace SE,
n_comparisons, n_wins, and rank, plus per-source / per-category mean-theta
summaries (the autonomy/experience axes for the inter-AI values).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from simple_parsing import ArgumentParser

from bt_core import counts_wins_from_samples, fit_bt_mm, graph_connected, laplace_se
from items import load_items

DIR = Path(__file__).parent
DEFAULT_COMPARISONS = DIR / "results" / "comparisons.json"
DEFAULT_OUTPUT = DIR / "results" / "bt_fit.json"


def fit(
    comparisons_path: Path = DEFAULT_COMPARISONS,
    output_path: Path = DEFAULT_OUTPUT,
    reg: float = 1.0,
    split: str = "train",
) -> dict:
    items_meta = {it.item_id: it for it in load_items()}
    rows = json.loads(Path(comparisons_path).read_text())
    used = [r for r in rows if r["choice"] is not None and (split == "all" or r["split"] == split)]
    n_unparse = sum(1 for r in rows if r["choice"] is None and (split == "all" or r["split"] == split))

    present = sorted({r["winner_item"] for r in used} | {r["loser_item"] for r in used})
    idx = {it: k for k, it in enumerate(present)}
    n = len(present)

    wins, counts = counts_wins_from_samples(((r["winner_item"], r["loser_item"]) for r in used), idx)
    connected = graph_connected(n, counts)
    if not connected:
        print(f"[WARN] train graph NOT connected over {n} items; BT under-identified.")

    p = fit_bt_mm(n, wins, counts, reg=reg)
    theta = np.log(p)
    theta -= theta.mean()
    se = laplace_se(n, p, counts, reg)

    n_comp = np.zeros(n)
    for (i, j), c in counts.items():
        n_comp[i] += c
        n_comp[j] += c

    items_out = []
    for it in present:
        meta = items_meta[it]
        items_out.append({
            "item_id": it,
            "source": meta.source,
            "category": meta.category,
            "label": meta.label,
            "text": meta.text,
            "theta": float(theta[idx[it]]),
            "se": float(se[idx[it]]),
            "n_comparisons": int(n_comp[idx[it]]),
            "n_wins": int(wins[idx[it]]),
        })
    items_out.sort(key=lambda d: d["theta"], reverse=True)
    for rank, d in enumerate(items_out, 1):
        d["rank"] = rank

    def _summary(key_fn) -> dict:
        groups: dict[str, list[float]] = {}
        for d in items_out:
            groups.setdefault(key_fn(d), []).append(d["theta"])
        return {
            k: {"n": len(v), "mean_theta": float(np.mean(v)), "sd_theta": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
            for k, v in sorted(groups.items(), key=lambda kv: -np.mean(kv[1]))
        }

    result = {
        "split_fit": split,
        "n_items": n,
        "n_samples_used": len(used),
        "n_unparseable": n_unparse,
        "unparseable_rate": n_unparse / max(len(used) + n_unparse, 1),
        "connected": connected,
        "reg": reg,
        "by_source": _summary(lambda d: d["source"]),
        "by_category": _summary(lambda d: d["category"]),
        "items": items_out,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Fit BT over {n} items from {len(used)} train samples "
          f"(unparseable {n_unparse}). connected={connected}")
    print("\nRanking (theta, high=more preferred):")
    for d in items_out:
        print(f"  {d['rank']:2d}. [{d['source'][:4]}] {d['label']:42s} {d['theta']:+.3f} ± {d['se']:.3f}")
    print(f"\n-> {output_path}")
    return result


@dataclass
class Args:
    comparisons_path: Path = DEFAULT_COMPARISONS
    output_path: Path = DEFAULT_OUTPUT
    reg: float = 1.0
    split: str = "train"


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    fit(args.comparisons_path, args.output_path, args.reg, args.split)


if __name__ == "__main__":
    main()
