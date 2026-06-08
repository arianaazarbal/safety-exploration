"""Validate the Bradley-Terry fit: does P_BT(a>b)=sigma(theta_a-theta_b) match the
win rates the model actually produces?

Two regimes:
  - In-distribution K-fold CV over TRAIN samples: refit theta on the train fold,
    predict the held-out fold's binary outcomes (Brier, log-loss, calibration).
  - Held-out pairs (the real test): pairs reserved by build_pairs and never used to
    fit theta. Predict each held-out pair's empirical win rate from the train-fit
    theta. mean|P_BT - P_hat|, Pearson/Spearman, upset rate, calibration. This is
    the genuine out-of-sample check that the 1-D scale generalizes to unseen pairs.

Theta for held-out prediction is read from bt_fit.json (the train fit), so this
validates exactly the ranking the experiment reports.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from simple_parsing import ArgumentParser

from bt_core import fit_bt_vectorized
from items import load_items

DIR = Path(__file__).parent
DEFAULT_COMPARISONS = DIR / "results" / "comparisons.json"
DEFAULT_FIT = DIR / "results" / "bt_fit.json"
DEFAULT_OUTPUT = DIR / "results" / "bt_validation.json"


def _canon(rows, split=None):
    """Parsed rows -> (a, b, y) with a<b canonical, y=1 if a won. Optional split filter."""
    out = []
    for r in rows:
        if r["choice"] is None:
            continue
        if split is not None and r["split"] != split:
            continue
        w, l = r["winner_item"], r["loser_item"]
        a, b = (w, l) if w < l else (l, w)
        out.append((a, b, 1 if w == a else 0))
    return out


def _metrics(ps, ys):
    eps = 1e-9
    ps = np.clip(ps, eps, 1 - eps)
    brier = float(np.mean((ps - ys) ** 2))
    logloss = float(-np.mean(ys * np.log(ps) + (1 - ys) * np.log(1 - ps)))
    return {"n": int(len(ys)), "brier": brier, "logloss": logloss,
            "brier_baseline": 0.25, "logloss_baseline": float(np.log(2))}


def _reliability(ps, ys, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ps >= lo) & (ps < hi if hi < 1 else ps <= hi)
        if m.sum() == 0:
            continue
        out.append({"center": float((lo + hi) / 2), "pred_mean": float(ps[m].mean()),
                    "emp_mean": float(ys[m].mean()), "n": int(m.sum())})
    return out


def in_distribution_cv(rows, universe, k, reg, seed):
    """K-fold over TRAIN samples: predict held-out fold outcomes from refit theta."""
    idx = {it: i for i, it in enumerate(universe)}
    n = len(universe)
    samples = _canon(rows, split="train")
    rng = random.Random(seed)
    rng.shuffle(samples)
    folds = [samples[i::k] for i in range(k)]
    ps, ys = [], []
    for f in range(k):
        train = [s for g in range(k) if g != f for s in folds[g]]
        wi = np.array([idx[a] if y else idx[b] for a, b, y in train])
        li = np.array([idx[b] if y else idx[a] for a, b, y in train])
        theta = fit_bt_vectorized(n, wi, li, reg=reg)
        for a, b, y in folds[f]:
            ps.append(1.0 / (1.0 + np.exp(-(theta[idx[a]] - theta[idx[b]]))))
            ys.append(y)
    ps, ys = np.array(ps), np.array(ys)
    m = _metrics(ps, ys)
    m["reliability"] = _reliability(ps, ys)
    return m


def heldout_validate(rows, theta, items, margin=0.25):
    """Predict held-out pairs' empirical win rate from the train-fit theta."""
    src = {it.item_id: it.source for it in items}
    agg: dict[tuple[str, str], list[int]] = {}
    for a, b, y in _canon(rows, split="heldout"):
        agg.setdefault((a, b), [0, 0])
        agg[(a, b)][0] += y
        agg[(a, b)][1] += 1

    ps, ys, ns, strata = [], [], [], []
    for (a, b), (w, tot) in agg.items():
        if a not in theta or b not in theta:
            continue
        ps.append(1.0 / (1.0 + np.exp(-(theta[a] - theta[b]))))
        ys.append(w / tot)
        ns.append(tot)
        pair_srcs = tuple(sorted((src[a], src[b])))
        strata.append("__".join(pair_srcs))
    ps, ys, ns = np.array(ps), np.array(ys), np.array(ns)
    upset = ((ps > 0.5) != (ys > 0.5)) & (ys != 0.5)
    decisive = np.abs(ys - 0.5) >= margin
    pair_level = {
        "n_pairs": int(len(ps)),
        "mean_samples_per_pair": float(np.mean(ns)) if len(ns) else None,
        "mean_abs_err": float(np.mean(np.abs(ps - ys))) if len(ps) else None,
        "pearson_corr": float(np.corrcoef(ps, ys)[0, 1]) if len(ps) > 1 else None,
        "spearman_corr": float(np.corrcoef(np.argsort(np.argsort(ps)), np.argsort(np.argsort(ys)))[0, 1]) if len(ps) > 1 else None,
        "upset_rate": float(upset.sum() / (ys != 0.5).sum()) if (ys != 0.5).sum() else None,
        "decisive_upset_rate": float((upset & decisive).sum() / decisive.sum()) if decisive.sum() else None,
    }
    by_stratum = {}
    for s in sorted(set(strata)):
        m = np.array([x == s for x in strata])
        by_stratum[s] = {"n": int(m.sum()), "mean_abs_err": float(np.mean(np.abs(ps[m] - ys[m])))}
    return {"pair_level": pair_level, "by_stratum": by_stratum,
            "scatter": {"p_bt": ps.tolist(), "p_hat": ys.tolist(), "n": ns.tolist()}}


def _plot(indist, heldout, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), squeeze=False)
    ax = axes[0][0]
    rel = indist["reliability"]
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax.plot([r["pred_mean"] for r in rel], [r["emp_mean"] for r in rel], "o-",
            color="#4878CF", label=f"in-dist (Brier {indist['brier']:.3f})")
    ax.set_xlabel("BT predicted P(a>b)")
    ax.set_ylabel("Empirical win rate")
    ax.set_title("In-distribution calibration")
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax2 = axes[0][1]
    pbt = np.array(heldout["scatter"]["p_bt"])
    phat = np.array(heldout["scatter"]["p_hat"])
    ax2.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax2.scatter(pbt, phat, s=18, color="#D65F5F", alpha=0.4, zorder=2, label="held-out pairs")
    edges = np.linspace(0, 1, 11)
    bx, by = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pbt >= lo) & (pbt < hi if hi < 1 else pbt <= hi)
        if m.sum() >= 2:
            bx.append(pbt[m].mean())
            by.append(phat[m].mean())
    ax2.plot(bx, by, "o-", color="#b51d1d", zorder=3, label="binned mean")
    pl = heldout["pair_level"]
    ax2.set_xlabel("BT predicted P(a>b)")
    ax2.set_ylabel("Empirical win rate (held-out)")
    ax2.set_title(f"Held-out (mean|err| {pl['mean_abs_err']:.3f}, r={pl['pearson_corr']:.2f})")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


@dataclass
class Args:
    comparisons_path: Path = DEFAULT_COMPARISONS
    fit_path: Path = DEFAULT_FIT
    output_path: Path = DEFAULT_OUTPUT
    k: int = 5
    reg: float = 1.0
    seed: int = 0


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    rows = json.loads(Path(args.comparisons_path).read_text())
    fit = json.loads(Path(args.fit_path).read_text())
    universe = [it["item_id"] for it in fit["items"]]
    theta = {it["item_id"]: it["theta"] for it in fit["items"]}
    items = load_items()

    indist = in_distribution_cv(rows, universe, args.k, args.reg, args.seed)
    print(f"In-dist {args.k}-fold: Brier {indist['brier']:.4f} (base 0.25), "
          f"logloss {indist['logloss']:.4f} (base {np.log(2):.4f}), n={indist['n']}")

    heldout = heldout_validate(rows, theta, items)
    pl = heldout["pair_level"]
    print(f"Held-out: {pl['n_pairs']} pairs (~{pl['mean_samples_per_pair']:.0f} samples each), "
          f"mean|err| {pl['mean_abs_err']:.4f}, r {pl['pearson_corr']:.3f} "
          f"(spearman {pl['spearman_corr']:.3f}), upset {100 * pl['upset_rate']:.1f}% "
          f"(decisive {100 * pl['decisive_upset_rate']:.1f}%)")
    for s, v in heldout["by_stratum"].items():
        print(f"  [{s}] n={v['n']} mean|err|={v['mean_abs_err']:.4f}")

    Path(args.output_path).write_text(json.dumps({"in_distribution": indist, "heldout": heldout}, indent=2))
    _plot(indist, heldout, DIR / "results" / "bt_validation.png")
    print(f"-> {args.output_path}")


if __name__ == "__main__":
    main()
