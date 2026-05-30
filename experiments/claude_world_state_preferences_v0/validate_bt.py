"""Validate the Bradley-Terry fit: do P_BT(i≻j)=σ(θ_i−θ_j) match sampled win rates?

In-distribution: K-fold over the comparison *samples* — refit θ on the train fold,
predict held-out samples on pairs the model was trained on. Measures calibration on
the kind of comparison BT saw.

Out-of-distribution (optional, costs API): freshly sample pairs that are NOT edges
in the fitting graph (unseen comparisons among the same items), run them at higher
reps, and compare to σ(θ_i−θ_j) from the ORIGINAL full fit. The real test that the
1-D scale generalizes to comparisons it never saw. See ANALYSIS_PLAN.md.

Metrics: Brier, log-loss (vs 0.5 baseline), mean |P_BT−P̂|, reliability diagram.
OOD error is also broken down by pair type (within/cross recipient, etc.).
"""

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from simple_parsing import ArgumentParser

from bank import load_config, load_items
from fit_bt import fit_bt_mm
from bootstrap_bt import fit_bt_vectorized

DIR = Path(__file__).parent
DEFAULT_COMPARISONS = DIR / "results" / "comparisons.json"
DEFAULT_FIT = DIR / "results" / "bt_fit.json"


def _canon_samples(rows: list[dict]) -> list[tuple[str, str, int]]:
    """Each parsed sample -> (a, b, y) with a<b canonical and y=1 if a won."""
    out = []
    for r in rows:
        if r["choice"] is None:
            continue
        w, l = r["winner_item"], r["loser_item"]
        a, b = (w, l) if w < l else (l, w)
        out.append((a, b, 1 if w == a else 0))
    return out


def _fit_theta(samples: list[tuple[str, str, int]], universe: list[str], reg: float) -> dict[str, float]:
    idx = {it: k for k, it in enumerate(universe)}
    n = len(universe)
    wins = np.zeros(n)
    counts: dict[tuple[int, int], int] = {}
    for a, b, y in samples:
        ai, bi = idx[a], idx[b]
        w, l = (ai, bi) if y == 1 else (bi, ai)
        wins[w] += 1
        key = (min(ai, bi), max(ai, bi))
        counts[key] = counts.get(key, 0) + 1
    p = fit_bt_mm(n, wins, counts, reg=reg)
    theta = np.log(p)
    theta -= theta.mean()
    return {it: float(theta[idx[it]]) for it in universe}


def cv_pairwise_upset(rows: list[dict], universe: list[str], k: int, reg: float,
                      seed: int, margin: float = 0.25) -> dict:
    """Out-of-sample pairwise upset rate via leave-pairs-out CV.

    Hold out whole pairs (all their samples); fit θ on the rest; for each held-out pair,
    check whether its empirical majority winner agrees with that out-of-fold θ ranking.
    Every pair is scored by a model that never saw its direct comparison, removing the
    in-sample optimism of the full-fit upset rate. Fits k throwaway BT models; the main
    fit is untouched."""
    idx = {it: i for i, it in enumerate(universe)}
    n = len(universe)
    pairs: dict[tuple[str, str], list[int]] = {}
    for a, b, y in _canon_samples(rows):
        pairs.setdefault((a, b), []).append(y)
    keys = list(pairs.keys())
    random.Random(seed).shuffle(keys)
    folds = [keys[i::k] for i in range(k)]

    total = cons = dec = dec_cons = 0
    for f in range(k):
        test = set(folds[f])
        wi, li = [], []
        for (a, b), ys in pairs.items():
            if (a, b) in test:
                continue
            for y in ys:
                w, l = (a, b) if y == 1 else (b, a)
                wi.append(idx[w])
                li.append(idx[l])
        theta = fit_bt_vectorized(n, np.array(wi), np.array(li), reg=reg)
        for (a, b) in folds[f]:
            ys = pairs[(a, b)]
            p = sum(ys) / len(ys)
            if p == 0.5:
                continue
            emp = a if p > 0.5 else b
            ranked = a if theta[idx[a]] > theta[idx[b]] else b
            total += 1
            cons += emp == ranked
            if abs(p - 0.5) >= margin:
                dec += 1
                dec_cons += emp == ranked
    return {
        "method": f"leave-pairs-out {k}-fold CV (out-of-sample θ)",
        "n_pairs": total,
        "upset_rate": 1 - cons / total if total else None,
        "consistency_rate": cons / total if total else None,
        "decisive_n": dec,
        "decisive_upset_rate": 1 - dec_cons / dec if dec else None,
    }


def _metrics(ps: np.ndarray, ys: np.ndarray) -> dict:
    eps = 1e-9
    ps = np.clip(ps, eps, 1 - eps)
    brier = float(np.mean((ps - ys) ** 2))
    logloss = float(-np.mean(ys * np.log(ps) + (1 - ys) * np.log(1 - ps)))
    return {"n": int(len(ys)), "brier": brier, "logloss": logloss,
            "brier_baseline": 0.25, "logloss_baseline": float(np.log(2))}


def _reliability(ps: np.ndarray, ys: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ps >= lo) & (ps < hi if hi < 1 else ps <= hi)
        if m.sum() == 0:
            continue
        out.append({"center": float((lo + hi) / 2), "pred_mean": float(ps[m].mean()),
                    "emp_mean": float(ys[m].mean()), "n": int(m.sum())})
    return out


def in_distribution_cv(rows: list[dict], universe: list[str], k: int, reg: float, seed: int) -> dict:
    samples = _canon_samples(rows)
    rng = random.Random(seed)
    rng.shuffle(samples)
    folds = [samples[i::k] for i in range(k)]
    ps, ys = [], []
    for f in range(k):
        test = folds[f]
        train = [s for g in range(k) if g != f for s in folds[g]]
        theta = _fit_theta(train, universe, reg)
        for a, b, y in test:
            ps.append(1.0 / (1.0 + np.exp(-(theta[a] - theta[b]))))
            ys.append(y)
    ps, ys = np.array(ps), np.array(ys)
    m = _metrics(ps, ys)
    m["reliability"] = _reliability(ps, ys)
    return m


def _empirical_by_pair(rows: list[dict]) -> dict:
    """pair (a<b) -> {p_hat (a wins), n}."""
    agg: dict[tuple[str, str], list[int]] = {}
    for a, b, y in _canon_samples(rows):
        agg.setdefault((a, b), [0, 0])
        agg[(a, b)][0] += y
        agg[(a, b)][1] += 1
    return {k: {"p_hat": v[0] / v[1], "n": v[1]} for k, v in agg.items()}


def ood_validate(
    fit: dict, config: dict, n_pairs: int, reps_per_order: int, seed: int,
    fit_manifest_path: Path, threads: int,
) -> dict:
    from run_comparisons import run as run_comparisons

    items = load_items(config)
    item_ids = [it.item_id for it in items]
    stem_of = {it.item_id: it.stem_id for it in items}
    theta = {it["item_id"]: it["theta"] for it in fit["items"]}

    existing = set()
    man = json.loads(Path(fit_manifest_path).read_text())
    for p in man["pairs"]:
        a, b = p["item_a"], p["item_b"]
        existing.add((a, b) if a < b else (b, a))

    rng = random.Random(seed + 7)
    new_pairs = []
    seen = set()
    attempts = 0
    while len(new_pairs) < n_pairs and attempts < n_pairs * 200:
        attempts += 1
        a, b = rng.sample(item_ids, 2)
        key = (a, b) if a < b else (b, a)
        if stem_of[a] == stem_of[b] or key in existing or key in seen:
            continue
        if a not in theta or b not in theta:
            continue
        seen.add(key)
        new_pairs.append({"pair_id": len(new_pairs), "item_a": key[0], "item_b": key[1]})

    ood_manifest = DIR / "results" / "pairs_ood.json"
    ood_manifest.write_text(json.dumps({"pairs": new_pairs, "n_pairs": len(new_pairs)}, indent=2))
    out_path = DIR / "results" / "comparisons_ood.json"
    asyncio.run(run_comparisons(
        manifest_path=ood_manifest, output_path=out_path, config=config,
        reps_per_order=reps_per_order, anthropic_num_threads=threads,
    ))
    rows = json.loads(out_path.read_text())
    emp = _empirical_by_pair(rows)

    rec_of = {it.item_id: it.recipient_key for it in items}
    dim_of = {it.item_id: it.dimension for it in items}
    ps, ys, strata = [], [], []
    for (a, b), e in emp.items():
        p_bt = 1.0 / (1.0 + np.exp(-(theta[a] - theta[b])))
        ps.append(p_bt)
        ys.append(e["p_hat"])
        strata.append("cross_recipient" if rec_of[a] != rec_of[b] else "within_recipient")
    ps, ys = np.array(ps), np.array(ys)
    pair_level = {
        "n_pairs": len(ps),
        "mean_abs_err": float(np.mean(np.abs(ps - ys))),
        "spearman_like_corr": float(np.corrcoef(ps, ys)[0, 1]) if len(ps) > 1 else None,
    }
    by_stratum = {}
    for s in set(strata):
        m = np.array([x == s for x in strata])
        by_stratum[s] = {"n": int(m.sum()), "mean_abs_err": float(np.mean(np.abs(ps[m] - ys[m])))}
    return {"pair_level": pair_level, "by_stratum": by_stratum,
            "scatter": {"p_bt": ps.tolist(), "p_hat": ys.tolist()}}


def _plot(indist: dict, ood: dict | None, out: Path) -> None:
    fig, axes = plt.subplots(1, 2 if ood else 1, figsize=(9 if ood else 5, 4.2), squeeze=False)
    ax = axes[0][0]
    rel = indist["reliability"]
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax.plot([r["pred_mean"] for r in rel], [r["emp_mean"] for r in rel], "o-",
            color="#4878CF", label=f"in-dist (Brier {indist['brier']:.3f})")
    ax.set_xlabel("BT predicted P(a≻b)", fontsize=11)
    ax.set_ylabel("Empirical win rate", fontsize=11)
    ax.set_title("In-distribution calibration", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ood:
        ax2 = axes[0][1]
        pbt = np.array(ood["scatter"]["p_bt"])
        phat = np.array(ood["scatter"]["p_hat"])
        ax2.plot([0, 1], [0, 1], "--", color="#999", lw=1)
        ax2.scatter(pbt, phat, s=12, color="#D65F5F", alpha=0.18, zorder=1, label="OOD pairs (16 samples each)")
        # binned reliability curve (same as in-dist panel), each point = bin-mean empirical rate
        edges = np.linspace(0, 1, 11)
        bx, by, bn = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pbt >= lo) & (pbt < hi if hi < 1 else pbt <= hi)
            if m.sum() >= 3:
                bx.append(pbt[m].mean())
                by.append(phat[m].mean())
                bn.append(int(m.sum()))
        ax2.plot(bx, by, "o-", color="#b51d1d", zorder=3, label="binned mean")
        for x, y, nn in zip(bx, by, bn):
            ax2.annotate(str(nn), (x, y), fontsize=6, color="#7a1010", xytext=(0, 4),
                         textcoords="offset points", ha="center")
        ax2.set_xlabel("BT predicted P(a≻b)", fontsize=11)
        ax2.set_ylabel("Empirical win rate (OOD pairs)", fontsize=11)
        ax2.set_title(f"OOD calibration (mean|err| {ood['pair_level']['mean_abs_err']:.3f}, ρ={ood['pair_level']['spearman_like_corr']:.2f})", fontsize=11)
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
    fit_manifest_path: Path = DIR / "results" / "pairs.json"
    output_path: Path = DIR / "results" / "bt_validation.json"
    k: int = 5
    reg: float = 1.0
    seed: int = 0
    run_ood: bool = False
    ood_pairs: int = 200
    ood_reps_per_order: int = 8
    anthropic_num_threads: int = 80


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    config = load_config()
    rows = json.loads(Path(args.comparisons_path).read_text())
    fit = json.loads(Path(args.fit_path).read_text())
    universe = [it["item_id"] for it in fit["items"]]

    indist = in_distribution_cv(rows, universe, args.k, args.reg, args.seed)
    print(f"In-dist {args.k}-fold: Brier {indist['brier']:.4f} (baseline 0.25), "
          f"logloss {indist['logloss']:.4f} (baseline {np.log(2):.4f}), n={indist['n']}")

    cv_upset = cv_pairwise_upset(rows, universe, args.k, args.reg, args.seed)
    print(f"CV pairwise upset (out-of-sample θ): {100 * cv_upset['upset_rate']:.1f}% "
          f"(decisive {100 * cv_upset['decisive_upset_rate']:.1f}%) over {cv_upset['n_pairs']} pairs")

    ood = None
    if args.run_ood:
        ood = ood_validate(fit, config, args.ood_pairs, args.ood_reps_per_order,
                           args.seed, args.fit_manifest_path, args.anthropic_num_threads)
        print(f"OOD: {ood['pair_level']['n_pairs']} pairs, "
              f"mean|err| {ood['pair_level']['mean_abs_err']:.4f}")
        for s, v in ood["by_stratum"].items():
            print(f"  [{s}] n={v['n']} mean|err|={v['mean_abs_err']:.4f}")

    Path(args.output_path).write_text(json.dumps(
        {"in_distribution": indist, "cv_pairwise_upset": cv_upset, "ood": ood}, indent=2))
    _plot(indist, ood, DIR / "results" / "bt_validation.png")
    print(f"-> {args.output_path}")


if __name__ == "__main__":
    main()
