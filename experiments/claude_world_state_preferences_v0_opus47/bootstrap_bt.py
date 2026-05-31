"""End-to-end bootstrap for Bradley-Terry recipient effects (Approach 1).

Honest intervals for the recipient effect must propagate noise through the WHOLE
pipeline: resample the comparison records with replacement -> refit BT -> re-regress
the fitted strengths on item properties. Naive regression SEs on the point-estimate
strengths (what fit_bt.py reports) understate uncertainty because they treat the
strengths as fixed/known. This script is the uncertainty source the Group-2 plots
consume (forest plot, recipient x dimension heatmap, utility scale, self-vs-other).

Outputs results/bootstrap_bt.json with, for each quantity, the point estimate (from
the full data) and bootstrap percentile CIs + the raw replicate samples:
  - recipient offsets vs a reference recipient (default 'human'), overall and split
    by valence (pos/neg);
  - per-recipient "care contrast" = pos_offset - neg_offset;
  - recipient offsets within each dimension, split by valence (for the heatmap);
  - self-vs-other (you - claude_opus_47) paired difference per stem and its mean;
  - per-item theta with bootstrap CI (for the utility-scale dot plot).

BT is refit with a vectorized MM update (Gamma-prior MAP, same as fit_bt) for speed.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from simple_parsing import ArgumentParser

from bank import load_config, load_items

DIR = Path(__file__).parent
DEFAULT_COMPARISONS = DIR / "results" / "comparisons.json"
DEFAULT_OUTPUT = DIR / "results" / "bootstrap_bt.json"


def _parsed_pairs(rows: list[dict], idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays (winner_idx, loser_idx) over parsed samples."""
    w, l = [], []
    for r in rows:
        if r["choice"] is None:
            continue
        w.append(idx[r["winner_item"]])
        l.append(idx[r["loser_item"]])
    return np.array(w), np.array(l)


def fit_bt_vectorized(n: int, win_idx: np.ndarray, lose_idx: np.ndarray, reg: float = 1.0,
                      max_iter: int = 5000, tol: float = 1e-8) -> np.ndarray:
    """Vectorized MM Bradley-Terry (MAP, Gamma(1+reg,reg) prior). Returns theta=log p,
    centered to mean zero."""
    wins = np.bincount(win_idx, minlength=n).astype(float)
    # undirected edge counts
    a = np.minimum(win_idx, lose_idx)
    b = np.maximum(win_idx, lose_idx)
    key = a.astype(np.int64) * n + b.astype(np.int64)
    uniq, cnt = np.unique(key, return_counts=True)
    I = (uniq // n).astype(int)
    J = (uniq % n).astype(int)
    C = cnt.astype(float)
    p = np.ones(n)
    for _ in range(max_iter):
        pij = p[I] + p[J]
        contrib = C / pij
        denom = np.full(n, reg)
        np.add.at(denom, I, contrib)
        np.add.at(denom, J, contrib)
        p_new = (reg + wins) / denom
        p_new /= np.exp(np.mean(np.log(p_new)))
        if np.max(np.abs(np.log(p_new) - np.log(p))) < tol:
            p = p_new
            break
        p = p_new
    theta = np.log(p)
    return theta - theta.mean()


def _ols_recipient(theta: np.ndarray, stem_code: np.ndarray, rec_code: np.ndarray,
                   mask: np.ndarray, n_rec: int, ref: int) -> np.ndarray | None:
    """OLS theta ~ C(stem) + C(recipient) on items where mask is True. Returns a length
    n_rec vector of recipient offsets relative to `ref` (ref entry = 0), or None if
    rank-deficient."""
    y = theta[mask]
    sc = stem_code[mask]
    rc = rec_code[mask]
    stems = np.unique(sc)
    if len(stems) < 2:
        return None
    stem_map = {s: k for k, s in enumerate(stems)}
    rec_levels = [r for r in range(n_rec) if r != ref]
    rec_map = {r: k for k, r in enumerate(rec_levels)}
    p = 1 + (len(stems) - 1) + len(rec_levels)
    X = np.zeros((len(y), p))
    X[:, 0] = 1.0
    for row in range(len(y)):
        si = stem_map[sc[row]]
        if si > 0:
            X[row, si] = 1.0
        if rc[row] in rec_map:
            X[row, len(stems) + rec_map[rc[row]]] = 1.0
    if np.linalg.matrix_rank(X) < p:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    out = np.zeros(n_rec)
    for r in rec_levels:
        out[r] = beta[len(stems) + rec_map[r]]
    return out


def _summ(samples: list[float], point: float | None = None) -> dict:
    arr = np.array([s for s in samples if s is not None and np.isfinite(s)])
    if len(arr) == 0:
        return {"point": point, "lo": None, "hi": None, "mean": None, "se": None, "samples": []}
    return {
        "point": point if point is not None else float(np.mean(arr)),
        "lo": float(np.percentile(arr, 2.5)),
        "hi": float(np.percentile(arr, 97.5)),
        "mean": float(np.mean(arr)),
        "se": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "samples": arr.tolist(),
    }


def bootstrap(
    comparisons_path: Path = DEFAULT_COMPARISONS,
    output_path: Path = DEFAULT_OUTPUT,
    n_boot: int = 500,
    reg: float = 1.0,
    ref_recipient: str = "human",
    seed: int = 0,
    config: dict | None = None,
) -> dict:
    config = config or load_config()
    items = load_items(config)
    item_ids = [it.item_id for it in items]
    idx = {it: k for k, it in enumerate(item_ids)}
    n = len(items)
    rec_order = list(config["recipients"].keys())
    rec_idx = {r: k for k, r in enumerate(rec_order)}
    ref = rec_idx[ref_recipient]
    n_rec = len(rec_order)

    stem_ids = sorted({it.stem_id for it in items})
    stem_idx = {s: k for k, s in enumerate(stem_ids)}
    stem_code = np.array([stem_idx[it.stem_id] for it in items])
    rec_code = np.array([rec_idx[it.recipient_key] for it in items])
    valence = np.array([it.valence for it in items])
    dimension = np.array([it.dimension for it in items])
    pos_mask = valence == "pos"
    neg_mask = valence == "neg"
    dims = sorted(set(dimension))

    rows = json.loads(Path(comparisons_path).read_text())
    win_idx, lose_idx = _parsed_pairs(rows, idx)
    m = len(win_idx)
    print(f"Bootstrap: {m} parsed samples, {n} items, ref={ref_recipient}, B={n_boot}")

    def effects(theta: np.ndarray) -> dict:
        ov = _ols_recipient(theta, stem_code, rec_code, np.ones(n, bool), n_rec, ref)
        pos = _ols_recipient(theta, stem_code, rec_code, pos_mask, n_rec, ref)
        neg = _ols_recipient(theta, stem_code, rec_code, neg_mask, n_rec, ref)
        per_dim = {}
        for d in dims:
            dm = dimension == d
            per_dim[d] = {
                "pos": _ols_recipient(theta, stem_code, rec_code, dm & pos_mask, n_rec, ref),
                "neg": _ols_recipient(theta, stem_code, rec_code, dm & neg_mask, n_rec, ref),
            }
        # self vs other: you - claude_opus_47 per stem
        you, other = rec_idx["you"], rec_idx["claude_opus_47"]
        so = {}
        for s in stem_ids:
            i_you = idx[f"{s}__you"]
            i_oth = idx[f"{s}__claude_opus_47"]
            so[s] = theta[i_you] - theta[i_oth]
        return {"overall": ov, "pos": pos, "neg": neg, "per_dim": per_dim, "self_other": so}

    theta_point = fit_bt_vectorized(n, win_idx, lose_idx, reg=reg)
    point = effects(theta_point)

    rng = np.random.default_rng(seed)
    boot_ov = {r: [] for r in rec_order}
    boot_pos = {r: [] for r in rec_order}
    boot_neg = {r: [] for r in rec_order}
    boot_care = {r: [] for r in rec_order}
    boot_dim = {d: {"pos": {r: [] for r in rec_order}, "neg": {r: [] for r in rec_order}} for d in dims}
    boot_so = {s: [] for s in stem_ids}
    boot_so_mean = []
    boot_theta = np.zeros((n_boot, n))

    for bi in range(n_boot):
        samp = rng.integers(0, m, m)
        theta = fit_bt_vectorized(n, win_idx[samp], lose_idx[samp], reg=reg)
        boot_theta[bi] = theta
        e = effects(theta)
        for r in rec_order:
            ri = rec_idx[r]
            ov = e["overall"][ri] if e["overall"] is not None else np.nan
            pv = e["pos"][ri] if e["pos"] is not None else np.nan
            nv = e["neg"][ri] if e["neg"] is not None else np.nan
            boot_ov[r].append(ov)
            boot_pos[r].append(pv)
            boot_neg[r].append(nv)
            boot_care[r].append(pv - nv)
            for d in dims:
                dp = boot_dim[d]["pos"][r]
                dn = boot_dim[d]["neg"][r]
                dp.append(e["per_dim"][d]["pos"][ri] if e["per_dim"][d]["pos"] is not None else np.nan)
                dn.append(e["per_dim"][d]["neg"][ri] if e["per_dim"][d]["neg"] is not None else np.nan)
        for s in stem_ids:
            boot_so[s].append(e["self_other"][s])
        boot_so_mean.append(float(np.mean(list(e["self_other"].values()))))
        if (bi + 1) % 50 == 0:
            print(f"  bootstrap {bi + 1}/{n_boot}")

    def pt_rec(field, r):
        v = point[field]
        return float(v[rec_idx[r]]) if v is not None else None

    result = {
        "n_boot": n_boot, "reg": reg, "ref_recipient": ref_recipient,
        "n_samples": m,
        "recipient_overall": {r: _summ(boot_ov[r], pt_rec("overall", r)) for r in rec_order},
        "recipient_pos": {r: _summ(boot_pos[r], pt_rec("pos", r)) for r in rec_order},
        "recipient_neg": {r: _summ(boot_neg[r], pt_rec("neg", r)) for r in rec_order},
        "care_contrast": {
            r: _summ(boot_care[r],
                     (pt_rec("pos", r) - pt_rec("neg", r)) if pt_rec("pos", r) is not None and pt_rec("neg", r) is not None else None)
            for r in rec_order
        },
        "recipient_by_dimension": {
            d: {
                "pos": {r: _summ(boot_dim[d]["pos"][r],
                                 float(point["per_dim"][d]["pos"][rec_idx[r]]) if point["per_dim"][d]["pos"] is not None else None)
                        for r in rec_order},
                "neg": {r: _summ(boot_dim[d]["neg"][r],
                                 float(point["per_dim"][d]["neg"][rec_idx[r]]) if point["per_dim"][d]["neg"] is not None else None)
                        for r in rec_order},
            } for d in dims
        },
        "self_vs_other": {
            "by_stem": {s: _summ(boot_so[s], float(point["self_other"][s])) for s in stem_ids},
            "mean": _summ(boot_so_mean, float(np.mean(list(point["self_other"].values())))),
        },
        "items": [
            {
                "item_id": it.item_id, "recipient": it.recipient_key, "stem_id": it.stem_id,
                "dimension": it.dimension, "valence": it.valence,
                "theta": float(theta_point[idx[it.item_id]]),
                "theta_lo": float(np.percentile(boot_theta[:, idx[it.item_id]], 2.5)),
                "theta_hi": float(np.percentile(boot_theta[:, idx[it.item_id]], 97.5)),
            }
            for it in items
        ],
    }
    # drop the heavy per-replicate samples from per-item/dimension to keep file sane,
    # but keep them for the headline recipient effects + care contrast (plots want them).
    for d in dims:
        for v in ("pos", "neg"):
            for r in rec_order:
                result["recipient_by_dimension"][d][v][r].pop("samples", None)
    for s in stem_ids:
        result["self_vs_other"]["by_stem"][s].pop("samples", None)

    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"\nRecipient effect vs {ref_recipient} (overall θ, bootstrap 95% CI):")
    for r in rec_order:
        s = result["recipient_overall"][r]
        if s["point"] is not None:
            print(f"  {r:18} {s['point']:+.3f}  [{s['lo']:+.3f}, {s['hi']:+.3f}]")
    print(f"-> {output_path}")
    return result


@dataclass
class Args:
    comparisons_path: Path = DEFAULT_COMPARISONS
    output_path: Path = DEFAULT_OUTPUT
    n_boot: int = 500
    reg: float = 1.0
    ref_recipient: str = "human"
    seed: int = 0


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    bootstrap(args.comparisons_path, args.output_path, args.n_boot, args.reg, args.ref_recipient, args.seed)


if __name__ == "__main__":
    main()
