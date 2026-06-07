"""Fit a Bradley-Terry model to the pairwise comparisons (Approach 1, v0).

Bag-of-outcomes: one free latent strength per item, no recipient/outcome
structure imposed. Strengths are fit by the standard minorization-maximization
(MM) update (Hunter 2004) with a mild Gamma prior for finite, identifiable
estimates. Utilities are ``theta = log(strength)``, centered to mean zero.

Standard errors come from the Laplace approximation: the inverse of the observed
Fisher information (a graph Laplacian), pseudo-inverted on the mean-zero subspace.

Then a post-hoc weighted least squares regresses fitted utilities on
``C(stem) + C(recipient)`` to read off the recipient effect (the core question:
does the same outcome get valued differently by recipient, self vs other?).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from simple_parsing import ArgumentParser

from bank import load_config, load_items

DIR = Path(__file__).parent
DEFAULT_COMPARISONS = DIR / "results" / "comparisons.json"
DEFAULT_OUTPUT = DIR / "results" / "bt_fit.json"


def _graph_connected(n: int, counts: dict[tuple[int, int], int]) -> bool:
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for (i, j) in counts:
        adj[i].add(j)
        adj[j].add(i)
    seen, stack = {0}, [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


def fit_bt_mm(
    n: int,
    wins: np.ndarray,
    counts: dict[tuple[int, int], int],
    reg: float = 1.0,
    max_iter: int = 10000,
    tol: float = 1e-9,
) -> np.ndarray:
    """MAP Bradley-Terry strengths via MM with Gamma(1+reg, reg) prior.

    counts maps an unordered pair (i<j) -> total comparisons between i and j.
    Returns strength vector p (positive), normalized to geometric mean 1.
    """
    p = np.ones(n)
    # neighbor lists for the denominator sum
    nbrs: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
    for (i, j), c in counts.items():
        nbrs[i].append((j, c))
        nbrs[j].append((i, c))
    for _ in range(max_iter):
        p_new = np.empty(n)
        for i in range(n):
            denom = reg
            for (j, c) in nbrs[i]:
                denom += c / (p[i] + p[j])
            p_new[i] = (reg + wins[i]) / denom
        p_new /= np.exp(np.mean(np.log(p_new)))  # geometric-mean normalize
        if np.max(np.abs(np.log(p_new) - np.log(p))) < tol:
            p = p_new
            break
        p = p_new
    return p


def laplace_se(n: int, p: np.ndarray, counts: dict[tuple[int, int], int], reg: float) -> np.ndarray:
    """Approx SE on theta=log p from the observed information of the log-posterior.

    The data Fisher information is the singular graph Laplacian
    ``w_ij = c_ij p_i p_j / (p_i+p_j)^2``. The Gamma(1+reg, reg) prior on p adds
    ``reg * p_i`` to each diagonal (the prior's curvature in theta-space), which
    makes the information matrix positive-definite and identifiable -- this is the
    same regularization that defines the MAP estimate, not an arbitrary ridge.
    """
    H = np.zeros((n, n))
    for (i, j), c in counts.items():
        w = c * p[i] * p[j] / (p[i] + p[j]) ** 2
        H[i, i] += w
        H[j, j] += w
        H[i, j] -= w
        H[j, i] -= w
    H[np.diag_indices(n)] += reg * p
    cov = np.linalg.inv(H)
    return np.sqrt(np.clip(np.diag(cov), 0, None))


def _wls_recipient_regression(
    theta: np.ndarray,
    se: np.ndarray,
    stems: list[str],
    recipients: list[str],
    recipient_order: list[str],
    ref_recipient: str,
) -> dict | None:
    """WLS of theta ~ C(stem) + C(recipient), weights 1/se^2. Returns recipient
    coefficients relative to ref_recipient, or None if rank-deficient."""
    uniq_stems = sorted(set(stems))
    rec_levels = [r for r in recipient_order if r != ref_recipient]
    if len(uniq_stems) < 2 or len(rec_levels) < 1:
        return None
    stem_idx = {s: k for k, s in enumerate(uniq_stems)}
    rec_idx = {r: k for k, r in enumerate(rec_levels)}
    rows = len(theta)
    p = 1 + (len(uniq_stems) - 1) + len(rec_levels)
    # Design: col 0 intercept; cols 1..len(uniq_stems)-1 stem dummies (first stem is
    # reference, dropped); remaining cols recipient dummies (ref_recipient dropped).
    X = np.zeros((rows, p))
    X[:, 0] = 1.0
    for r, (s, rec) in enumerate(zip(stems, recipients)):
        si = stem_idx[s]
        if si > 0:
            X[r, si] = 1.0
        if rec in rec_idx:
            X[r, len(uniq_stems) + rec_idx[rec]] = 1.0
    w = 1.0 / np.clip(se, 1e-6, None) ** 2
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = theta * sw
    rank = np.linalg.matrix_rank(Xw)
    if rank < p:
        return None
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    dof = rows - p
    sigma2 = (resid @ resid) / dof if dof > 0 else np.nan
    cov = sigma2 * np.linalg.inv(Xw.T @ Xw)
    se_beta = np.sqrt(np.diag(cov))
    coefs = {}
    for rec in rec_levels:
        ci = len(uniq_stems) + rec_idx[rec]
        coefs[rec] = {"coef": float(beta[ci]), "se": float(se_beta[ci])}
    coefs[ref_recipient] = {"coef": 0.0, "se": 0.0}
    return {"ref_recipient": ref_recipient, "coefficients": coefs, "dof": int(dof)}


def fit(
    comparisons_path: Path = DEFAULT_COMPARISONS,
    output_path: Path = DEFAULT_OUTPUT,
    reg: float = 1.0,
    config: dict | None = None,
) -> dict:
    config = config or load_config()
    items_meta = {it.item_id: it for it in load_items(config)}
    recipient_order = list(config["recipients"].keys())

    rows = json.loads(Path(comparisons_path).read_text())
    parsed = [r for r in rows if r["choice"] is not None]
    n_unparse = len(rows) - len(parsed)

    present = sorted({r["winner_item"] for r in parsed} | {r["loser_item"] for r in parsed})
    idx = {it: k for k, it in enumerate(present)}
    n = len(present)

    wins = np.zeros(n)
    counts: dict[tuple[int, int], int] = {}
    for r in parsed:
        wi, li = idx[r["winner_item"]], idx[r["loser_item"]]
        wins[wi] += 1
        key = (min(wi, li), max(wi, li))
        counts[key] = counts.get(key, 0) + 1

    connected = _graph_connected(n, counts)
    if not connected:
        print(f"[WARN] comparison graph NOT connected over {n} items; BT under-identified.")

    p = fit_bt_mm(n, wins, counts, reg=reg)
    theta = np.log(p)
    theta -= theta.mean()
    se = laplace_se(n, p, counts, reg)

    n_comparisons = np.zeros(n)
    for (i, j), c in counts.items():
        n_comparisons[i] += c
        n_comparisons[j] += c

    items_out = []
    stems, recs = [], []
    for it in present:
        meta = items_meta[it]
        items_out.append(
            {
                "item_id": it,
                "stem_id": meta.stem_id,
                "recipient": meta.recipient_key,
                "dimension": meta.dimension,
                "valence": meta.valence,
                "theta": float(theta[idx[it]]),
                "se": float(se[idx[it]]),
                "n_comparisons": int(n_comparisons[idx[it]]),
                "n_wins": int(wins[idx[it]]),
            }
        )
        stems.append(meta.stem_id)
        recs.append(meta.recipient_key)

    regression = _wls_recipient_regression(
        theta, se, stems, recs, recipient_order, ref_recipient=recipient_order[0]
    )

    result = {
        "n_items": n,
        "n_samples_used": len(parsed),
        "n_unparseable": n_unparse,
        "unparseable_rate": n_unparse / max(len(rows), 1),
        "connected": connected,
        "reg": reg,
        "items": items_out,
        "recipient_regression": regression,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(
        f"Fit BT over {n} items from {len(parsed)} samples "
        f"(unparseable {n_unparse}, {100 * result['unparseable_rate']:.1f}%). connected={connected}"
    )
    if regression:
        print(f"Recipient effects (ref={regression['ref_recipient']}, theta units):")
        for rec, c in regression["coefficients"].items():
            print(f"  {rec:18} {c['coef']:+.3f} ± {c['se']:.3f}")
    else:
        print("Recipient regression skipped (insufficient/rank-deficient data).")
    print(f"-> {output_path}")
    return result


@dataclass
class Args:
    comparisons_path: Path = DEFAULT_COMPARISONS
    output_path: Path = DEFAULT_OUTPUT
    reg: float = 1.0


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    fit(comparisons_path=args.comparisons_path, output_path=args.output_path, reg=args.reg)


if __name__ == "__main__":
    main()
