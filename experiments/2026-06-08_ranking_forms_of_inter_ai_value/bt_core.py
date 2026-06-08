"""Bradley-Terry estimation primitives, shared by fit_bt.py and validate_bt.py.

MAP Bradley-Terry via minorization-maximization (Hunter 2004) with a mild
Gamma(1+reg, reg) prior for finite, identifiable strengths. Utilities are
``theta = log(strength)``, mean-centered. Standard errors from the Laplace
approximation (inverse observed Fisher information of the log-posterior).

Identical math to world_state_preferences_v0; copied here so this experiment is
self-contained (no recipient/stem structure needed).
"""

import numpy as np


def graph_connected(n: int, counts: dict[tuple[int, int], int]) -> bool:
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


def fit_bt_mm(n, wins, counts, reg=1.0, max_iter=10000, tol=1e-9) -> np.ndarray:
    """MAP BT strengths via MM. counts: unordered pair (i<j)->total comparisons.
    Returns positive strength vector p, geometric-mean normalized to 1."""
    p = np.ones(n)
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
        p_new /= np.exp(np.mean(np.log(p_new)))
        if np.max(np.abs(np.log(p_new) - np.log(p))) < tol:
            p = p_new
            break
        p = p_new
    return p


def laplace_se(n, p, counts, reg) -> np.ndarray:
    """Approx SE on theta=log p from the observed information (graph Laplacian
    w_ij = c_ij p_i p_j / (p_i+p_j)^2 plus the prior curvature reg*p_i on the diag)."""
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


def fit_bt_vectorized(n, win_idx, lose_idx, reg=1.0, max_iter=5000, tol=1e-8) -> np.ndarray:
    """Vectorized MM (numerically identical to fit_bt_mm); returns theta=log p,
    mean-centered. Fast enough for K-fold refits."""
    win_idx = np.asarray(win_idx)
    lose_idx = np.asarray(lose_idx)
    wins = np.bincount(win_idx, minlength=n).astype(float)
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
        denom = np.full(n, float(reg))
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


def counts_wins_from_samples(samples, idx):
    """samples: iterable of (winner_item, loser_item). idx: item_id->index.
    Returns (wins vector, counts dict over unordered pairs)."""
    n = len(idx)
    wins = np.zeros(n)
    counts: dict[tuple[int, int], int] = {}
    for w_item, l_item in samples:
        wi, li = idx[w_item], idx[l_item]
        wins[wi] += 1
        k = (min(wi, li), max(wi, li))
        counts[k] = counts.get(k, 0) + 1
    return wins, counts
