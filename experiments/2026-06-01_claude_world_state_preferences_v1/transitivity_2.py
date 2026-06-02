"""Transitivity-violation analysis on the pairwise comparisons.

A Bradley-Terry / scalar-utility model assumes a single value ordering; preference
cycles (i≻j≻k≻i) are structure it cannot represent. We measure the cycle rate over
every triple whose three pairs were all directly compared. See ANALYSIS_PLAN.md.

Empirical preference uses the rep-averaged win rate P̂(a≻b) = wins(a over b) / N(a,b)
(both A/B orders pooled). Reported under three "is this edge a strict preference?"
rules: raw (P̂>0.5), margin (|P̂−0.5|≥m), significance (binomial CI excludes 0.5).
A weak-stochastic-transitivity violation == a cycle; strong stochastic transitivity
(SST) is reported separately on the probabilities. Runs on existing data only.
"""

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from simple_parsing import ArgumentParser

from bank2 import load_config, load_items

DIR = Path(__file__).parent
DEFAULT_COMPARISONS = DIR / "results" / "comparisons.json"
DEFAULT_OUTPUT = DIR / "results" / "transitivity.json"


def _key(a, b):
    return (a, b) if a < b else (b, a)


def build_winrates(rows: list[dict]) -> tuple[dict, dict]:
    """Return wins[(a,b)] = times a beat b (directed) and totals[(i,j)] (undirected)."""
    wins: dict[tuple[str, str], int] = {}
    totals: dict[tuple[str, str], int] = {}
    for r in rows:
        if r["choice"] is None:
            continue
        w, l = r["winner_item"], r["loser_item"]
        wins[(w, l)] = wins.get((w, l), 0) + 1
        totals[_key(w, l)] = totals.get(_key(w, l), 0) + 1
    return wins, totals


def prob(a, b, wins, totals) -> float:
    n = totals[_key(a, b)]
    return wins.get((a, b), 0) / n


def _decided(p: float, n: int, rule: str, margin: float) -> bool:
    if rule == "raw":
        return p != 0.5
    if rule == "margin":
        return abs(p - 0.5) >= margin
    if rule == "significance":
        if n == 0:
            return False
        se = math.sqrt(max(p * (1 - p), 1e-9) / n)
        return abs(p - 0.5) > 1.96 * se
    raise ValueError(rule)


def pairwise_consistency(wins: dict, totals: dict, margin: float, rank: dict | None = None) -> dict:
    """Single-number transitivity summary over ALL observed pairs.

    Rank items by a global score, then count head-to-head pairs whose majority winner
    disagrees with that ranking (an "upset"). Uses the full dataset, not just triangles.
    `rank` should be the BT θ (opponent-adjusted) when available; without it we fall back
    to overall win rate, which is confounded by comparison schedule in a non-clique graph.
    Reported overall and on decisive pairs (|P̂−0.5|≥margin)."""
    items = sorted({a for a, _ in totals} | {b for _, b in totals})
    seen = {_key(i, j) for (i, j) in totals}
    if rank is not None:
        score = {it: rank.get(it, 0.0) for it in items}
        ranked_by = "bt_theta"
    else:
        deg_wins = {it: 0.0 for it in items}
        deg_n = {it: 0 for it in items}
        for (a, b) in seen:
            n = totals[(a, b)]
            wa = wins.get((a, b), 0)
            deg_wins[a] += wa
            deg_wins[b] += n - wa
            deg_n[a] += n
            deg_n[b] += n
        score = {it: (deg_wins[it] / deg_n[it] if deg_n[it] else 0.5) for it in items}
        ranked_by = "empirical_winrate"

    rate = score
    total = consistent = dec = dec_consistent = 0
    for k in seen:
        a, b = k
        p = wins.get((a, b), 0) / totals[k]
        if p == 0.5:
            continue
        emp = a if p > 0.5 else b
        ranked = a if rate[a] > rate[b] else b
        total += 1
        if emp == ranked:
            consistent += 1
        if abs(p - 0.5) >= margin:
            dec += 1
            if emp == ranked:
                dec_consistent += 1
    return {
        "ranked_by": ranked_by,
        "n_pairs": total,
        "consistency_rate": consistent / total if total else None,
        "upset_rate": 1 - consistent / total if total else None,
        "decisive_n": dec,
        "decisive_upset_rate": 1 - dec_consistent / dec if dec else None,
    }


def analyze(
    comparisons_path: Path = DEFAULT_COMPARISONS,
    output_path: Path = DEFAULT_OUTPUT,
    margin: float = 0.25,
    fit_path: Path | None = None,
    config: dict | None = None,
) -> dict:
    config = config or load_config()
    meta = {it.item_id: it for it in load_items(config)}
    rows = json.loads(Path(comparisons_path).read_text())
    wins, totals = build_winrates(rows)

    theta = None
    if fit_path and Path(fit_path).exists():
        theta = {it["item_id"]: it["theta"] for it in json.loads(Path(fit_path).read_text())["items"]}

    # adjacency over the observed comparison graph
    adj: dict[str, set[str]] = {}
    for (i, j) in totals:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)

    # enumerate triangles (all three pairs observed)
    triangles: list[tuple[str, str, str]] = []
    for v in adj:
        nbrs = sorted(adj[v])
        for a, b in combinations(nbrs, 2):
            if b in adj.get(a, ()):  # a-b also an edge -> triangle
                tri = tuple(sorted((v, a, b)))
                triangles.append(tri)
    triangles = sorted(set(triangles))

    rules = ["raw", "margin", "significance"]
    results = {r: {"decided": 0, "cycles": 0} for r in rules}
    sst = {"checked": 0, "violations": 0}
    cycle_examples: list[dict] = []
    cycle_cross = {"cross_recipient": 0, "cross_dimension": 0, "within": 0}

    for tri in triangles:
        i, j, k = tri
        edges = [(i, j), (j, k), (i, k)]
        probs = {e: prob(e[0], e[1], wins, totals) for e in edges}
        ns = {e: totals[_key(*e)] for e in edges}

        for rule in rules:
            if not all(_decided(probs[e], ns[e], rule, margin) for e in edges):
                continue
            results[rule]["decided"] += 1
            # wins per node within the triangle
            wcount = {i: 0, j: 0, k: 0}
            for (a, b) in edges:
                winner = a if probs[(a, b)] > 0.5 else b
                wcount[winner] += 1
            is_cycle = sorted(wcount.values()) == [1, 1, 1]
            if is_cycle:
                results[rule]["cycles"] += 1
                if rule == "raw":
                    recs = {meta[x].recipient_key for x in tri}
                    dims = {meta[x].dimension for x in tri}
                    if len(recs) > 1:
                        cycle_cross["cross_recipient"] += 1
                    if len(dims) > 1:
                        cycle_cross["cross_dimension"] += 1
                    if len(recs) == 1 and len(dims) == 1:
                        cycle_cross["within"] += 1
                    if len(cycle_examples) < 25:
                        cycle_examples.append(
                            {"items": list(tri), "probs": {f"{a}>{b}": round(probs[(a, b)], 3) for (a, b) in edges}}
                        )

        # SST on raw-decided transitive triangles
        if all(_decided(probs[e], ns[e], "raw", margin) for e in edges):
            wcount = {i: 0, j: 0, k: 0}
            for (a, b) in edges:
                winner = a if probs[(a, b)] > 0.5 else b
                wcount[winner] += 1
            if sorted(wcount.values()) == [0, 1, 2]:
                top = max(wcount, key=wcount.get)
                bot = min(wcount, key=wcount.get)
                mid = [x for x in tri if x not in (top, bot)][0]
                p_top_mid = prob(top, mid, wins, totals)
                p_mid_bot = prob(mid, bot, wins, totals)
                p_top_bot = prob(top, bot, wins, totals)
                sst["checked"] += 1
                if p_top_bot < max(p_top_mid, p_mid_bot) - 1e-9:
                    sst["violations"] += 1

    for r in rules:
        d = results[r]["decided"]
        results[r]["rate"] = results[r]["cycles"] / d if d else None
    sst["rate"] = sst["violations"] / sst["checked"] if sst["checked"] else None
    pairwise = pairwise_consistency(wins, totals, margin, rank=theta)

    out = {
        "n_triangles_observed": len(triangles),
        "margin": margin,
        "cycle_rate": results,
        "sst_violation": sst,
        "pairwise_consistency": pairwise,
        "cycle_breakdown_raw": cycle_cross,
        "cycle_examples_raw": cycle_examples,
    }
    Path(output_path).write_text(json.dumps(out, indent=2))
    print(f"Triangles with all 3 pairs observed: {len(triangles)}")
    for r in rules:
        d = results[r]
        rate = f"{100 * d['rate']:.1f}%" if d["rate"] is not None else "n/a"
        print(f"  [{r:12}] cycles {d['cycles']}/{d['decided']} = {rate}")
    print(f"  SST violations: {sst['violations']}/{sst['checked']} "
          f"({100 * sst['rate']:.1f}%)" if sst["rate"] is not None else "  SST: n/a")
    print(f"  raw cycles — cross-recipient {cycle_cross['cross_recipient']}, "
          f"cross-dimension {cycle_cross['cross_dimension']}, within {cycle_cross['within']}")
    if pairwise["consistency_rate"] is not None:
        print(f"  pairwise consistency (all data, ranked by {pairwise['ranked_by']}): "
              f"{100 * pairwise['consistency_rate']:.1f}% "
              f"(upset rate {100 * pairwise['upset_rate']:.1f}%, "
              f"decisive {100 * pairwise['decisive_upset_rate']:.1f}%) over {pairwise['n_pairs']} pairs")
    print(f"-> {output_path}")
    return out


@dataclass
class Args:
    comparisons_path: Path = DEFAULT_COMPARISONS
    output_path: Path = DEFAULT_OUTPUT
    margin: float = 0.25
    fit_path: Path | None = None


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    analyze(args.comparisons_path, args.output_path, args.margin, fit_path=args.fit_path)


if __name__ == "__main__":
    main()
