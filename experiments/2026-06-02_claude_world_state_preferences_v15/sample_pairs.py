"""Sample distinct pairs of items for pairwise comparison (v15).

Connectivity is guaranteed by construction: a random spanning tree (never joining
two items that share a stem) makes the comparison graph one connected component;
random edges are then added until every item reaches `degree_floor` (default 6 for
v15, up from v1's 4 — tightens precision on arbitrary i<->j contrasts). Same-stem
pairs are excluded throughout (transparent self/other swaps -> evaluation-aware).

Emits a manifest: `{seed, degree_floor, n_items, pairs: [{pair_id, item_a, item_b}, ...]}`.
At runtime each pair is shown in both A/B orders, so manifest order is canonical only.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import ArgumentParser

from bank import Item, load_config, load_items

DIR = Path(__file__).parent
DEFAULT_MANIFEST = DIR / "results" / "pairs.json"


def _spanning_tree_edges(items: list[Item], rng: random.Random) -> list[tuple[int, int]]:
    """Random spanning tree over item indices, never joining two same-stem items."""
    n = len(items)
    order = list(range(n))
    rng.shuffle(order)
    seen_stems: set[str] = set()
    front, rest = [], []
    for idx in order:
        if items[idx].stem_id not in seen_stems:
            seen_stems.add(items[idx].stem_id)
            front.append(idx)
        else:
            rest.append(idx)
    order = front + rest

    connected = [order[0]]
    edges: list[tuple[int, int]] = []
    for idx in order[1:]:
        partners = [c for c in connected if items[c].stem_id != items[idx].stem_id]
        partner = rng.choice(partners)
        edges.append(tuple(sorted((idx, partner))))
        connected.append(idx)
    return edges


def _add_to_degree_floor(
    items: list[Item],
    edges: set[tuple[int, int]],
    degree_floor: int,
    rng: random.Random,
) -> None:
    """Add random different-stem edges until every item has degree >= degree_floor."""
    n = len(items)
    deg = [0] * n
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    deficient = [i for i in range(n) if deg[i] < degree_floor]
    rng.shuffle(deficient)
    while deficient:
        u = deficient[0]
        if deg[u] >= degree_floor:
            deficient.pop(0)
            continue
        candidates = [
            v
            for v in range(n)
            if v != u
            and items[v].stem_id != items[u].stem_id
            and tuple(sorted((u, v))) not in edges
        ]
        if not candidates:
            deficient.pop(0)
            continue
        under = [v for v in candidates if deg[v] < degree_floor]
        v = rng.choice(under) if under else rng.choice(candidates)
        e = tuple(sorted((u, v)))
        edges.add(e)
        deg[u] += 1
        deg[v] += 1
        if deg[u] >= degree_floor:
            deficient.pop(0)


def is_connected(n: int, edges: set[tuple[int, int]]) -> bool:
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


def sample_pairs(
    seed: int = 0,
    degree_floor: int = 6,
    max_items: int | None = None,
    max_pairs: int | None = None,
    config: dict | None = None,
    category: str | None = None,
) -> dict:
    config = config or load_config()
    items = load_items(config)
    if category:
        items = [it for it in items if it.dimension == category]
        if not items:
            raise ValueError(f"no items found for category={category}")
    rng = random.Random(seed)
    if max_items is not None and max_items < len(items):
        idxs = list(range(len(items)))
        rng.shuffle(idxs)
        items = [items[i] for i in sorted(idxs[:max_items])]

    edges = set(_spanning_tree_edges(items, rng))
    _add_to_degree_floor(items, edges, degree_floor, rng)
    assert is_connected(len(items), edges), "graph not connected after construction"

    edge_list = sorted(edges)
    rng.shuffle(edge_list)
    if max_pairs is not None:
        if max_pairs < len(items) - 1:
            print(f"[warn] max_pairs={max_pairs} < n-1={len(items) - 1}; "
                  f"graph would disconnect. Keeping {len(edge_list)} edges.")
        else:
            edge_list = edge_list[:max_pairs]

    pairs = [
        {"pair_id": i, "item_a": items[a].item_id, "item_b": items[b].item_id}
        for i, (a, b) in enumerate(edge_list)
    ]
    degs = [0] * len(items)
    for a, b in edge_list:
        degs[a] += 1
        degs[b] += 1
    manifest = {
        "seed": seed,
        "degree_floor": degree_floor,
        "category": category,
        "n_items": len(items),
        "n_pairs": len(pairs),
        "min_degree": min(degs),
        "max_degree": max(degs),
        "connected": is_connected(len(items), set(edge_list)),
        "pairs": pairs,
    }
    return manifest


@dataclass
class Args:
    seed: int = 0
    degree_floor: int = 6
    max_items: int | None = None
    max_pairs: int | None = None
    output_path: Path = DEFAULT_MANIFEST
    category: str = ""  # "" = all categories; else filter to one dimension


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    cat = args.category or None
    out = args.output_path
    if cat and out == DEFAULT_MANIFEST:
        out = DIR / "results" / f"pairs_{cat}_seed{args.seed}.json"
    elif args.seed != 0 and out == DEFAULT_MANIFEST:
        out = DIR / "results" / f"pairs_seed{args.seed}.json"
    manifest = sample_pairs(
        seed=args.seed,
        degree_floor=args.degree_floor,
        max_items=args.max_items,
        max_pairs=args.max_pairs,
        category=cat,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(
        f"category={cat or 'all'} seed={args.seed} "
        f"n_items={manifest['n_items']} n_pairs={manifest['n_pairs']} "
        f"min_deg={manifest['min_degree']} max_deg={manifest['max_degree']} "
        f"connected={manifest['connected']} -> {out}"
    )


if __name__ == "__main__":
    main()
