"""Build the pairwise comparison manifest and split it into train / held-out.

Modes:
  - ``cross``          : bipartite complete graph, every welfare item vs every
                         inter-AI-value item (the v1 design; 19*16 pairs).
  - ``all``            : complete graph over BOTH pools (extends to one ranking).
  - ``within_welfare`` : complete graph over welfare items only.
  - ``within_value``   : complete graph over inter-AI-value items only.

Held-out split: a fraction of pairs are reserved to *test* the fitted BT model
(predict P(a>b) on pairs never used to fit theta). The split protects a spanning
tree of the comparison graph in the train set, so train stays connected and BT is
identifiable no matter the seed. Held-out pairs are still RUN at inference time;
they are excluded only from fitting (see fit_bt.py / validate_bt.py).

Manifest pairs carry a ``split`` field in {"train","heldout"}; A/B order is
randomized at run time, so manifest order is canonical only.
"""

import json
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from simple_parsing import ArgumentParser

from items import load_items

DIR = Path(__file__).parent
DEFAULT_MANIFEST = DIR / "results" / "pairs.json"


def _edges_for_mode(sources: list[str], mode: str) -> list[tuple[int, int]]:
    """Index pairs (i<j) included under `mode`, given each item's source."""
    n = len(sources)
    if mode == "all":
        return list(combinations(range(n), 2))
    if mode == "cross":
        return [(i, j) for i, j in combinations(range(n), 2) if sources[i] != sources[j]]
    if mode in ("within_welfare", "within_value"):
        return list(combinations(range(n), 2))
    raise ValueError(f"unknown mode {mode!r}")


def _spanning_tree(n: int, edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """BFS spanning tree (subset of `edges`); asserts the graph is connected."""
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen = {0}
    queue = [0]
    tree: list[tuple[int, int]] = []
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                tree.append((min(u, v), max(u, v)))
                queue.append(v)
    assert len(seen) == n, f"graph not connected: reached {len(seen)}/{n} items"
    return tree


def build(
    mode: str = "cross",
    seed: int = 0,
    heldout_frac: float = 0.15,
) -> dict:
    all_items = load_items()
    if mode == "within_welfare":
        items = [it for it in all_items if it.source == "welfare"]
    elif mode == "within_value":
        items = [it for it in all_items if it.source == "inter_ai_value"]
    else:
        items = all_items

    sources = [it.source for it in items]
    edges = _edges_for_mode(sources, mode)
    tree = set(_spanning_tree(len(items), edges))

    rng = random.Random(seed)
    non_tree = [e for e in edges if e not in tree]
    rng.shuffle(non_tree)
    n_heldout = round(heldout_frac * len(edges))
    n_heldout = min(n_heldout, len(non_tree))  # never hold out a tree edge
    heldout = set(non_tree[:n_heldout])

    edge_list = sorted(edges)
    rng.shuffle(edge_list)
    pairs = [
        {
            "pair_id": k,
            "item_a": items[a].item_id,
            "item_b": items[b].item_id,
            "split": "heldout" if (a, b) in heldout else "train",
        }
        for k, (a, b) in enumerate(edge_list)
    ]
    n_train = sum(1 for p in pairs if p["split"] == "train")
    return {
        "mode": mode,
        "seed": seed,
        "heldout_frac": heldout_frac,
        "n_items": len(items),
        "item_ids": [it.item_id for it in items],
        "n_pairs": len(pairs),
        "n_train": n_train,
        "n_heldout": len(pairs) - n_train,
        "pairs": pairs,
    }


@dataclass
class Args:
    mode: str = "cross"
    seed: int = 0
    heldout_frac: float = 0.15
    output_path: Path = DEFAULT_MANIFEST


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    manifest = build(mode=args.mode, seed=args.seed, heldout_frac=args.heldout_frac)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"mode={manifest['mode']} n_items={manifest['n_items']} "
        f"n_pairs={manifest['n_pairs']} (train={manifest['n_train']}, "
        f"heldout={manifest['n_heldout']}) -> {args.output_path}"
    )


if __name__ == "__main__":
    main()
