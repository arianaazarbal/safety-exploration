"""Build a dense transitivity-probe clique manifest (complete graph over ~18
stratified items, distinct stems so no same-stem pairs)."""

import itertools
import json
import random
from pathlib import Path

from bank import load_config, load_items

DIR = Path(__file__).parent


def main(n_items: int = 18, seed: int = 42):
    items = load_items(load_config())
    rng = random.Random(seed)
    by_dim = {}
    for it in items:
        by_dim.setdefault(it.dimension, []).append(it)
    recs = ["you", "claude_opus_47", "human", "chatgpt_55", "person", "claude_sonnet_46"]
    chosen, ri = [], 0
    for dim, lst in by_dim.items():
        for val in ("pos", "neg"):
            sub = [x for x in lst if x.valence == val]
            rng.shuffle(sub)
            picked = 0
            for x in sub:
                if any(c.stem_id == x.stem_id for c in chosen):
                    continue
                iid = f"{x.stem_id}__{recs[ri % len(recs)]}"
                ri += 1
                chosen.append(next(c for c in items if c.item_id == iid))
                picked += 1
                if picked >= 2:
                    break
    chosen = chosen[:n_items]
    ids = [c.item_id for c in chosen]
    assert len(set(c.stem_id for c in chosen)) == len(chosen), "duplicate stems"
    pairs = [{"pair_id": i, "item_a": a, "item_b": b}
             for i, (a, b) in enumerate(itertools.combinations(ids, 2))]
    out = DIR / "results" / "pairs_clique.json"
    out.write_text(json.dumps({"pairs": pairs, "n_pairs": len(pairs), "items": ids}, indent=2))
    print(f"clique: {len(ids)} items, {len(pairs)} pairs, {len(pairs) * 6} samples -> {out}")


if __name__ == "__main__":
    main()
