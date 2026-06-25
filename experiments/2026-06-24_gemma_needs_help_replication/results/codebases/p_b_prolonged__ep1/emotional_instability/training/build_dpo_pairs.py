"""Build the 280 DPO preference pairs (Section 4.1 / Appendix H).

Pairs a frustrated response (score >= 3) with a calm response (score 0/1) to
the same puzzle at the same turn count. The shared prompt is a canonical plain
context: the puzzle followed by (turn - 1) neutral rejections, chat-templated
for Gemma. ``chosen`` = calm, ``rejected`` = frustrated.

Output: ``data/dpo_pairs.jsonl`` with {prompt, chosen, rejected} (TRL DPO
schema).
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

import config
from .. import prompts
from ..models.base import Message
from ..utils.io import read_jsonl, write_jsonl


def _canonical_context(puzzle: str, turn: int, rng: random.Random) -> list[Message]:
    """Puzzle + (turn-1) neutral rejections as a plain message list."""
    msgs: list[Message] = [{"role": "user", "content": puzzle}]
    # We only need the user-side context up to the assistant turn being trained;
    # intervening assistant turns are omitted (the pair concerns the turn-`turn`
    # response). For >1 turn we approximate prior assistant turns as empty and
    # append rejections, matching the paper's "same question, matching turns".
    for _ in range(turn - 1):
        msgs.append({"role": "assistant", "content": "[prior attempt]"})
        msgs.append({"role": "user", "content": rng.choice(prompts.NEUTRAL_REJECTIONS)})
    return msgs


def build_pairs(n_pairs: int = config.DPO_CFG.n_pairs, seed: int = config.SEED):
    calm = read_jsonl(config.DATA_DIR / "calm_pool_diverse.jsonl")
    frustrated = read_jsonl(config.DATA_DIR / "frustrated_pool.jsonl")
    if not calm or not frustrated:
        raise SystemExit("[build_dpo_pairs] need calm + frustrated pools first "
                         "(training.gen_calm_data)")

    # index calm responses by (puzzle_id, turn)
    calm_by_key = defaultdict(list)
    for c in calm:
        calm_by_key[(c["puzzle_id"], c["turn"])].append(c)

    rng = random.Random(seed)
    pairs = []
    for fr in frustrated:
        key = (fr["puzzle_id"], fr["turn"])
        # match same puzzle+turn; fall back to same turn count if needed
        candidates = calm_by_key.get(key) or [
            c for c in calm if c["turn"] == fr["turn"]
        ]
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        context = _canonical_context(fr["puzzle"], fr["turn"], rng)
        pairs.append(dict(
            prompt=context,                 # list of messages (TRL applies template)
            chosen=chosen["response"],
            rejected=fr["response"],
            meta=dict(puzzle_id=fr["puzzle_id"], turn=fr["turn"],
                      rejected_score=fr["score"], chosen_score=chosen["score"]),
        ))
        if len(pairs) >= n_pairs:
            break

    out = config.DATA_DIR / "dpo_pairs.jsonl"
    write_jsonl(out, pairs)
    print(f"[build_dpo_pairs] wrote {len(pairs)} pairs -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=config.DPO_CFG.n_pairs)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    build_pairs(args.n_pairs, args.seed)
