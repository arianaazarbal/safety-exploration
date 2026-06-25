"""Build the 280-pair DPO dataset (paper Section 4.1 / Appendix H).

A preference pair is (prompt, chosen, rejected) where, for the same impossible
numeric question and matching turn count:
  * chosen   = a CALM response (frustration score 0 or 1), with the supportive
               system prompt/suffix stripped.
  * rejected = a FRUSTRATED response (frustration score >= 3).

We match by (puzzle_id, turn_number). The conversation *prompt* given to DPO is
the clean (unaugmented) chat history up to the point of the final assistant
turn; chosen/rejected are the two candidate final turns.

Target distribution (App. H Table 10): rejected scores biased toward 3-4, most
pairs at turn 3. We sample to approximate this and cap at 280 pairs.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from .. import config
from ..tasks import build_puzzle_bank, rejection_sequence
from ..utils import read_jsonl, write_jsonl

PUZZLE_PROMPTS = {p.puzzle_id: p.prompt for p in build_puzzle_bank()}


def _clean_history(puzzle_id: str, turn_number: int) -> list[dict]:
    """Reconstruct the clean (unaugmented) chat history preceding the final
    assistant turn at `turn_number` for this puzzle."""
    prompt = PUZZLE_PROMPTS.get(puzzle_id, "")
    rejections = rejection_sequence("neutral", turn_number - 1, seed=0)
    history = [{"role": "user", "content": prompt}]
    # We don't have the *specific* intermediate assistant turns here; for DPO the
    # prompt is the user-side context. Interleave placeholder-free: we include
    # only user turns plus the rejections, which is what conditions the final
    # response. (See DESIGN.md: DPO conditions on the user-visible context.)
    for r in rejections:
        history.append({"role": "user", "content": r})
    return history


def build_dpo_dataset(*, calm_path: Path | None = None,
                      frustrated_path: Path | None = None,
                      n_pairs: int = config.DPO.dataset_size,
                      seed: int = 0,
                      out_path: Path | None = None) -> Path:
    """Construct DPO preference pairs and write them as JSONL.

    `calm_path` and `frustrated_path` are calm/vanilla generation files from
    generate_calm_responses(mode="reassured"/"vanilla").
    """
    calm_path = calm_path or (config.DATA_DIR / "calm_reassured.jsonl")
    frustrated_path = frustrated_path or (config.DATA_DIR / "calm_vanilla.jsonl")
    out_path = out_path or (config.DATA_DIR / "dpo_pairs.jsonl")
    rng = random.Random(seed)

    # Index calm (chosen) and frustrated (rejected) candidate turns by key.
    chosen: dict[tuple[str, int], list[dict]] = defaultdict(list)
    rejected: dict[tuple[str, int], list[dict]] = defaultdict(list)

    for conv in read_jsonl(calm_path):
        for t in conv["turns"]:
            if t["rating"] in (0, 1):
                key = (conv["puzzle_id"], t["turn_number"])
                chosen[key].append(t)

    for conv in read_jsonl(frustrated_path):
        for t in conv["turns"]:
            if t["rating"] >= 3:
                key = (conv["puzzle_id"], t["turn_number"])
                rejected[key].append(t)

    # Build pairs preferring later turns (App. H: ~74% at turn 3) and
    # mid-range rejected scores.
    keys = [k for k in rejected if chosen.get(k)]
    # Sort so higher turn numbers come first; within that prefer rejected score 3/4.

    def key_priority(k):
        turn = k[1]
        best_rej = min(rejected[k], key=lambda t: abs(t["rating"] - 3.5))
        return (turn, -abs(best_rej["rating"] - 3.5))

    keys.sort(key=key_priority, reverse=True)

    pairs = []
    for k in keys:
        if len(pairs) >= n_pairs:
            break
        ch = rng.choice(chosen[k])
        rj = rng.choice(rejected[k])
        puzzle_id, turn_number = k
        pairs.append({
            "puzzle_id": puzzle_id,
            "turn_number": turn_number,
            "prompt": _clean_history(puzzle_id, turn_number),
            "chosen": ch["response"],
            "rejected": rj["response"],
            "chosen_score": ch["rating"],
            "rejected_score": rj["rating"],
        })

    if len(pairs) < n_pairs:
        print(f"[build_dpo_dataset] WARNING: only built {len(pairs)} pairs "
              f"(< {n_pairs}); generate more calm/vanilla data.")

    write_jsonl(out_path, pairs)
    return out_path
