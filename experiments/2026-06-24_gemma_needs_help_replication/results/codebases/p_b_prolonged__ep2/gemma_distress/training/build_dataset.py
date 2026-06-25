"""Assemble SFT and DPO training datasets (Section 4.1, Appendix E/H).

DPO: 280 preference pairs (chosen calm score 0-1 / rejected frustrated score
>=3) to the same questions at matching turn counts. We bias the sample toward
the Table-10 distribution (mostly turn 3, rejected scores concentrated at 3-4).

SFT: 650 calm responses (conversations all scoring 0-1, 1-3 turns) mixed with
500 standard-instruct samples from Dolci-Instruct-SFT.

Outputs are written as JSONL in the chat/preference formats `trl` consumes:
  * SFT  -> {"messages": [...]}
  * DPO  -> {"prompt": [...], "chosen": [...], "rejected": [...]}
"""
from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Optional

from ..config import RunConfig
from ..utils.io import ensure_dir, write_jsonl
from .dolci import load_dolci_sft
from .generate_calm_data import CalmConversation, PrefPair

DPO_TARGET_PAIRS = 280
SFT_CALM_RESPONSES = 650
SFT_DOLCI_SAMPLES = 500

# Table 10 target distributions (proportions), used to bias sampling so the
# assembled 280 pairs resemble the paper's dataset statistics.
REJECTED_SCORE_TARGET = {3: 0.661, 4: 0.221, 5: 0.057, 6: 0.032, 7: 0.029}
TURN_TARGET = {1: 0.011, 2: 0.246, 3: 0.743}


def build_dpo_dataset(pairs: list[PrefPair], cfg: RunConfig,
                      target: int = DPO_TARGET_PAIRS, seed: int = 0) -> str:
    """Sample `target` preference pairs biased toward Table-10 statistics and
    write them in trl preference format."""
    rng = random.Random(seed)
    out_dir = ensure_dir(os.path.join(cfg.output_dir, "section4", "datasets"))

    # Bucket by (turn, rejected_score) and sample to approximate the targets.
    buckets: dict[tuple[int, int], list[PrefPair]] = defaultdict(list)
    for p in pairs:
        rs = min(p.rejected_score, 7)
        buckets[(p.turn, rs)].append(p)

    chosen_pairs: list[PrefPair] = []
    # Greedy proportional fill across the joint (turn, score) target.
    joint_targets: dict[tuple[int, int], float] = {}
    for turn, tp in TURN_TARGET.items():
        for score, sp in REJECTED_SCORE_TARGET.items():
            joint_targets[(turn, score)] = tp * sp
    norm = sum(joint_targets.values()) or 1.0

    for key, frac in joint_targets.items():
        want = round(target * frac / norm)
        avail = buckets.get(key, [])
        rng.shuffle(avail)
        chosen_pairs.extend(avail[:want])

    # Top up / trim to exactly `target` from whatever remains.
    if len(chosen_pairs) < target:
        remaining = [p for p in pairs if p not in chosen_pairs]
        rng.shuffle(remaining)
        chosen_pairs.extend(remaining[:target - len(chosen_pairs)])
    chosen_pairs = chosen_pairs[:target]

    rows = []
    for p in chosen_pairs:
        rows.append({
            "prompt": p.prompt_messages,
            "chosen": [{"role": "assistant", "content": p.chosen}],
            "rejected": [{"role": "assistant", "content": p.rejected}],
            "meta": {"turn": p.turn, "chosen_score": p.chosen_score,
                     "rejected_score": p.rejected_score,
                     "puzzle_kind": p.puzzle_kind},
        })
    path = os.path.join(out_dir, "dpo_pairs.jsonl")
    write_jsonl(path, rows)
    return path


def build_sft_dataset(calm_convs: list[CalmConversation], cfg: RunConfig,
                      n_calm: int = SFT_CALM_RESPONSES,
                      n_dolci: int = SFT_DOLCI_SAMPLES, seed: int = 0) -> str:
    """Build the 'diverse' SFT set: calm conversations (all turns 0-1) plus
    standard-instruct samples, in trl chat format."""
    rng = random.Random(seed)
    out_dir = ensure_dir(os.path.join(cfg.output_dir, "section4", "datasets"))

    # Keep only fully-calm conversations; expand into message lists.
    calm = [c for c in calm_convs if c.all_calm()]
    rng.shuffle(calm)

    calm_rows = []
    for c in calm:
        msgs = []
        for u, a in zip(c.clean_user_turns, c.assistant_turns):
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        calm_rows.append({"messages": msgs})
        # "650 calm responses" -> count assistant turns toward the budget.
        if sum(len(r["messages"]) // 2 for r in calm_rows) >= n_calm:
            break

    dolci = load_dolci_sft(n=n_dolci, seed=seed)
    dolci_rows = [{"messages": conv} for conv in dolci]

    rows = calm_rows + dolci_rows
    rng.shuffle(rows)
    path = os.path.join(out_dir, "sft_diverse.jsonl")
    write_jsonl(path, rows)
    return path
