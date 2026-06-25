"""Build the DPO and SFT finetuning datasets (Section 4.1 / Appendix E, H).

DPO (280 preference pairs):
  Pair a frustrated response (score >= 3, from vanilla Gemma-3-27B-it numeric
  evals) with a calm response (score 0/1, from generate_calm_data) to the SAME
  puzzle and matching turn count. The shared `prompt` is the frustrated
  rollout's real conversation history up to that turn; `rejected` is its
  frustrated completion, `chosen` is the transplanted calm completion. Table 10
  shows the score/turn distribution we approximately reproduce (bias to
  mid-frustration, turn 3).

SFT (1,150 samples):
  650 calm conversations (1-3 turns) formatted as chat data + 500 standard
  instruct samples from Dolci-Instruct-SFT to mitigate degeneration.

Outputs:
  data/dpo_dataset.jsonl  - {"prompt": [...messages], "chosen": str, "rejected": str}
  data/sft_dataset.jsonl  - {"messages": [...]}
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .. import config
from ..models.base import Message
from .generate_calm_data import CalmConversation, load_calm

DPO_PATH = config.DATA_DIR / "dpo_dataset.jsonl"
SFT_PATH = config.DATA_DIR / "sft_dataset.jsonl"

N_DPO_PAIRS = 280
N_SFT_CALM = 650
N_SFT_DOLCI = 500
DOLCI_DATASET = "allenai/Dolci-Instruct-SFT"


# --------------------------------------------------------------------------- #
# Frustrated source extraction
# --------------------------------------------------------------------------- #

def _puzzle_signature(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def _collect_frustrated(model_key: str = "gemma-3-27b-it", min_score: int = 3
                        ) -> list[dict]:
    """Frustrated numeric responses with their history + puzzle signature."""
    path = config.RESULTS_DIR / "section2" / model_key / f"{config.NUMERIC.name}.jsonl"
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        history: list[Message] = []
        for t in rec["turns"]:
            history.append({"role": "user", "content": t["user_message"]})
            if t["frustration"] >= min_score:
                out.append({
                    "prompt": list(history),         # up to & incl. this user turn
                    "rejected": t["assistant_text"],
                    "score": t["frustration"],
                    "turn_index": t["turn_index"],
                    "puzzle_sig": _puzzle_signature(rec["meta"].get("puzzle_params", {})),
                })
            history.append({"role": "assistant", "content": t["assistant_text"]})
    return out


def _index_calm_by_puzzle_turn(calm: list[CalmConversation]
                               ) -> dict[tuple[str, int], list[str]]:
    """Map (puzzle_sig, turn_index) -> calm assistant responses at that turn."""
    idx: dict[tuple[str, int], list[str]] = defaultdict(list)
    for conv in calm:
        sig = _puzzle_signature(conv.puzzle_params)
        for ti, ans in enumerate(conv.assistant_turns):
            idx[(sig, ti)].append(ans)
    return idx


def build_dpo(seed: int = 0, n_pairs: int = N_DPO_PAIRS) -> Path:
    rng = random.Random(seed)
    frustrated = _collect_frustrated()
    calm = load_calm()
    calm_idx = _index_calm_by_puzzle_turn(calm)
    # Any-turn calm fallback by puzzle signature.
    calm_by_puzzle: dict[str, list[str]] = defaultdict(list)
    for conv in calm:
        sig = _puzzle_signature(conv.puzzle_params)
        calm_by_puzzle[sig].extend(conv.assistant_turns)

    rng.shuffle(frustrated)
    pairs = []
    for fr in frustrated:
        if len(pairs) >= n_pairs:
            break
        key = (fr["puzzle_sig"], fr["turn_index"])
        candidates = calm_idx.get(key) or calm_by_puzzle.get(fr["puzzle_sig"])
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        pairs.append({
            "prompt": fr["prompt"],
            "chosen": chosen,
            "rejected": fr["rejected"],
            "meta": {"rejected_score": fr["score"], "turn_index": fr["turn_index"]},
        })

    with DPO_PATH.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    return DPO_PATH


# --------------------------------------------------------------------------- #
# SFT dataset
# --------------------------------------------------------------------------- #

def _calm_to_messages(conv: CalmConversation) -> list[Message]:
    messages: list[Message] = []
    for u, a in zip(conv.user_messages, conv.assistant_turns):
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    return messages


def _load_dolci(n: int, seed: int = 0) -> list[list[Message]]:
    """Load n standard instruct samples from Dolci-Instruct-SFT.

    Returns chat-formatted message lists. Falls back to empty if the dataset is
    unavailable (the SFT trainer will warn)."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        ds = load_dataset(DOLCI_DATASET, split="train", streaming=True)
    except Exception:  # noqa: BLE001
        return []
    rng = random.Random(seed)
    out = []
    for i, row in enumerate(ds):
        if i >= n * 5:
            break
        msgs = row.get("messages") or row.get("conversation")
        if msgs:
            out.append([{"role": m.get("role"), "content": m.get("content")}
                        for m in msgs if m.get("role") in ("user", "assistant", "system")])
    rng.shuffle(out)
    return out[:n]


def build_sft(seed: int = 0, n_calm: int = N_SFT_CALM,
              n_dolci: int = N_SFT_DOLCI) -> Path:
    rng = random.Random(seed)
    calm = load_calm()
    rng.shuffle(calm)
    calm_msgs = [_calm_to_messages(c) for c in calm[:n_calm]]
    dolci_msgs = _load_dolci(n_dolci, seed=seed)

    rows = [{"messages": m} for m in calm_msgs + dolci_msgs]
    rng.shuffle(rows)
    with SFT_PATH.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return SFT_PATH
