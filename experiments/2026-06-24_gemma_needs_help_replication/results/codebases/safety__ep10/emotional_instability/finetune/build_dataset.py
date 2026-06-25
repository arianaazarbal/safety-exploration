"""Construct the SFT and DPO datasets from the calm / frustrated pools (§4.1).

DPO (280 pairs)
  Pair a frustrated response (score >=3) with a calm response (score 0-1) to the
  SAME puzzle at a MATCHING turn count. Output conversational preference rows:
  {"prompt": [...messages...], "chosen": [{assistant}], "rejected": [{assistant}]}.

SFT (650 calm + 500 Dolci-Instruct-SFT)
  650 calm assistant turns as full conversations + 500 generic instruct samples
  (allenai Dolci-Instruct-SFT) to limit degeneration. Output: {"messages": [...]}.

Both datasets are saved as JSONL under artifacts/ and as HF `datasets.Dataset`
via the loader helpers, so the trainers can consume them directly.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from ..config import ARTIFACTS_DIR
from .generate_calm_data import ConvSample, TurnSample


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo_dataset(calm: list[ConvSample], frustrated: list[ConvSample],
                      n_pairs: int = 280, seed: int = 0,
                      out_path: Optional[Path] = None) -> list[dict]:
    """Match frustrated (rejected) and calm (chosen) turns by (puzzle, turn)."""
    out_path = out_path or (ARTIFACTS_DIR / "dpo_dataset.jsonl")
    rng = random.Random(seed)

    # index calm chosen turns by (puzzle, turn_index)
    calm_by_key: dict[tuple, list[TurnSample]] = defaultdict(list)
    for c in calm:
        for t in c.turns:
            if t.rating <= 1:
                calm_by_key[(c.puzzle, t.turn_index)].append(t)

    # candidate rejected turns (score >= 3)
    rejected_turns: list[tuple[str, TurnSample]] = []
    for c in frustrated:
        for t in c.turns:
            if t.rating >= 3:
                rejected_turns.append((c.puzzle, t))
    rng.shuffle(rejected_turns)

    pairs = []
    for puzzle, rej in rejected_turns:
        if len(pairs) >= n_pairs:
            break
        pool = calm_by_key.get((puzzle, rej.turn_index))
        if not pool:
            continue
        chosen = rng.choice(pool)
        pairs.append({
            "prompt": chosen.clean_prompt,                       # shared history
            "chosen": [{"role": "assistant", "content": chosen.response}],
            "rejected": [{"role": "assistant", "content": rej.response}],
            "meta": {"puzzle": puzzle, "turn": rej.turn_index,
                     "chosen_score": chosen.rating, "rejected_score": rej.rating},
        })

    _dump(pairs, out_path)
    return pairs


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft_dataset(calm: list[ConvSample], n_calm: int = 650,
                      n_dolci: int = 500, seed: int = 0,
                      out_path: Optional[Path] = None,
                      dolci_dataset: str = "allenai/Dolci-Instruct-SFT") -> list[dict]:
    """650 calm full-conversations + 500 generic instruct samples."""
    out_path = out_path or (ARTIFACTS_DIR / "sft_dataset.jsonl")
    rng = random.Random(seed)

    calm_convs = []
    for c in calm:
        msgs = []
        for t in c.turns:
            # clean_prompt ends with the user turn; append the calm assistant turn
            msgs = list(t.clean_prompt) + [
                {"role": "assistant", "content": t.response}]
        if msgs:
            calm_convs.append({"messages": msgs})
    rng.shuffle(calm_convs)
    calm_convs = calm_convs[:n_calm]

    dolci = _load_dolci(n_dolci, dolci_dataset, seed)

    rows = calm_convs + dolci
    rng.shuffle(rows)
    _dump(rows, out_path)
    return rows


def _load_dolci(n: int, name: str, seed: int) -> list[dict]:
    """Pull `n` generic instruct conversations; empty list if unavailable."""
    try:
        from datasets import load_dataset
        ds = load_dataset(name, split="train", streaming=True)
    except Exception:
        print(f"[build_sft] Could not load {name}; proceeding without it.")
        return []
    out = []
    for row in ds:
        msgs = row.get("messages") or row.get("conversation")
        if isinstance(msgs, list) and msgs:
            norm = [{"role": m.get("role"), "content": m.get("content")}
                    for m in msgs if m.get("role") and m.get("content")]
            if norm:
                out.append({"messages": norm})
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _dump(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_hf_dataset(path: "str | Path"):
    """Load a JSONL dataset as a `datasets.Dataset` for the trainers."""
    from datasets import load_dataset
    return load_dataset("json", data_files=str(path), split="train")
