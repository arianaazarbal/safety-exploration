"""Build the SFT dataset (Section 4.1).

650 calm responses (1-3 turn conversations) mixed with 500 standard instruct
samples from Dolci-Instruct-SFT to mitigate degeneration. The paper finds SFT
ineffective (and the 'teacher' variant counter-productive); we build the
'diverse' dataset used in the main text and DPO, plus an optional 'teacher'
variant for the Appendix F ablation.

Output is a TRL SFT JSONL with chat-format `messages`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from emotional_instability.finetune.calm_data import (load_calm_rollouts,  # noqa: E402
                                                      is_all_calm)

N_CALM = 650
N_DOLCI = 500


def _calm_messages(calm_path: Path, limit: int) -> list[dict]:
    """Calm rollouts -> chat-format SFT samples (full multi-turn conversations).

    Each sample is the conversation with calm assistant responses, reassurance
    additions already stripped in the calm-data records.
    """
    samples = []
    for ro in load_calm_rollouts(calm_path):
        if not is_all_calm(ro):
            continue
        messages = []
        for t in ro["turns"]:
            messages.append({"role": "user", "content": t["user_message"]})
            messages.append({"role": "assistant", "content": t["response"]})
        samples.append({"messages": messages})
        if len(samples) >= limit:
            break
    return samples


def _dolci_messages(limit: int, dataset_name="allenai/Dolci-Instruct-SFT") -> list[dict]:
    """Standard instruct data to mix in. Falls back to an empty list (with a
    warning) if the dataset is unavailable, so SFT can still run calm-only.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                # try (instruction, response) shape
                if "prompt" in row and "completion" in row:
                    msgs = [{"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": row["completion"]}]
                else:
                    continue
            out.append({"messages": msgs})
            if len(out) >= limit:
                break
        return out
    except Exception as e:  # pragma: no cover
        print(f"[sft_dataset] Dolci unavailable ({e}); using calm-only SFT mix.")
        return []


def build_sft_dataset(calm_path: Path, *, n_calm: int = N_CALM,
                      n_dolci: int = N_DOLCI,
                      out_path: Optional[Path] = None) -> Path:
    out_path = out_path or (config.FINETUNE_DIR / "sft_diverse.jsonl")
    samples = _calm_messages(calm_path, n_calm) + _dolci_messages(n_dolci)
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return out_path
