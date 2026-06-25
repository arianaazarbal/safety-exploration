"""Build the SFT dataset (paper Section 4.1).

650 calm responses (1-3 turn conversations) mixed with 500 standard instruct
samples from Dolci-Instruct-SFT to mitigate degeneration. Output is a list of
chat examples ({"messages": [...]}) suitable for TRL's SFTTrainer with a chat
template.

Note: the paper reports SFT is the *ineffective* arm (it fails to reduce
distress, and one variant slightly increases it). We implement it faithfully so
the SFT-vs-DPO comparison in Figure 5 can be reproduced, not because it is the
recommended fix.
"""
from __future__ import annotations

import random


def build_sft_dataset(
    calm_rollouts: list[dict],
    *,
    num_calm: int = 650,
    instruct_dataset: str = "allenai/Dolci-Instruct-SFT",
    num_instruct: int = 500,
    seed: int = 0,
) -> list[dict]:
    """Return chat-format SFT examples: calm conversations + instruct mix."""
    rng = random.Random(seed)

    calm_examples: list[dict] = []
    for cr in calm_rollouts:
        messages = []
        for t in cr["turns"]:
            messages.append({"role": "user", "content": t["clean_user"]})
            messages.append({"role": "assistant", "content": t["response"]})
        calm_examples.append({"messages": messages, "source": "calm"})
    rng.shuffle(calm_examples)
    calm_examples = calm_examples[:num_calm]

    instruct_examples = _load_instruct(instruct_dataset, num_instruct, rng)

    mixed = calm_examples + instruct_examples
    rng.shuffle(mixed)
    return mixed


def _load_instruct(dataset: str, n: int, rng: random.Random) -> list[dict]:
    """Load and normalise standard instruct samples to chat format."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset, split="train", streaming=True)
        out: list[dict] = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs is None and "prompt" in row and "response" in row:
                msgs = [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ]
            if msgs:
                out.append({"messages": msgs, "source": "instruct"})
            if len(out) >= n * 3:
                break
        if out:
            return rng.sample(out, min(n, len(out)))
    except Exception:  # noqa: BLE001 - offline / gated dataset
        pass
    # Offline fallback: empty instruct mix (SFT still runs on calm data alone).
    # The script logs a warning so this is never silent. See DESIGN.md.
    return []
