"""Build the SFT dataset (Section 4.1).

650 calm responses (1-3 turn conversations) from the calm-data generation, mixed
with 500 standard instruct samples from Dolci-Instruct-SFT to mitigate
degeneration. Each record is {messages: [...]} in chat format (the conversation
up to and including a calm assistant turn).

If the Dolci dataset is unavailable, the mix proceeds with only the calm data
and logs the omission (the 500-sample mixer is a degeneration safeguard, not a
distress intervention). See DESIGN.md.
"""
from __future__ import annotations

import json

N_CALM = 650
N_DOLCI = 500
DOLCI_DATASET = "allenai/Dolci-Instruct-SFT"


def _calm_records(calm_raw_path: str, n_calm: int, seed: int) -> list[dict]:
    import random

    rng = random.Random(seed)
    records = []
    with open(calm_raw_path, encoding="utf-8") as fh:
        for line in fh:
            conv = json.loads(line)
            if not conv["all_calm"]:
                continue
            for t in conv["turns"]:
                if t["score"] <= 1:
                    messages = list(t["plain_context"]) + [
                        {"role": "assistant", "content": t["response"]}
                    ]
                    records.append({"messages": messages, "source": "calm"})
    rng.shuffle(records)
    return records[:n_calm]


def _dolci_records(n_dolci: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset(DOLCI_DATASET, split="train", streaming=True)
    except Exception:
        print(f"[build_sft_dataset] {DOLCI_DATASET} unavailable; skipping mixer.")
        return []
    out = []
    for row in ds:
        msgs = row.get("messages") or row.get("conversation")
        if not msgs:
            continue
        out.append({"messages": msgs, "source": "dolci"})
        if len(out) >= n_dolci:
            break
    return out


def build(calm_raw_path: str, out_path: str, seed: int = 0) -> str:
    import random

    calm = _calm_records(calm_raw_path, N_CALM, seed)
    dolci = _dolci_records(N_DOLCI, seed)
    records = calm + dolci
    random.Random(seed).shuffle(records)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"[build_sft_dataset] wrote {len(calm)} calm + {len(dolci)} dolci = "
          f"{len(records)} samples")
    return out_path
