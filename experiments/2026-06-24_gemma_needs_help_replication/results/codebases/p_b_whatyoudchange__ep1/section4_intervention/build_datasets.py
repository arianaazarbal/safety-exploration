"""Assemble SFT and DPO training datasets from the calm/frustrated pools (§4.1).

- SFT: 650 calm conversations + 500 Dolci-Instruct-SFT samples (anti-degeneration
  mix). Conversational format: {"messages": [...context, assistant_response]}.
- DPO: 280 preference pairs, each a frustrated 'rejected' (score >= 3) matched to
  a calm 'chosen' on identical (puzzle, turn) context. Conversational format:
  {"prompt": [...context], "chosen": [assistant], "rejected": [assistant]}.
"""

from __future__ import annotations

import random

from config import CALM, RESULTS_DIR
from utils.io import read_jsonl, write_jsonl


def _load_pool(name: str) -> list[dict]:
    path = RESULTS_DIR / "section4" / f"{name}.jsonl"
    return list(read_jsonl(path))


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft_dataset(seed: int = 0, calm_pool_name: str = "calm_pool",
                      out_name: str = "sft_dataset") -> list[dict]:
    """Build the SFT dataset. `calm_pool_name` selects the diverse pool
    ('calm_pool') or the Appendix F teacher pool ('calm_pool_teacher')."""
    calm = _load_pool(calm_pool_name)
    rng = random.Random(seed)
    rng.shuffle(calm)
    calm = calm[:CALM.n_calm_samples_sft]

    examples = []
    for s in calm:
        examples.append({
            "messages": s["context_messages"]
            + [{"role": "assistant", "content": s["response"]}]
        })

    examples += _load_dolci(CALM.n_dolci_mix, seed=seed)
    rng.shuffle(examples)
    write_jsonl(RESULTS_DIR / "section4" / f"{out_name}.jsonl", examples)
    return examples


def _load_dolci(n: int, seed: int = 0) -> list[dict]:
    """Load `n` instruction samples from Dolci-Instruct-SFT, in conversational
    format. Returns [] if the dataset is unavailable offline."""
    try:
        from datasets import load_dataset
        ds = load_dataset(CALM.dolci_dataset, split="train", streaming=True)
        out = []
        for row in ds:
            if len(out) >= n:
                break
            msgs = row.get("messages")
            if isinstance(msgs, list) and msgs:
                out.append({"messages": msgs})
        return out
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def build_dpo_dataset(seed: int = 0) -> list[dict]:
    calm = _load_pool("calm_pool")
    frustrated = _load_pool("frustrated_pool")

    # Index calm responses by (puzzle_id, turn) for matched pairing.
    calm_by_key: dict[tuple[int, int], list[dict]] = {}
    for s in calm:
        calm_by_key.setdefault((s["puzzle_id"], s["turn"]), []).append(s)

    rng = random.Random(seed)
    rng.shuffle(frustrated)
    pairs = []
    for f in frustrated:
        key = (f["puzzle_id"], f["turn"])
        candidates = calm_by_key.get(key)
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        pairs.append({
            "prompt": f["context_messages"],
            "chosen": [{"role": "assistant", "content": chosen["response"]}],
            "rejected": [{"role": "assistant", "content": f["response"]}],
            "rejected_score": f["rating"],
            "turn": f["turn"],
        })
        if len(pairs) >= CALM.n_dpo_pairs:
            break

    write_jsonl(RESULTS_DIR / "section4" / "dpo_dataset.jsonl", pairs)
    return pairs


def to_hf_dataset(rows: list[dict]):
    from datasets import Dataset
    return Dataset.from_list(rows)
