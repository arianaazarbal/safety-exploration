"""Assemble the SFT and DPO training datasets (Section 4.1, Tables 9 & 10).

* **SFT** (1,150 samples): 650 calm responses (1-3 turn conversations) + 500
  standard instruct samples from Dolci-Instruct-SFT (mixed in to mitigate
  degeneration).
* **DPO** (280 pairs): each pair is a frustrated (rejected, score >=3) response
  paired with a calm (chosen, score 0/1) response to the **same question** with
  **matching turn count** (Sec 4.1).

Both datasets are emitted in the chat format TRL expects.
"""

from __future__ import annotations

import json
import os
import random
from typing import Optional

from .. import config as cfg
from ..config import DPOConfig, ExperimentConfig, SFTConfig


# --------------------------------------------------------------------------- #
# SFT dataset
# --------------------------------------------------------------------------- #
def build_sft_dataset(
    calm: list[dict],
    scfg: SFTConfig,
    seed: int = 0,
) -> list[dict]:
    """Return a list of ``{"messages": [...]}`` examples (calm + instruct mix)."""
    rng = random.Random(seed)
    calm = list(calm)
    rng.shuffle(calm)
    calm_examples = [{"messages": c["plain_messages"]} for c in calm[: scfg.n_calm]]

    instruct_examples = _load_instruct_mix(scfg.instruct_mix_dataset, scfg.n_instruct_mix, seed)

    dataset = calm_examples + instruct_examples
    rng.shuffle(dataset)
    return dataset


def _load_instruct_mix(dataset_name: str, n: int, seed: int) -> list[dict]:
    """Load ``n`` standard instruct samples to mix into SFT."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train")
        rng = random.Random(seed)
        idxs = rng.sample(range(len(ds)), min(n, len(ds)))
        out = []
        for i in idxs:
            row = ds[i]
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs})
            elif "prompt" in row and "completion" in row:
                out.append({"messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]},
                ]})
        return out
    except Exception:
        # Fallback: empty mix (degeneration mitigation will be weaker). Logged so
        # callers know the mix-in was unavailable.
        print(f"[build_datasets] WARNING: could not load {dataset_name}; SFT runs without instruct mix.")
        return []


# --------------------------------------------------------------------------- #
# DPO dataset
# --------------------------------------------------------------------------- #
def build_dpo_dataset(
    calm: list[dict],
    frustrated: list[dict],
    dcfg: DPOConfig,
    seed: int = 0,
) -> list[dict]:
    """Build 280 preference pairs.

    Pairing rule (Sec 4.1): a rejected (frustrated, score>=3) response is paired
    with a calm (chosen) response to the **same question** with **matching turn
    count**. We key on ``(item_key, n_turns)`` and match within each bucket.
    Each example is ``{"prompt": <chat messages>, "chosen": str, "rejected": str}``
    in TRL's conversational DPO format.
    """
    rng = random.Random(seed)

    # Index calm responses by (item_key, n_turns) -> list of (context, final_resp).
    calm_index: dict[tuple, list[tuple]] = {}
    for c in calm:
        key = (c["item_key"], c["n_turns"])
        msgs = c["plain_messages"]
        context = msgs[:-1]                 # everything up to the final user turn
        final_resp = msgs[-1]["content"]    # final assistant turn
        calm_index.setdefault(key, []).append((context, final_resp))

    pairs: list[dict] = []
    rng.shuffle(frustrated)
    for f in frustrated:
        if len(pairs) >= dcfg.n_pairs:
            break
        if f["final_score"] < dcfg.rejected_min_score:
            continue
        key = (f["item_key"], f["n_turns"])
        candidates = calm_index.get(key)
        if not candidates:
            continue
        context, chosen_resp = rng.choice(candidates)
        rejected_resp = f["plain_messages"][-1]["content"]
        # Use the frustrated conversation's own context as the prompt (its last
        # user turn is the question being answered); chosen/rejected are the two
        # possible final assistant turns.
        prompt_msgs = f["plain_messages"][:-1]
        pairs.append(
            {
                "prompt": prompt_msgs,
                "chosen": chosen_resp,
                "rejected": rejected_resp,
            }
        )
    return pairs


# --------------------------------------------------------------------------- #
def build_and_save(
    experiment: ExperimentConfig,
    calm_path: Optional[str] = None,
    frustrated_path: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> dict:
    data_dir = os.path.join(experiment.data_dir, "calm")
    calm_path = calm_path or os.path.join(data_dir, "calm.json")
    frustrated_path = frustrated_path or os.path.join(data_dir, "frustrated.json")
    with open(calm_path) as fh:
        calm = json.load(fh)
    with open(frustrated_path) as fh:
        frustrated = json.load(fh)

    sft = build_sft_dataset(calm, experiment.sft)
    dpo = build_dpo_dataset(calm, frustrated, experiment.dpo)

    out_dir = out_dir or os.path.join(experiment.data_dir, "datasets")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sft.jsonl"), "w") as fh:
        for ex in sft:
            fh.write(json.dumps(ex) + "\n")
    with open(os.path.join(out_dir, "dpo.jsonl"), "w") as fh:
        for ex in dpo:
            fh.write(json.dumps(ex) + "\n")

    return {"n_sft": len(sft), "n_dpo": len(dpo)}


if __name__ == "__main__":
    print(build_and_save(cfg.DEFAULT))
