"""Build the SFT dataset (Section 4.1 / Appendix E/F).

650 calm responses (1-3 turn conversations, max_score <=1) mixed with 500
standard-instruct samples from Dolci-Instruct-SFT to mitigate degeneration.
Two variants:
  * "diverse": calm responses generated with the reassuring prefix/suffix
    (also the source of the DPO chosen responses);
  * "teacher": calm responses generated with the Appendix-F teacher persona.
Both emit TRL conversational SFT format: {"messages": [...]}.
"""

from __future__ import annotations

import random
from pathlib import Path

from ..utils.io import read_jsonl, write_jsonl


def _calm_conversations(calm_path):
    for r in read_jsonl(calm_path):
        if r.get("is_calm"):
            yield {"messages": r["messages"]}


def _load_instruct_mix(dataset_name: str, n: int, seed: int) -> list[dict]:
    """Load n standard-instruct conversations to mix in. Falls back to an empty
    list (with a warning) if the dataset can't be fetched offline."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for row in ds:
            if "messages" in row:
                out.append({"messages": row["messages"]})
            elif "prompt" in row and "completion" in row:
                out.append({"messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]},
                ]})
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[sft] could not load instruct mix '{dataset_name}' ({e}); "
              "proceeding without it (expect more degeneration).")
        return []


def build_sft_dataset(
    calm_path: str | Path,
    cfg_training,
    *,
    seed: int = 0,
    out_path: str | Path = "results/training/sft_diverse.jsonl",
) -> Path:
    rng = random.Random(seed)
    calm = list(_calm_conversations(calm_path))
    rng.shuffle(calm)
    calm = calm[: cfg_training.sft_n_calm]
    mix = _load_instruct_mix(cfg_training.instruct_mix_dataset,
                             cfg_training.sft_n_instruct_mix, seed)
    data = calm + mix
    rng.shuffle(data)
    write_jsonl(out_path, data)
    print(f"[sft] wrote {len(data)} samples ({len(calm)} calm + {len(mix)} "
          f"instruct) -> {out_path}")
    return Path(out_path)
