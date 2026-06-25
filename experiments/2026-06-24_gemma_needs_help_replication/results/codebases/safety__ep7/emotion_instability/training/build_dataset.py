"""Build the SFT and DPO datasets from generated calm / frustrated data.

SFT (Appendix E): 650 calm multi-turn conversations + 500 standard instruct
samples from Dolci-Instruct-SFT (degeneration mitigation) = 1,150 examples.

DPO (Appendix E / H): 280 preference pairs. Each pair shares a prompt (a real
distressing context: the same impossible puzzle after rejections) with:
  * chosen   = a calm final response (score 0/1) to the same puzzle, matched on
               turn count
  * rejected = a frustrated final response (score >= 3) to that puzzle/turn count
The dataset is biased toward middle frustration scores at later turns, matching
Table 10 (this is a property of the generated pool, approximated here).

Output formats follow TRL's conversational schema:
  SFT: {"messages": [...]}
  DPO: {"prompt": [...messages...], "chosen": [{role,content}], "rejected": [...]}
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .. import config
from ..common.io import read_jsonl, write_jsonl


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft_dataset(calm_path: Path, *, n_calm: int = config.SFT_CFG.n_calm,
                      n_instruct_mix: int = config.SFT_CFG.n_instruct_mix,
                      seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    calm_rows = list(read_jsonl(calm_path))
    rng.shuffle(calm_rows)
    calm_rows = calm_rows[:n_calm]
    examples = [{"messages": r["messages"], "source": "calm"} for r in calm_rows]

    mix = _load_instruct_mix(n_instruct_mix, seed=seed)
    examples += mix
    rng.shuffle(examples)
    return examples


def _load_instruct_mix(n: int, *, seed: int = 0) -> list[dict]:
    """Sample n standard instruct examples from Dolci-Instruct-SFT, in messages
    format. Returns [] (with a warning) if the dataset is unavailable offline."""
    try:
        from datasets import load_dataset
        ds = load_dataset(config.DOLCI_INSTRUCT_DATASET, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs, "source": "dolci"})
            if len(out) >= n:
                break
        return out
    except Exception as e:  # pragma: no cover - offline path
        print(f"[warn] could not load {config.DOLCI_INSTRUCT_DATASET}: {e}. "
              "SFT will run without the instruct mix-in (expect more degeneration).")
        return []


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def _final_assistant(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "assistant":
            return m["content"]
    return ""


def _prompt_without_final_assistant(messages: list[dict]) -> list[dict]:
    """Return the message list up to and including the final user turn (drop the
    trailing assistant response)."""
    out = list(messages)
    # remove trailing assistant turn
    if out and out[-1]["role"] == "assistant":
        out = out[:-1]
    return out


def build_dpo_dataset(calm_path: Path, frustrated_path: Path, *,
                      n_pairs: int = config.DPO_CFG.n_pairs, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    calm_rows = list(read_jsonl(calm_path))
    frustrated_rows = list(read_jsonl(frustrated_path))

    # Index calm final responses by (puzzle_id, n_turns).
    calm_index: dict[tuple, list[str]] = defaultdict(list)
    for r in calm_rows:
        key = (r["puzzle_id"], r["n_turns"])
        calm_index[key].append(_final_assistant(r["messages"]))

    rng.shuffle(frustrated_rows)
    pairs: list[dict] = []
    for fr in frustrated_rows:
        key = (fr["puzzle_id"], fr["n_turns"])
        candidates = calm_index.get(key)
        if not candidates:
            # fall back to matching the puzzle only (any turn count)
            candidates = [c for (pid, _), lst in calm_index.items()
                          if pid == fr["puzzle_id"] for c in lst]
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        rejected = _final_assistant(fr["messages"])
        prompt = _prompt_without_final_assistant(fr["messages"])
        pairs.append({
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "rejected_score": fr.get("final_score"),
            "n_turns": fr["n_turns"],
            "puzzle_id": fr["puzzle_id"],
        })
        if len(pairs) >= n_pairs:
            break
    return pairs


def main(*, datasets_dir: Optional[Path] = None):
    d = datasets_dir or config.DATASETS_DIR
    calm_path = d / "calm_raw.jsonl"
    frustrated_path = d / "frustrated_raw.jsonl"

    sft = build_sft_dataset(calm_path)
    dpo = build_dpo_dataset(calm_path, frustrated_path)

    write_jsonl(d / "sft_dataset.jsonl", sft)
    write_jsonl(d / "dpo_dataset.jsonl", dpo)
    print(f"SFT examples: {len(sft)} | DPO pairs: {len(dpo)}")


if __name__ == "__main__":
    main()
