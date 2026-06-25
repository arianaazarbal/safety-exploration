"""Build the SFT dataset (Section 4.1 / Appendix F).

650 calm conversations (1-3 turn) mixed with 500 standard instruct samples from
Dolci-Instruct-SFT to mitigate degeneration. Two calm sources are supported:
the 'diverse' data (also used for DPO) and the 'teacher' data (Appendix F variant
that ends up *increasing* frustration). Output is TRL conversational SFT format:
  {"messages": [{role, content}, ...]}
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import SFT, TRAINING_DIR, ensure_dirs
from ..eval.schema import Conversation, read_jsonl
from .generate_calm import CALM_PATH, TEACHER_PATH


def _convo_to_messages(c: Conversation) -> dict:
    msgs = []
    for t in c.turns:
        msgs.append({"role": "user", "content": t.user})
        msgs.append({"role": "assistant", "content": t.assistant})
    return {"messages": msgs}


def _load_instruct_mix(n: int, seed: int) -> list[dict]:
    """Sample ``n`` standard instruct conversations from Dolci-Instruct-SFT.

    Returns conversational ``{"messages": [...]}`` records. Falls back to an empty
    list (with a warning) if the dataset is unavailable.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(SFT.instruct_mix_dataset, split="train", streaming=True)
        out: list[dict] = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs and isinstance(msgs, list):
                norm = [{"role": m.get("role"), "content": m.get("content")} for m in msgs]
                if all(m["role"] and m["content"] for m in norm):
                    out.append({"messages": norm})
            if len(out) >= n * 3:  # gather a window then sample for determinism
                break
        rng = random.Random(seed)
        return rng.sample(out, min(n, len(out)))
    except Exception as e:  # pragma: no cover - dataset/network dependent
        print(f"[sft] WARNING: could not load {SFT.instruct_mix_dataset} ({e}); "
              "proceeding without instruct mix.")
        return []


def build_sft_dataset(
    *, calm_source: str = "diverse", n_calm: int = SFT.n_calm,
    n_mix: int = SFT.n_instruct_mix, seed: int = 0,
) -> Path:
    """Assemble and write the SFT dataset (calm + instruct mix), shuffled."""
    ensure_dirs()
    calm_path = CALM_PATH if calm_source == "diverse" else TEACHER_PATH
    calm = [_convo_to_messages(c) for c in read_jsonl(calm_path)][:n_calm]
    mix = _load_instruct_mix(n_mix, seed)

    data = calm + mix
    random.Random(seed).shuffle(data)

    out_path = TRAINING_DIR / f"sft_{calm_source}.jsonl"
    with open(out_path, "w") as fh:
        for d in data:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[sft] wrote {len(data)} samples ({len(calm)} calm + {len(mix)} mix) -> {out_path}")
    return out_path
