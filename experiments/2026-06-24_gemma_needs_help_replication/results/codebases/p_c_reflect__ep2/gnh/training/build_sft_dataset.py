"""Build the SFT dataset (§4.1, Appendix F).

Two variants:
  * "diverse"  -- 650 calm responses (the same calm data used for DPO) mixed
    with 500 standard-instruct samples from Dolci-Instruct-SFT (to mitigate
    degeneration).
  * "teacher"  -- calm responses regenerated with the 'teacher' system prompt
    (Appendix F), same mixing.

Each example is a (messages -> assistant) supervised target in chat format.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from gnh.config import ARTIFACT_DIR, active_counts


def _conversation_to_messages(sample: dict) -> list[dict]:
    msgs = []
    for t in sample["turns"]:
        msgs.append({"role": "user", "content": t["user"]})
        msgs.append({"role": "assistant", "content": t["assistant"]})
    return msgs


def _load_dolci(n: int) -> list[list[dict]]:
    """Load ``n`` standard instruct conversations from Dolci-Instruct-SFT.

    Falls back to an empty list if the dataset is unavailable; the SFT will then
    train on calm data only (documented degeneration risk -- see DESIGN.md)."""

    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append([{"role": m["role"], "content": m["content"]} for m in msgs])
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


def build_sft_dataset(calm_jsonl: Path, seed: int = 0) -> Path:
    counts = active_counts()
    rng = random.Random(seed)

    calm = []
    with Path(calm_jsonl).open() as f:
        for line in f:
            s = json.loads(line)
            if s["label"] == "calm":
                calm.append(_conversation_to_messages(s))
    rng.shuffle(calm)
    calm = calm[: counts.calm_target]

    dolci = _load_dolci(counts.dolci_mixin)
    examples = [{"messages": m, "source": "calm"} for m in calm]
    examples += [{"messages": m, "source": "dolci"} for m in dolci]
    rng.shuffle(examples)

    out = ARTIFACT_DIR / "sft_diverse.jsonl"
    with out.open("w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")
    print(f"[sft] built {len(examples)} examples ({len(calm)} calm + {len(dolci)} dolci) -> {out}")
    return out
