"""Build the SFT dataset (Section 4.1).

650 calm responses (1-3 turn conversations) in chat format, mixed with 500
standard-instruct samples from Dolci-Instruct-SFT to mitigate degeneration.
Output is TRL conversational format: {"messages": [<role/content>...]}.

Two calm sources are supported (Appendix F): the 'diverse' reassured data and
the 'teacher' data; the caller chooses which CalmConversation list to pass.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import ExperimentConfig
from .calm_data import CalmConversation


def _calm_to_messages(c: CalmConversation) -> dict:
    msgs = [{"role": m.role, "content": m.content} for m in c.stripped_messages()]
    return {"messages": msgs}


def _load_dolci(n: int, dataset: str, seed: int) -> list[dict]:
    """Load n standard-instruct samples in chat format. Falls back to empty."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset, split="train")
        idxs = random.Random(seed).sample(range(len(ds)), k=min(n, len(ds)))
        out = []
        for i in idxs:
            row = ds[i]
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": [
                    {"role": m["role"], "content": m["content"]} for m in msgs
                ]})
        return out
    except Exception:  # noqa: BLE001 - dataset unavailable offline
        return []


def build_sft_dataset(
    calm: list[CalmConversation],
    exp: ExperimentConfig,
    out_path: str | Path,
) -> Path:
    cfg = exp.section("calm_data")
    rng = random.Random(exp.seed)

    calm_msgs = [_calm_to_messages(c) for c in calm]
    rng.shuffle(calm_msgs)
    calm_msgs = calm_msgs[: cfg["sft_calm_responses"]]

    dolci = _load_dolci(cfg["sft_dolci_samples"], cfg["dolci_dataset"], exp.seed)

    combined = calm_msgs + dolci
    rng.shuffle(combined)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in combined:
            f.write(json.dumps(rec) + "\n")
    return out_path
