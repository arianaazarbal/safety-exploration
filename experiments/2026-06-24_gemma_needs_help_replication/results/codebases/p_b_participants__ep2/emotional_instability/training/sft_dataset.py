"""Build the SFT dataset (Section 4.1).

650 calm responses (1-3 turn conversations) mixed with 500 samples of standard
instruct data from Dolci-Instruct-SFT (Team-Olmo et al. 2025) to mitigate
degeneration. Total 1,150 samples (Table 9). Calm conversations are converted to
full multi-turn chat examples; the Dolci samples are single-turn instruct pairs.

If the Dolci dataset is unavailable, we warn loudly and fall back to a small
generic instruct mix so the pipeline still runs (results will not match paper).
"""

from __future__ import annotations

import logging
import os

from ..config import RunConfig
from ..storage import write_json
from .calm_data import CalmConversation

logger = logging.getLogger("emotional_instability.training.sft_dataset")

N_CALM = 650
N_DOLCI = 500


def _calm_to_messages(c: CalmConversation) -> list[dict]:
    msgs = []
    for t in c.turns:
        msgs.append({"role": "user", "content": t["user"]})
        msgs.append({"role": "assistant", "content": t["assistant"]})
    return msgs


def _load_dolci(n: int) -> list[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs})
            elif row.get("prompt") and row.get("completion"):
                out.append({"messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]},
                ]})
            if len(out) >= n:
                break
        if out:
            return out
        raise RuntimeError("no usable Dolci rows")
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Could not load Dolci-Instruct-SFT (%s); using a tiny generic "
            "fallback. Results will not match the paper.", e,
        )
        return [{"messages": [
            {"role": "user", "content": f"Explain concept #{i} simply."},
            {"role": "assistant", "content": "Here is a clear explanation ..."},
        ]} for i in range(n)]


def build_sft_dataset(cfg: RunConfig, calm: list[CalmConversation],
                      n_calm: int = N_CALM, n_dolci: int = N_DOLCI) -> list[dict]:
    calm_examples = [{"messages": _calm_to_messages(c)} for c in calm[:n_calm]]
    if len(calm_examples) < n_calm:
        logger.warning("Only %d calm examples available (< %d requested)",
                       len(calm_examples), n_calm)
    dolci = _load_dolci(n_dolci)

    dataset = calm_examples + dolci
    out_path = os.path.join(cfg.output_dir, "training", "sft", "dataset.json")
    write_json(out_path, dataset)
    logger.info("Built SFT dataset: %d calm + %d dolci = %d total",
                len(calm_examples), len(dolci), len(dataset))
    return dataset
