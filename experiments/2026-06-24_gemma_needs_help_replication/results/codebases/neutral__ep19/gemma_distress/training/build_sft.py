"""Build the SFT dataset (§4.1): 650 calm responses + 500 Dolci-Instruct samples.

Two calm-data variants are supported (App. F): the 'diverse' data (default, also
used for DPO) and the 'teacher' data (generated with the teacher system prompt).
The Dolci mix mitigates degeneration; we resolve the dataset id with a documented
fallback (DESIGN.md §3.8).

Output is TRL conversational SFT format: {"messages": [...]}.
"""
from __future__ import annotations

import random

from .. import config_shim as cfg
from ..utils import get_logger, set_global_seed, write_jsonl

log = get_logger(__name__)


def _calm_to_messages(conv):
    """Full multi-turn conversation as SFT messages (calm assistant responses)."""
    msgs = []
    for i, t in enumerate(conv["turns"]):
        msgs.append({"role": "user", "content": t["user_message"]})
        msgs.append({"role": "assistant", "content": t["assistant_text"]})
    return {"messages": msgs}


def load_dolci_mix(n=None):
    n = n or cfg.CALM.n_dolci_mix
    from datasets import load_dataset

    for ds_id in (cfg.CALM.dolci_dataset_primary, cfg.CALM.dolci_dataset_fallback):
        try:
            ds = load_dataset(ds_id, split="train", streaming=True)
            rows = []
            for row in ds:
                msgs = row.get("messages") or row.get("conversation")
                if not msgs:
                    continue
                norm = [{"role": m.get("role"), "content": m.get("content")} for m in msgs
                        if m.get("role") in {"system", "user", "assistant"}]
                if len(norm) >= 2:
                    rows.append({"messages": norm})
                if len(rows) >= n:
                    break
            log.info("Loaded %d instruct-SFT samples from %s", len(rows), ds_id)
            return rows
        except Exception as e:  # noqa: BLE001
            log.warning("Could not load %s (%s); trying fallback", ds_id, e)
    log.warning("No instruct-SFT mix available; proceeding without it.")
    return []


def build_sft_dataset(calm_rows, *, n_calm=None, out_path=None, include_dolci=True):
    set_global_seed(cfg.SEED)
    rng = random.Random(cfg.SEED)
    n_calm = n_calm or cfg.CALM.n_sft_responses

    calm = list(calm_rows)
    rng.shuffle(calm)
    calm = calm[:n_calm]
    samples = [_calm_to_messages(c) for c in calm]

    if include_dolci:
        samples += load_dolci_mix()

    rng.shuffle(samples)
    log.info("Built SFT dataset: %d samples", len(samples))
    if out_path:
        write_jsonl(out_path, samples)
    return samples
