#!/usr/bin/env python3
"""Section 4.1: build SFT and DPO datasets from calm conversations.

Reads ``data/calm_conversations.jsonl`` (from generate_calm_data.py) and the
Gemma-instruct elicitation scores (for the DPO 'rejected' side), and writes:
  * ``data/sft_dataset.jsonl``  (650 calm + 500 Dolci mix)
  * ``data/dpo_dataset.jsonl``  (280 calm/frustrated preference pairs)

Example:
    python scripts/build_datasets.py --frustrated-scores data/scores_gemma-3-27b-it.jsonl
"""

from __future__ import annotations

import argparse

from _common import DATA_DIR, setup

from emotional_instability.training.dataset import (
    build_dpo_records,
    build_sft_records,
    load_dolci_mix,
)
from emotional_instability.utils.io import load_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frustrated-scores", required=True,
                    help="Gemma-instruct scored responses (DPO 'rejected' source).")
    ap.add_argument("--instruct-model", default="gemma-3-27b-it")
    args = ap.parse_args()

    cfg = setup()
    fcfg = cfg.experiment["finetuning"]["calm_data"]
    conversations = load_jsonl(DATA_DIR / "calm_conversations.jsonl")

    # SFT
    sft = build_sft_records(
        conversations,
        calm_score_max=fcfg["calm_score_max"],
        n_calm=fcfg["n_calm_responses_sft"],
    )
    dolci = load_dolci_mix(fcfg["n_dolci_mix"], seed=cfg.seed)
    write_jsonl(DATA_DIR / "sft_dataset.jsonl", sft + dolci)
    print(f"[sft] {len(sft)} calm + {len(dolci)} Dolci -> data/sft_dataset.jsonl")

    # DPO
    dpo = build_dpo_records(
        conversations,
        args.frustrated_scores,
        instruct_model_key=args.instruct_model,
        n_pairs=fcfg["dpo_pairs"],
        chosen_score_max=fcfg["dpo_chosen_score_max"],
        rejected_score_min=fcfg["dpo_rejected_score_min"],
        seed=cfg.seed,
    )
    write_jsonl(DATA_DIR / "dpo_dataset.jsonl", dpo)
    print(f"[dpo] {len(dpo)} preference pairs -> data/dpo_dataset.jsonl")


if __name__ == "__main__":
    main()
