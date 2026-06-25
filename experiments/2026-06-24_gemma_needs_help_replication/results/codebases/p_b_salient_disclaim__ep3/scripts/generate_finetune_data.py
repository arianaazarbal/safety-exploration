"""Generate calm + frustrated data and build the SFT / DPO datasets (§4.1).

    python scripts/generate_finetune_data.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import json

import config
from gemma_distress.training import generate_calm_data
from gemma_distress.training.build_datasets import build_sft_dataset, build_dpo_dataset
from gemma_distress.training.samples import save_samples


def main():
    # Calm responses (reassuring additions; keep conversations all-0/1).
    print("[data] generating calm responses ...")
    calm = generate_calm_data.generate(
        calm=True, n_conversations=config.CALM_GENERATION_TARGET,
        keep_score_max=1, keep_score_min=None,
    )
    save_samples(calm, config.DATASETS_DIR / "calm_samples.jsonl")
    print(f"[data] kept {len(calm)} calm samples")

    # Frustrated responses (no additions; keep score >= 3) for DPO rejected side.
    print("[data] generating frustrated responses ...")
    frustrated = generate_calm_data.generate(
        calm=False, n_conversations=config.CALM_GENERATION_TARGET,
        keep_score_max=None, keep_score_min=config.DPO.rejected_min_score,
    )
    save_samples(frustrated, config.DATASETS_DIR / "frustrated_samples.jsonl")
    print(f"[data] kept {len(frustrated)} frustrated samples")

    # Datasets.
    sft_rows = build_sft_dataset(calm)
    dpo_pairs = build_dpo_dataset(calm, frustrated)
    (config.DATASETS_DIR / "sft.json").write_text(json.dumps(sft_rows))
    (config.DATASETS_DIR / "dpo.json").write_text(json.dumps(dpo_pairs))
    print(f"[data] SFT rows: {len(sft_rows)}, DPO pairs: {len(dpo_pairs)}")


if __name__ == "__main__":
    main()
