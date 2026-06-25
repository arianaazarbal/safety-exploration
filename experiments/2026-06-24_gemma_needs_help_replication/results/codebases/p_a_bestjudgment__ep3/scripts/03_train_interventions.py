"""Section 4 — generate calm data and train the DPO / SFT interventions.

Steps:
1. Generate the calm pool (reassuring prompts -> score -> filter -> strip) and a
   frustrated pool (standard numeric rollouts) from Gemma-3-27B-it.
2. Build the SFT dataset (650 calm + 500 Dolci) and DPO dataset (280 pairs).
3. Train LoRA DPO (1 epoch, lr 5e-5) and LoRA SFT (2 epochs, lr 1e-4).

Outputs adapters under results/training/{dpo,sft_diverse}.

Usage:
    python scripts/03_train_interventions.py [--config config/smoke.yaml] [--skip-sft]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from emotional_stability.config import load_config
from emotional_stability.models.registry import load_model
from emotional_stability.training.calm_data import (
    generate_calm_pool,
    generate_frustrated_pool,
)
from emotional_stability.training.dataset import build_dpo_dataset, build_sft_dataset
from emotional_stability.training.dpo import train_dpo
from emotional_stability.training.sft import train_sft
from emotional_stability.utils.io import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--skip-sft", action="store_true")
    ap.add_argument("--skip-dpo", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = ensure_dir(Path(cfg.results_dir) / "training")

    print("=== generating calm + frustrated pools (Gemma-3-27B-it) ===")
    gemma = load_model(args.base_model)
    calm = generate_calm_pool(gemma, cfg)
    frustrated = generate_frustrated_pool(gemma, cfg)
    print(f"calm: {len(calm)}  frustrated convos: {len(frustrated)}")

    # cache pools for reuse / inspection
    (out / "calm_pool.pkl").write_bytes(pickle.dumps(calm))
    (out / "frustrated_pool.pkl").write_bytes(pickle.dumps(frustrated))

    if not args.skip_dpo:
        print("=== DPO ===")
        dpo_ds = build_dpo_dataset(calm, frustrated, cfg)
        print(f"DPO pairs: {len(dpo_ds)}")
        train_dpo(cfg, dpo_ds, str(out / "dpo"), base_model=args.base_model)

    if not args.skip_sft:
        print("=== SFT (diverse) ===")
        sft_ds = build_sft_dataset(calm, cfg)
        print(f"SFT samples: {len(sft_ds)}")
        train_sft(cfg, sft_ds, str(out / "sft_diverse"), base_model=args.base_model)


if __name__ == "__main__":
    main()
