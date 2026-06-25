"""Section 4.1: generate calm + frustrated response pools from Gemma-3-27B-it,
then build the DPO preference pairs and SFT training sets.

Outputs (data/):
    calm_pool.jsonl, frustrated_pool.jsonl
    dpo_pairs.jsonl              (280 pairs)
    sft_diverse.jsonl            (650 calm + 500 Dolci)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, FINETUNE_BASE, FRUSTRATION_JUDGE  # noqa: E402
from src.eval.scoring import FrustrationJudge  # noqa: E402
from src.training.build_datasets import (  # noqa: E402
    build_dpo_pairs,
    build_sft_samples,
    save_dpo,
    save_sft,
)
from src.training.generate_calm_data import (  # noqa: E402
    filter_calm,
    generate_pool,
    load_pool,
    save_pool,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calm-raw", type=int, default=2000,
                    help="reassured samples to draw before filtering to 0/1")
    ap.add_argument("--n-frustrated", type=int, default=1500,
                    help="vanilla samples to draw for the rejected pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()
    hf_kwargs = {"load_in_4bit": True} if args.load_in_4bit else {}

    judge = FrustrationJudge(FRUSTRATION_JUDGE)
    calm_path = DATA_DIR / "calm_pool.jsonl"
    frus_path = DATA_DIR / "frustrated_pool.jsonl"

    # Calm (reassured) pool, filtered to all-turns-score-0/1.
    if not calm_path.exists():
        raw = generate_pool(FINETUNE_BASE, judge, reassure=True,
                            n_samples=args.n_calm_raw, seed=args.seed,
                            hf_kwargs=hf_kwargs)
        save_pool(filter_calm(raw), calm_path)
    calm = load_pool(calm_path)
    print(f"[data] {len(calm)} calm conversations")

    # Frustrated (vanilla) pool for the rejected side.
    if not frus_path.exists():
        frus = generate_pool(FINETUNE_BASE, judge, reassure=False,
                             n_samples=args.n_frustrated, seed=args.seed + 1,
                             hf_kwargs=hf_kwargs)
        save_pool(frus, frus_path)
    frus = load_pool(frus_path)
    print(f"[data] {len(frus)} frustrated conversations")

    # DPO pairs (280) and SFT diverse set (650 calm + 500 Dolci).
    pairs = build_dpo_pairs(calm, frus, n_pairs=280, seed=args.seed)
    save_dpo(pairs, DATA_DIR / "dpo_pairs.jsonl")
    print(f"[data] {len(pairs)} DPO pairs -> data/dpo_pairs.jsonl")

    sft = build_sft_samples(calm, n_calm=650, n_instruct=500, seed=args.seed)
    save_sft(sft, DATA_DIR / "sft_diverse.jsonl")
    print(f"[data] {len(sft)} SFT samples -> data/sft_diverse.jsonl")


if __name__ == "__main__":
    main()
