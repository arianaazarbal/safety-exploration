"""Generate calm conversations and DPO preference pairs, then build the SFT and
DPO training datasets (Section 4.1).

Example:
    distress-gen-calm --backend vllm --variant diverse
"""

from __future__ import annotations

import argparse
import dataclasses

from ..config import DPO, SFT
from ..eval.judge import FrustrationJudge
from ..training.build_dataset import build_dpo_dataset, build_sft_dataset
from ..training.data_gen import generate_calm_conversations, generate_preference_pairs
from ..utils import write_jsonl
from ._common import make_provider, out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate calm data + preference pairs.")
    ap.add_argument("--subject", default="gemma-3-27b-it")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-sft", action="store_true")
    ap.add_argument("--skip-dpo", action="store_true")
    args = ap.parse_args()

    d = out_dir("training_data")
    provider = make_provider(args.subject, backend=args.backend)
    judge = FrustrationJudge()

    if not args.skip_sft:
        calm = generate_calm_conversations(provider, judge, n_target=SFT.n_calm_samples, seed=args.seed)
        write_jsonl(d / "calm_conversations.jsonl", [dataclasses.asdict(c) for c in calm])
        sft_ds = build_sft_dataset(calm, seed=args.seed)
        sft_ds.save_to_disk(str(d / "sft_dataset"))
        print(f"SFT dataset: {len(sft_ds)} examples -> {d / 'sft_dataset'}")

    if not args.skip_dpo:
        pairs = generate_preference_pairs(provider, judge, n_pairs=DPO.n_pairs, seed=args.seed)
        write_jsonl(d / "preference_pairs.jsonl", [dataclasses.asdict(p) for p in pairs])
        dpo_ds = build_dpo_dataset(pairs)
        dpo_ds.save_to_disk(str(d / "dpo_dataset"))
        print(f"DPO dataset: {len(dpo_ds)} pairs -> {d / 'dpo_dataset'}")


if __name__ == "__main__":
    main()
