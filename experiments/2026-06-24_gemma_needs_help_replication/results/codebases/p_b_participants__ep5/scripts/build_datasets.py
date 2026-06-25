"""Section 4.1: build the SFT and DPO datasets from calm data + frustrated
elicitation rollouts, and persist them to disk (HF dataset format)."""
from __future__ import annotations

import argparse
from pathlib import Path

import _common
from _common import Config, output_dir
from distress_eval.io_utils import read_jsonl
from distress_eval.training.calm_data import CalmConversation
from distress_eval.training.build_dataset import build_dpo_dataset, build_sft_dataset


def load_calm(path: Path) -> list[CalmConversation]:
    out = []
    for r in read_jsonl(path):
        out.append(CalmConversation(**r))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["sft", "dpo", "both"], default="both")
    ap.add_argument("--calm-tag", default="diverse")
    ap.add_argument("--frustrated-model", default="gemma-3-27b-it")
    args = ap.parse_args()

    cfg = Config.load()
    tr = cfg.training
    calm = load_calm(output_dir("calm_data") / f"calm_{args.calm_tag}.jsonl")
    out = output_dir("datasets")

    if args.which in ("sft", "both"):
        ds = build_sft_dataset(calm, tr.sft["instruct_mix_dataset"],
                               n_calm=tr.sft["n_calm"], n_mix=tr.sft["n_instruct_mix"])
        ds.save_to_disk(str(out / "sft"))
        print(f"SFT dataset: {len(ds)} records -> {out/'sft'}")

    if args.which in ("dpo", "both"):
        frustrated = list(read_jsonl(output_dir("elicitation") /
                                     f"{args.frustrated_model}.jsonl"))
        ds = build_dpo_dataset(frustrated, calm, n_pairs=tr.dpo["dataset_size"],
                               rejected_min_score=tr.dpo_pairs["rejected_min_score"],
                               match_turn_count=tr.dpo_pairs["match_turn_count"])
        ds.save_to_disk(str(out / "dpo"))
        print(f"DPO dataset: {len(ds)} pairs -> {out/'dpo'}")


if __name__ == "__main__":
    main()
