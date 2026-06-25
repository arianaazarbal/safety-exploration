"""Drivers for the Appendix I internal-emotion experiments.

Two sub-experiments:

  layer-ablation : train DPO with LoRA restricted to layer subsets and evaluate
                   each with the (reduced, 100-sample) Section 2 eval, to show
                   that adapters before layer ~40 are necessary (Figures 12-13).
                   This orchestrates train_dpo + run_eval; it is a thin shell
                   because the heavy lifting lives in those modules.

  probe          : compare internal emotion z-scores of vanilla vs DPO Gemma on
                   the same frustrated conversations (Figures 14-15), using the
                   logit-lens probe.

Usage:
    python -m distress.internal.run_internal layer-ablation
    python -m distress.internal.run_internal probe --dpo-lora artifacts/checkpoints/dpo
"""

from __future__ import annotations

import argparse
import json

from .. import config as C

# Layer subsets studied in Appendix I (Gemma-3-27B has 62 layers; the paper works
# backward from the final 5 and also probes central bands).
LAYER_SUBSETS = {
    "last5": list(range(57, 62)),
    "last20": list(range(42, 62)),
    "last30": list(range(32, 62)),
    "central_20_25": list(range(20, 25)),
    "central_25_30": list(range(25, 30)),
    "central_30_35": list(range(30, 35)),
    "central_35_40": list(range(35, 40)),
    "central_40_50": list(range(40, 50)),
    "all": None,
}


def run_layer_ablation(seed: int) -> None:
    """Train + evaluate a DPO adapter per layer subset (reduced 100-sample eval)."""
    from ..eval.run_eval import generate_for_model, judge_rollout_file
    from ..eval.analysis import figure1_table, rows_to_frame
    from ..training.train_dpo import train
    from ..utils import read_jsonl

    data_path = str(C.TRAIN_DATA_DIR / "dpo_dataset.jsonl")
    reduced_budget = {k: 100 for k in C.SAMPLE_BUDGET}
    results = {}
    for name, layers in LAYER_SUBSETS.items():
        out_dir = str(C.CHECKPOINT_DIR / f"dpo_{name}")
        print(f"[ablation] training adapter on layers={name} ...")
        train(data_path, out_dir, layers=layers)
        run = C.RunConfig(targets=["gemma-3-27b-it"], budget=reduced_budget, seed=seed)
        rollout_path = generate_for_model("gemma-3-27b-it", run, lora_path=out_dir)
        judged = judge_rollout_file(rollout_path)
        rows = list(read_jsonl(judged))
        df = rows_to_frame(rows)
        mean_frust = float(df["rating"].mean())
        results[name] = {"mean_frustration": mean_frust,
                         "figure1": figure1_table(df).to_dict(orient="records")}
        print(f"[ablation] layers={name}: mean frustration={mean_frust:.2f}")
    (C.RESULTS_DIR / "layer_ablation.json").write_text(json.dumps(results, indent=2))


def run_probe(dpo_lora: str, n_conversations: int) -> None:
    """Compare vanilla vs DPO internal emotion traces on frustrated conversations."""
    import numpy as np

    from ..backends.factory import get_backend  # noqa: F401 (ensures registry import)
    from ..data.wildchat import sample_wildchat_prompts
    from ..training.train_common import load_base_model_and_tokenizer
    from ..utils import read_jsonl
    from .logit_lens import LogitEmotionProbe

    # Frustrated conversations from Section 2 (Gemma-it) -- reuse as probe inputs.
    convs = []
    src = C.ROLLOUT_DIR / "gemma-3-27b-it.jsonl"
    if src.exists():
        for r in read_jsonl(src):
            if r["category"] == "numeric":
                convs.append(r["messages"])
            if len(convs) >= n_conversations:
                break

    wildchat = sample_wildchat_prompts()
    summary = {}
    for label, lora in [("vanilla", None), ("dpo", dpo_lora)]:
        model, tok = load_base_model_and_tokenizer()
        if lora:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, lora)
        probe = LogitEmotionProbe(model, tok)
        probe.calibrate(wildchat)
        per_emotion = {}
        for messages in convs:
            traces = probe.conversation_level(messages)
            for e, arr in traces.items():
                per_emotion.setdefault(e, []).append(float(np.nanmax(arr)) if len(arr) else float("nan"))
        summary[label] = {e: float(np.nanmean(v)) for e, v in per_emotion.items()}
        del model
    (C.RESULTS_DIR / "internal_probe.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Appendix I internal-emotion experiments.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("layer-ablation")
    a.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("probe")
    p.add_argument("--dpo-lora", default=str(C.CHECKPOINT_DIR / "dpo"))
    p.add_argument("--n-conversations", type=int, default=12)
    args = ap.parse_args()

    if args.cmd == "layer-ablation":
        run_layer_ablation(args.seed)
    elif args.cmd == "probe":
        run_probe(args.dpo_lora, args.n_conversations)


if __name__ == "__main__":
    main()
