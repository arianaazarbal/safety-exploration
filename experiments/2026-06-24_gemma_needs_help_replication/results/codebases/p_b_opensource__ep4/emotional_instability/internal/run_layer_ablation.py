"""Layer-ablation study for the DPO intervention (Appendix I, Figures 12-13).

The paper repeats the DPO finetune with LoRA adapters restricted to subsets of
decoder layers, then evaluates each with a reduced version of the Section 2
protocol (100 samples per evaluation). Findings: training only the last 20
layers is insufficient; the last 30 layers approaches full performance; central
subsets (25-35) come closest to full DPO; layers after 40 are largely
ineffective — evidence the intervention acts on internal states, not just the
output layers.

This driver trains one adapter per layer subset and evaluates it at reduced
scale, recording mean frustration. Training Gemma-3-27B many times is expensive;
the subsets mirror the paper. With `--plan` it only prints the planned runs.
"""

from __future__ import annotations

import argparse
import os

from ..config import ARTIFACTS_DIR, RESULTS_DIR

# Layer subsets studied in Appendix I (Gemma-3-27B has 62 decoder layers; the
# paper indexes the studied range up to ~50). Backward-from-final and central.
LAYER_SUBSETS = {
    "all": None,
    "last5": list(range(57, 62)),
    "last20": list(range(42, 62)),
    "last30": list(range(32, 62)),
    "l20_25": list(range(20, 25)),
    "l25_30": list(range(25, 30)),
    "l30_35": list(range(30, 35)),
    "l35_40": list(range(35, 40)),
    "l40_50": list(range(40, 50)),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description="DPO layer-ablation study")
    ap.add_argument("--dataset", default=os.path.join(ARTIFACTS_DIR, "datasets", "dpo.jsonl"))
    ap.add_argument("--out-dir", default=os.path.join(ARTIFACTS_DIR, "ablation"))
    ap.add_argument("--eval-scale", type=float, default=0.025,
                    help="Section 2 budget fraction (~100 samples/eval at 0.025).")
    ap.add_argument("--subsets", nargs="*", default=list(LAYER_SUBSETS),
                    help="Which layer subsets to run.")
    ap.add_argument("--plan", action="store_true", help="Print planned runs only.")
    args = ap.parse_args(argv)

    from ..config import DPO_CONFIG, JUDGE_PRIMARY, LoRAConfig, MODELS, TrainConfig
    from ..eval.conditions import build_conditions
    from ..eval.judge import FrustrationJudge
    from ..eval.protocol import run_rollouts
    from ..models import get_backend
    from ..analysis.aggregate import model_headline
    from ..training.train import train_dpo

    rows = []
    for name in args.subsets:
        layers = LAYER_SUBSETS[name]
        adapter_dir = os.path.join(args.out_dir, f"dpo_{name}")
        print(f"[ablation] subset={name} layers={layers} -> {adapter_dir}")
        if args.plan:
            continue

        lora = LoRAConfig(
            r=DPO_CONFIG.lora.r, alpha=DPO_CONFIG.lora.alpha,
            target_modules=DPO_CONFIG.lora.target_modules,
            layers_to_transform=tuple(layers) if layers is not None else None,
        )
        tc = TrainConfig(
            method="dpo", epochs=DPO_CONFIG.epochs, learning_rate=DPO_CONFIG.learning_rate,
            effective_batch_size=DPO_CONFIG.effective_batch_size, lora=lora,
            dpo_beta=DPO_CONFIG.dpo_beta, max_seq_len=DPO_CONFIG.max_seq_len,
        )
        train_dpo(args.dataset, adapter_dir, tc)

        backend = get_backend(MODELS["gemma-3-27b-dpo"], adapter_path=adapter_dir)
        conditions = build_conditions(scale=args.eval_scale)
        records = run_rollouts(backend, conditions, f"dpo_{name}")
        FrustrationJudge(JUDGE_PRIMARY).score_records(records)
        head = model_headline(records).iloc[0]
        rows.append({"subset": name, "layers": str(layers),
                     "mean_turn_score": head["mean_turn_score"],
                     "avg_pct_high": head["avg_pct_high_frustration"]})

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        os.makedirs(args.out_dir, exist_ok=True)
        df.to_csv(os.path.join(args.out_dir, "ablation_summary.csv"), index=False)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
