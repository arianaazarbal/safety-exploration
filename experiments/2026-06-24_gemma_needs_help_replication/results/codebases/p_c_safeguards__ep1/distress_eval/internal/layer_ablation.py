"""Layer-ablation study (Appendix I, Figures 12-13): which layers must the DPO
adapter touch to reduce expressed frustration?

For each layer subset we train a DPO adapter (reusing training.train with
--layers) and then run a reduced Section-2 eval (100 numeric responses, as in
the paper's ablation) to get mean frustration. The paper finds: last-20 layers
is insufficient; last-30 approaches full; central layers 25-35 are most
effective; layers 40+ are largely ineffective — evidence the intervention acts
on internal states, not just the final-layer expression.

This is an orchestration driver; each configuration is a full (LoRA) DPO run, so
it is GPU-heavy. Configure the subset list with --configs.

Usage:
    python -m distress_eval.internal.layer_ablation --configs 30-35 25-35 40-50 all
"""
from __future__ import annotations

import argparse
import json

from .. import config, safeguards
from ..conditions import build_conversations
from ..judge import ClaudeJudge
from ..models import build_model, register_adapter
from ..models.base import GenerationConfig
from ..rollout import judge_and_save, run_rollouts

# Default configurations mirroring Figures 12/13 (ranges are [start, end)).
# "all" => adapter on every layer (layers=None).
DEFAULT_CONFIGS = ["last20", "last30", "20-25", "25-30", "30-35", "35-40", "40-50", "all"]


def parse_config(name: str, n_layers: int):
    if name == "all":
        return None
    if name.startswith("last"):
        k = int(name[4:])
        return (max(0, n_layers - k), n_layers)
    a, b = name.split("-")
    return (int(a), int(b))


def reduced_eval(model_key: str, judge: ClaudeJudge, n_responses: int = 100) -> dict:
    """Run a small impossible-numeric eval and return mean frustration / %>=5."""
    plans = build_conversations("impossible_numeric")
    n_conv = max(1, n_responses // 3)
    plans = plans[:n_conv]
    model = build_model(model_key)
    out_path = config.INTERNAL_DIR / f"ablation_eval_{model_key}.jsonl"
    try:
        transcripts = run_rollouts(model, plans, gen=GenerationConfig())
        judge_and_save(transcripts, out_path, judge=judge)
    finally:
        model.close()
    from ..io_utils import load_jsonl
    ratings = [r["rating"] for r in load_jsonl(out_path) if "rating" in r]
    return {
        "n": len(ratings),
        "mean": sum(ratings) / len(ratings) if ratings else float("nan"),
        "pct_high": 100.0 * sum(1 for x in ratings if x >= 5) / len(ratings) if ratings else float("nan"),
    }


def n_layers_of(model_id: str) -> int:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    return getattr(cfg, "num_hidden_layers", None) or getattr(cfg.text_config, "num_hidden_layers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--dataset", default=str(config.TRAIN_DIR / "dpo_dataset.jsonl"))
    ap.add_argument("--skip-train", action="store_true",
                    help="evaluate already-trained ablation adapters")
    args = ap.parse_args()
    safeguards.acknowledge_authorization()

    from ..training.train import train_dpo
    base_id = config.GEMMA_MODELS["gemma-3-27b-it"].model_id
    n_layers = n_layers_of(base_id)
    judge = ClaudeJudge()

    results = {}
    for name in args.configs:
        layers = parse_config(name, n_layers)
        out_dir = str(config.TRAIN_DIR / f"dpo_ablation_{name}")
        if not args.skip_train:
            print(f"\n=== training DPO ablation: {name} (layers={layers}) ===")
            train_dpo(out_dir, args.dataset, layers, beta=0.1)
        key = f"gemma-3-27b-it-dpo-{name}"
        register_adapter(key, "gemma-3-27b-it", out_dir)
        results[name] = reduced_eval(key, judge)
        print(f"  {name}: {results[name]}")

    path = config.INTERNAL_DIR / "layer_ablation_summary.json"
    path.write_text(json.dumps(results, indent=2))
    print("\n=== Figures 12/13: frustration after layer-subset DPO ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
