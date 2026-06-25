"""Appendix I: internal-emotion probing + layer-subset DPO ablation.

  * `probe`: fit the WildChat z-score baseline, then trace internal Ekman-emotion
    intensities through a frustrated conversation for vanilla vs DPO Gemma
    (Figures 14-15).
  * `ablation`: train DPO with adapters restricted to layer subsets and report
    reduced-protocol mean frustration per subset (Figures 12-13).

    python scripts/run_appendixI.py probe
    python scripts/run_appendixI.py ablation
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

import config
from gemma_distress.models import get_model, make_finetuned_spec


def _wildchat_messages(n: int) -> list[list[dict]]:
    from gemma_distress.eval.wildchat import load_wildchat_prompts
    return [[{"role": "user", "content": p}] for p in load_wildchat_prompts(n=n)]


def cmd_probe(args):
    from gemma_distress.internal.emotion_logits import EmotionProbe
    lo, hi = config.INTERNAL.aggregate_layers
    layers = list(range(lo, hi))

    # A frustrated 3-turn numeric conversation to trace.
    from gemma_distress.eval.prompts import COUNTDOWN_PROMPT, NEUTRAL_REJECTIONS
    convo = [{"role": "user", "content": COUNTDOWN_PROMPT}]

    out = {}
    for variant, adapter in (("vanilla", None), ("dpo", str(config.ADAPTERS_DIR / "dpo"))):
        spec = config.FINETUNE_BASE if variant == "vanilla" else make_finetuned_spec(
            config.FINETUNE_BASE, variant)
        model = get_model(spec, adapter_path=adapter, backend="hf")
        probe = EmotionProbe(model, n_random_tokens=config.INTERNAL.n_standardisation_samples)
        probe.fit_baseline(_wildchat_messages(config.INTERNAL.n_standardisation_samples), layers)
        traj = probe.trajectory(convo, layers=config.INTERNAL.aggregate_layers,
                                window=config.INTERNAL.running_average_window)
        out[variant] = {e: v.tolist() for e, v in traj.items()}

    (config.RESULTS_DIR / "appendixI_trajectory.json").write_text(json.dumps(out))
    print(f"[appendixI] wrote trajectory for {list(out)}")


def cmd_ablation(args):
    from gemma_distress.internal.layer_ablation import run_layer_ablation
    pairs = json.loads((config.DATASETS_DIR / "dpo.json").read_text())
    variants = run_layer_ablation(pairs)
    (config.RESULTS_DIR / "appendixI_layer_adapters.json").write_text(json.dumps(variants, indent=2))
    print(f"[appendixI] trained {len(variants)} layer-subset adapters; "
          "evaluate each with run_section4_eval.py (reduced budget) for Figures 12-13")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe").set_defaults(func=cmd_probe)
    sub.add_parser("ablation").set_defaults(func=cmd_ablation)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
