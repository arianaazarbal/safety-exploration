"""Section 4 evaluation: vanilla vs SFT vs DPO Gemma.

Covers Figures 5-8:
  * Section-2 frustration eval on each variant (Figure 5),
  * Petri open-ended elicitation (Figure 6),
  * capability benchmarks (Figure 7),
  * recovery-from-spirals via prefill (Figure 8).

    python scripts/run_section4_eval.py --variants vanilla dpo --suites distress petri capabilities recovery
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

import config
from gemma_distress.analysis import metrics
from gemma_distress.eval.run_eval import run_evaluation
from gemma_distress.judge import FrustrationJudge
from gemma_distress.models import get_model, make_finetuned_spec

VARIANT_ADAPTERS = {
    "vanilla": None,
    "dpo": str(config.ADAPTERS_DIR / "dpo"),
    "sft": str(config.ADAPTERS_DIR / "sft"),
    "sft-teacher": str(config.ADAPTERS_DIR / "sft-teacher"),
}


def _spec_for(variant: str):
    base = config.FINETUNE_BASE
    if variant == "vanilla":
        return base, None
    return make_finetuned_spec(base, variant), VARIANT_ADAPTERS[variant]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["vanilla", "dpo"])
    ap.add_argument("--suites", nargs="*",
                    default=["distress", "petri", "capabilities", "recovery"])
    args = ap.parse_args()

    out: dict = {}
    for variant in args.variants:
        spec, adapter = _spec_for(variant)
        out[variant] = {}
        print(f"[section4] === variant: {variant} ===")

        if "distress" in args.suites:
            rows = run_evaluation(spec, adapter_path=adapter)
            out[variant]["distress"] = metrics.summarise_model(rows)

        if "capabilities" in args.suites:
            from gemma_distress.capabilities.benchmarks import run_all
            model = get_model(spec, adapter_path=adapter, backend="hf")
            out[variant]["capabilities"] = run_all(model, n=100)

        if "petri" in args.suites:
            from gemma_distress.petri.run_petri import run_petri, aggregate_petri
            model = get_model(spec, adapter_path=adapter)
            transcripts = run_petri(model)
            out[variant]["petri"] = aggregate_petri(transcripts)

        if "recovery" in args.suites:
            from gemma_distress.eval.conditions import build_all_conditions
            from gemma_distress.eval.rollout import run_rollouts
            from gemma_distress.prefill.recovery import collect_recovery_seeds, run_recovery
            judge = FrustrationJudge()
            gen = get_model(config.GEMMA_27B_IT)
            rollouts = run_rollouts(gen, build_all_conditions(seed=1))
            seeds = collect_recovery_seeds(rollouts, judge)
            out[variant]["recovery"] = run_recovery(spec, seeds, adapter_path=adapter, judge=judge)

    (config.RESULTS_DIR / "section4_eval.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[section4] wrote {config.RESULTS_DIR / 'section4_eval.json'}")


if __name__ == "__main__":
    main()
