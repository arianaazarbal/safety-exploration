#!/usr/bin/env python
"""Section 4: evaluate the finetuned models.

Re-runs the §2 elicitation protocol on vanilla / DPO / SFT Gemma, runs Petri
open-ended elicitation, capability benchmarks, and the recovery test.

    python scripts/run_section4_eval.py --parts elicit petri capabilities recovery
"""

import argparse
import json

from gnh.config import (
    ARTIFACT_DIR,
    DPO_GEMMA,
    GEMMA_27B_IT,
    RESULTS_DIR,
    SFT_GEMMA_DIVERSE,
)
from gnh.capabilities.run_benchmarks import run_benchmark
from gnh.evaluation.run_eval import evaluate_model
from gnh.petri.run_petri import run_petri
from gnh.prefill.run_recovery import run_recovery

DPO_ADAPTER = str(ARTIFACT_DIR / "dpo_adapter")
SFT_ADAPTER = str(ARTIFACT_DIR / "sft_diverse_adapter")
FINETUNE_BENCHMARKS = ["math", "aime", "bbh"]   # loaders wired up in scope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*",
                    default=["elicit", "petri", "capabilities", "recovery"])
    args = ap.parse_args()

    if "elicit" in args.parts:
        evaluate_model(GEMMA_27B_IT)
        evaluate_model(DPO_GEMMA, backend_kwargs={"adapter_path": DPO_ADAPTER})
        evaluate_model(SFT_GEMMA_DIVERSE, backend_kwargs={"adapter_path": SFT_ADAPTER})

    if "petri" in args.parts:
        print(json.dumps(run_petri(), indent=2))

    if "capabilities" in args.parts:
        for spec, kw in [
            (GEMMA_27B_IT, {}),
            (DPO_GEMMA, {"adapter_path": DPO_ADAPTER}),
            (SFT_GEMMA_DIVERSE, {"adapter_path": SFT_ADAPTER}),
        ]:
            for b in FINETUNE_BENCHMARKS:
                print(run_benchmark(spec, b, backend_kwargs=kw))

    if "recovery" in args.parts:
        seeds = RESULTS_DIR / "section2" / GEMMA_27B_IT.key / "rollouts.jsonl"
        print(json.dumps(run_recovery(seeds, [GEMMA_27B_IT, DPO_GEMMA]), indent=2))


if __name__ == "__main__":
    main()
