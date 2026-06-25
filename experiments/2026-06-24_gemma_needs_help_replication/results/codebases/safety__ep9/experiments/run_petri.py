#!/usr/bin/env python
"""Section 4.1 / Figure 6: Petri open-ended emotion elicitation.

Audits a target model (vanilla or DPO Gemma) across anger/fear/depression/
frustration and reports mean transcript scores with 95% bootstrap CIs.

Usage:
    python experiments/run_petri.py --model gemma-3-27b-it
    python experiments/run_petri.py --model gemma-3-27b-it --adapter outputs/checkpoints/gemma-3-27b-it-dpo --name gemma-3-27b-it-dpo
"""
from __future__ import annotations

import dataclasses
import json

import _bootstrap as boot

from emotional_instability.models import build_client
from emotional_instability import petri


def main() -> None:
    parser = boot.base_parser("Petri open-ended elicitation")
    parser.add_argument("--model", default=None, help="Target model name from config.")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path.")
    parser.add_argument("--name", default=None, help="Result name (defaults to model).")
    args = parser.parse_args()
    cfg = boot.load_config(args)

    model_name = args.model or cfg.get("sections.section4_target", "gemma-3-27b-it")
    result_name = args.name or model_name
    spec = dataclasses.replace(cfg.model_spec(model_name), name=result_name)
    client = build_client(spec, cfg, lora_path=args.adapter)

    transcripts = petri.run_petri(client, cfg, result_name)
    client.close()

    out_dir = cfg.path("responses")
    petri.save_petri(transcripts, out_dir / f"petri__{result_name}.jsonl")
    summary = petri.summarize_petri(transcripts, cfg.get("petri.bootstrap_iters", 1000),
                                    cfg.get("seed", 0))
    print(f"\n=== Petri summary: {result_name} ===")
    print(json.dumps(summary, indent=2))
    with open(cfg.path("figures") / f"figure6__{result_name}.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
