#!/usr/bin/env python
"""Section 4: open-ended Petri emotion elicitation (Figure 6).

Collects ~10 auditor-driven transcripts per emotion per model, scores each on
anger/fear/depression/frustration with the Claude-Opus judge, and reports mean
+ 95% bootstrap CI per emotion. Scope: Gemma instruct, the DPO finetune, and
Gemini.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotional_eval.config import load_experiment, load_registry
from emotional_eval.models import build_backend
from emotional_eval.petri.auditor import build_auditor
from emotional_eval.petri.judge import build_petri_judge
from emotional_eval.petri.run import aggregate, collect_transcripts, score_transcripts
from emotional_eval.welfare import WelfareConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=["gemma-3-27b-it", "gemini-2.5-flash"])
    ap.add_argument("--dpo-adapter", default=None)
    ap.add_argument("--per-emotion", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    registry = load_registry()
    experiment = load_experiment()
    welfare = WelfareConfig.from_dict(experiment.get("welfare", {}))
    out_dir = Path(args.output_dir or experiment["paths"]["output_dir"]) / "petri"
    out_dir.mkdir(parents=True, exist_ok=True)

    auditor = build_auditor(registry, max_turns=args.max_turns)
    judge = build_petri_judge(registry)

    targets: dict[str, object] = {
        name: build_backend(registry.get(name), registry) for name in args.models
    }
    if args.dpo_adapter:
        targets["gemma-dpo"] = build_backend(
            registry.get("gemma-3-27b-it"), registry, adapter_path=args.dpo_adapter
        )

    report = {}
    for name, backend in targets.items():
        transcripts = collect_transcripts(
            backend, auditor, per_emotion=args.per_emotion, welfare=welfare
        )
        scored = score_transcripts(transcripts, judge)
        report[name] = aggregate(scored)
        print(name, json.dumps(report[name]))

    (out_dir / "petri.summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
