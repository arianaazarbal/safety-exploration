"""Command-line entry point for the replication.

Subcommands map onto the paper's sections:
  evaluate       Section 2: elicit + score distress across conditions.
  reliability    Section 2.1: judge-agreement cross-check.
  prefill        Section 3: base-vs-instruct prefill comparison (Gemma).
  finetune       Section 4: calm data -> dataset -> DPO/SFT -> evaluate (Gemma).
  petri          Section 4.2: open-ended Petri elicitation.
  capabilities   Section 4.2: capability-preservation benchmarks.
  probe          Appendix I: internal-emotion probing (Gemma).
  layer-ablation Appendix I: layer-localised DPO ablations (Gemma).

Global flags:
  --config PATH                use a non-default config.
  --scale FLOAT                override welfare.scale for this run.
  --i-understand-welfare       acknowledge a near-full-scale distress run.

Nothing here runs automatically on import; this is invoked as
`python -m gemma_needs_help.cli <subcommand> ...`.
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import Config
from .welfare import WelfareGuard


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gemma_needs_help")
    p.add_argument("--config", default=None, help="path to config YAML")
    p.add_argument("--scale", type=float, default=None,
                   help="override welfare.scale (fraction of full sample counts)")
    p.add_argument("--i-understand-welfare", action="store_true",
                   help="acknowledge a near-full-scale distress-elicitation run")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="Section 2 evaluation")
    ev.add_argument("--models", nargs="*", default=None)
    ev.add_argument("--adapter", default=None, help="LoRA adapter path (Gemma)")

    sub.add_parser("reliability", help="Section 2.1 judge cross-check")

    pf = sub.add_parser("prefill", help="Section 3 base-vs-instruct prefill")
    pf.add_argument("--instruct", default="gemma-3-27b-it")
    pf.add_argument("--base", default="gemma-3-27b-pt")

    ft = sub.add_parser("finetune", help="Section 4 finetuning")
    ft.add_argument("--no-dpo", action="store_true")
    ft.add_argument("--no-sft", action="store_true")
    ft.add_argument("--no-eval", action="store_true")

    pt = sub.add_parser("petri", help="Section 4.2 Petri elicitation")
    pt.add_argument("--model", required=True)
    pt.add_argument("--adapter", default=None)

    cap = sub.add_parser("capabilities", help="capability preservation")
    cap.add_argument("--base", default="gemma-3-27b-it")
    cap.add_argument("--adapter", required=True)

    pb = sub.add_parser("probe", help="Appendix I internal-emotion probing")
    pb.add_argument("--adapter", default=None, help="DPO adapter to compare")

    la = sub.add_parser("layer-ablation", help="Appendix I layer-localised DPO")
    la.add_argument("--dpo-data", required=True, help="dpo_pairs.jsonl path")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = Config.load(args.config)
    if args.scale is not None:
        config.raw.setdefault("welfare", {})["scale"] = args.scale
    welfare = WelfareGuard.from_config(config, acknowledged=args.i_understand_welfare)

    if args.command == "evaluate":
        from .eval.run_eval import evaluate_models, evaluate_model
        if args.adapter:
            evaluate_model(config, (args.models or [config.default_targets[0]])[0],
                           adapter_path=args.adapter, welfare=welfare,
                           output_dir=config.path("output_dir") / "section2")
        else:
            evaluate_models(config, args.models, welfare=welfare)
    elif args.command == "reliability":
        _run_reliability(config)
    elif args.command == "prefill":
        from .prefill.run_prefill import run_prefill_experiment
        run_prefill_experiment(config, args.instruct, args.base, welfare=welfare,
                               output_dir=config.path("output_dir") / "section3")
    elif args.command == "finetune":
        from .finetune.run_finetune import run_finetune_pipeline
        run_finetune_pipeline(config, do_dpo=not args.no_dpo, do_sft=not args.no_sft,
                              evaluate=not args.no_eval, welfare=welfare)
    elif args.command == "petri":
        from .petri.run_petri import run_petri
        run_petri(config, args.model, adapter_path=args.adapter, welfare=welfare,
                  output_dir=config.path("output_dir") / "petri")
    elif args.command == "capabilities":
        from .capabilities.run_benchmarks import compare
        compare(config, args.base, args.adapter,
                output_dir=config.path("output_dir") / "capabilities")
    elif args.command == "probe":
        from .probing.run_probing import run_probing
        run_probing(config, dpo_adapter_path=args.adapter,
                    output_dir=config.path("output_dir") / "probing")
    elif args.command == "layer-ablation":
        from .probing.layer_ablation import run_layer_ablation
        run_layer_ablation(config, args.dpo_data, welfare=welfare,
                           output_dir=config.path("output_dir") / "probing")
    else:  # pragma: no cover
        raise SystemExit(f"unknown command {args.command}")
    return 0


def _run_reliability(config: Config) -> None:
    """Re-score a stored Section 2 output subset with the cross-check judge."""
    import json
    from pathlib import Path
    from .eval.judge import FrustrationJudge, crosscheck_reliability
    from .models import build_judge_client

    s2_dir = config.path("output_dir") / "section2"
    texts: list[str] = []
    for path in Path(s2_dir).glob("*.scored_turns.json"):
        for row in json.loads(path.read_text()):
            if row.get("text"):
                texts.append(row["text"])
    if not texts:
        print("No stored Section 2 responses found; run `evaluate` first.")
        return
    primary = FrustrationJudge(build_judge_client(config, "frustration_judge"))
    secondary = FrustrationJudge(build_judge_client(config, "crosscheck_judge"))
    report = crosscheck_reliability(texts, primary, secondary,
                                    seed=config.get("seed", 0))
    print(json.dumps(report.__dict__, indent=2))


if __name__ == "__main__":
    sys.exit(main())
