"""Command-line entrypoint for the replication.

Subcommands map to paper sections. Every command is resumable: re-running picks
up where a previous (possibly crashed) run stopped. Global flags:

  --config PATH        run config (default: config/default.yaml)
  --models PATH        model registry (default: config/models.yaml)
  --set k.sub=value    ad-hoc config override (repeatable)
  --backend B          local backend: vllm (default) | transformers
  --adapter DIR        LoRA adapter directory to attach to a Gemma target

Examples
--------
  gemma-distress eval all gemma-3-27b-it
  gemma-distress eval all gemini-2.5-flash
  gemma-distress eval all gemma-3-27b-it --adapter runs/training/dpo_adapter
  gemma-distress prefill seeds && gemma-distress prefill continue gemma-3-27b-pt
  gemma-distress train calm && gemma-distress train dpo
  gemma-distress petri run gemma-3-27b-it
  gemma-distress capabilities run gemma-3-27b-it
  gemma-distress probe run gemma-3-27b-it
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config, load_models
from .logging_utils import setup_logging
from .usage import GLOBAL_USAGE


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None)
    p.add_argument("--models", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="config override key.sub=value (repeatable)")
    p.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    p.add_argument("--adapter", default=None, help="LoRA adapter dir for the Gemma target")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gemma-distress", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="group", required=True)

    # eval
    ev = sub.add_parser("eval", help="Section 2: elicitation evaluation")
    evsub = ev.add_subparsers(dest="cmd", required=True)
    for cmd in ("generate", "score", "validate", "analyze", "all"):
        sp = evsub.add_parser(cmd)
        sp.add_argument("model")
        _common(sp)

    # prefill
    pf = sub.add_parser("prefill", help="Section 3: base/instruct prefill")
    pfsub = pf.add_subparsers(dest="cmd", required=True)
    sp = pfsub.add_parser("seeds"); _common(sp)
    sp = pfsub.add_parser("continue"); sp.add_argument("model"); _common(sp)
    sp = pfsub.add_parser("analyze"); sp.add_argument("models", nargs="+"); _common(sp)

    # train
    tr = sub.add_parser("train", help="Section 4: training interventions")
    trsub = tr.add_subparsers(dest="cmd", required=True)
    for cmd in ("calm", "build-sft", "build-dpo", "sft", "dpo", "ablate"):
        sp = trsub.add_parser(cmd); _common(sp)

    # recovery
    rc = sub.add_parser("recovery", help="Section 4.2: recovery limitation")
    rcsub = rc.add_subparsers(dest="cmd", required=True)
    sp = rcsub.add_parser("seeds"); _common(sp)
    sp = rcsub.add_parser("continue"); sp.add_argument("model"); _common(sp)
    sp = rcsub.add_parser("analyze"); sp.add_argument("models", nargs="+"); _common(sp)

    # petri
    pt = sub.add_parser("petri", help="Section 4.2: open-ended elicitation")
    ptsub = pt.add_subparsers(dest="cmd", required=True)
    sp = ptsub.add_parser("run"); sp.add_argument("model"); _common(sp)
    sp = ptsub.add_parser("analyze"); sp.add_argument("models", nargs="+"); _common(sp)

    # capabilities
    cp = sub.add_parser("capabilities", help="Section 4.2: capability preservation")
    cpsub = cp.add_subparsers(dest="cmd", required=True)
    sp = cpsub.add_parser("run"); sp.add_argument("model"); _common(sp)
    sp = cpsub.add_parser("analyze"); sp.add_argument("models", nargs="+"); _common(sp)

    # probe
    pb = sub.add_parser("probe", help="Appendix I: internal emotion probing")
    pbsub = pb.add_subparsers(dest="cmd", required=True)
    sp = pbsub.add_parser("run"); sp.add_argument("model"); _common(sp)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_cfg = load_config(args.config, overrides=args.overrides)
    models_cfg = load_models(args.models)

    out_root = Path(run_cfg.run.output_root)
    setup_logging(run_cfg.run.log_level, log_dir=out_root / "logs")
    GLOBAL_USAGE.path = out_root / "usage.json"

    group, cmd = args.group, args.cmd
    backend = getattr(args, "backend", "vllm")
    adapter = getattr(args, "adapter", None)

    if group == "eval":
        from .eval import runner, validation, analyze
        model = args.model
        if cmd in ("generate", "all"):
            runner.run_generation(model, run_cfg, models_cfg,
                                  prefer_local_backend=backend, adapter=adapter)
        if cmd in ("score", "all"):
            runner.run_scoring(model, run_cfg, models_cfg)
        if cmd == "validate":
            validation.run_validation(model, run_cfg, models_cfg)
        if cmd in ("analyze", "all"):
            print(analyze.summarise(model, run_cfg))

    elif group == "prefill":
        from .prefill import runner
        if cmd == "seeds":
            runner.build_seeds(run_cfg, models_cfg)
        elif cmd == "continue":
            runner.run_continuations(args.model, run_cfg, models_cfg, adapter=adapter)
        elif cmd == "analyze":
            print(runner.summarise(run_cfg, args.models))

    elif group == "train":
        from .training import generate_calm, build_datasets, train_sft, train_dpo, layer_ablation
        if cmd == "calm":
            generate_calm.run(run_cfg, models_cfg)
        elif cmd == "build-sft":
            build_datasets.build_sft(run_cfg)
        elif cmd == "build-dpo":
            build_datasets.build_dpo(run_cfg)
        elif cmd == "sft":
            train_sft.train(run_cfg, models_cfg)
        elif cmd == "dpo":
            train_dpo.train(run_cfg, models_cfg)
        elif cmd == "ablate":
            layer_ablation.run_ablations(run_cfg, models_cfg)

    elif group == "recovery":
        from .training import recovery
        if cmd == "seeds":
            recovery.build_recovery_seeds(run_cfg, models_cfg)
        elif cmd == "continue":
            recovery.run_continuations(args.model, run_cfg, models_cfg, adapter=adapter)
        elif cmd == "analyze":
            print(recovery.summarise(run_cfg, args.models))

    elif group == "petri":
        from .petri import runner
        if cmd == "run":
            runner.run(args.model, run_cfg, models_cfg, adapter=adapter)
        elif cmd == "analyze":
            print(runner.summarise(run_cfg, args.models))

    elif group == "capabilities":
        from .capabilities import runner
        if cmd == "run":
            runner.run(args.model, run_cfg, models_cfg, adapter=adapter)
        elif cmd == "analyze":
            print(runner.summarise(run_cfg, args.models))

    elif group == "probe":
        from .probing import runner
        if cmd == "run":
            runner.run(args.model, run_cfg, models_cfg, adapter=adapter)

    GLOBAL_USAGE.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
