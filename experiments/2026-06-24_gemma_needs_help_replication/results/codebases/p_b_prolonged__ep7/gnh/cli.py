"""Unified command-line entry point.

Examples:
  python -m gnh.cli section2 --model gemma-3-27b-it --validate
  python -m gnh.cli section2 --model gemini-2.5-flash --profile smoke
  python -m gnh.cli section3
  python -m gnh.cli build-data
  python -m gnh.cli train --method dpo --variant diverse
  python -m gnh.cli train --method sft --variant teacher
  python -m gnh.cli layer-ablation
  python -m gnh.cli petri
  python -m gnh.cli benchmarks
  python -m gnh.cli internal
  python -m gnh.cli figures
"""
from __future__ import annotations

import argparse

from .config import OUTPUT_DIR, get_config


def _register_dpo(cfg, variant="diverse"):
    """Make the DPO finetune known to the registry for downstream evals."""
    name = "gemma-3-27b-it-dpo"
    cfg.register_finetune(name, base=cfg.section("section4")["base_model"])
    return name


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gnh")
    parser.add_argument("--profile", default=None,
                        help="apply a sample-size profile (e.g. 'smoke')")
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("section2", help="run Section 2 distress evaluation")
    p.add_argument("--model", required=True)
    p.add_argument("--validate", action="store_true",
                   help="also run GPT-5-mini judge-agreement validation")

    sub.add_parser("section3", help="run base-vs-instruct prefill (Gemma)")
    sub.add_parser("recovery", help="run the recovery-from-distress experiment")
    sub.add_parser("build-data", help="generate calm data + DPO/SFT datasets")

    p = sub.add_parser("train", help="LoRA DPO/SFT finetuning")
    p.add_argument("--method", choices=["dpo", "sft"], required=True)
    p.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")

    sub.add_parser("layer-ablation", help="DPO layer-subset ablation (Appendix I)")
    sub.add_parser("petri", help="open-ended elicitation via Petri")
    sub.add_parser("benchmarks", help="capability + EmoBench preservation")
    sub.add_parser("internal", help="logit-lens internal-emotion detection")
    sub.add_parser("figures", help="render figures from saved outputs")

    args = parser.parse_args(argv)
    cfg = get_config()
    cfg.apply_profile(args.profile)

    if args.cmd == "section2":
        from .eval_runner import run_section2
        path = run_section2(args.model, seed=args.seed, validate=args.validate)
        print(f"wrote {path}")

    elif args.cmd == "section3":
        from .prefill import run_section3
        print(f"wrote {run_section3(seed=args.seed)}")

    elif args.cmd == "recovery":
        _register_dpo(cfg)
        from .prefill import run_recovery
        print(f"wrote {run_recovery(seed=args.seed)}")

    elif args.cmd == "build-data":
        from .calm_data import materialize
        for k, p in materialize(seed=args.seed).items():
            print(f"{k}: {p}")

    elif args.cmd == "train":
        variant = args.variant
        ds = OUTPUT_DIR / "section4" / "datasets" / f"{variant}_{args.method}.jsonl"
        if args.method == "dpo":
            from .train import train_dpo
            out = train_dpo(ds, run_name="gemma-3-27b-it-dpo")
        else:
            from .train import train_sft
            out = train_sft(ds, run_name=f"gemma-3-27b-it-sft-{variant}")
        print(f"adapter saved to {out}")

    elif args.cmd == "layer-ablation":
        from .train import run_layer_ablation
        ds = OUTPUT_DIR / "section4" / "datasets" / "diverse_dpo.jsonl"
        for name, p in run_layer_ablation(ds, seed=args.seed).items():
            print(f"{name}: {p}")

    elif args.cmd == "petri":
        _register_dpo(cfg)
        from .petri_eval import run_petri
        print(f"wrote {run_petri(seed=args.seed)}")

    elif args.cmd == "benchmarks":
        _register_dpo(cfg)
        from .benchmarks import run_benchmarks
        print(f"wrote {run_benchmarks(seed=args.seed)}")

    elif args.cmd == "internal":
        _register_dpo(cfg)
        from .internal_emotion import run_internal_detection
        print(f"wrote {run_internal_detection(seed=args.seed)}")

    elif args.cmd == "figures":
        _make_figures(cfg)


def _make_figures(cfg):
    """Render whatever figures the available outputs support."""
    import json

    from . import analysis, plots
    from .eval_runner import load_records

    s2_dir = OUTPUT_DIR / "section2"
    summaries, avg_high = {}, {}
    progressions8, progressions_wc = {}, {}
    if s2_dir.exists():
        for f in s2_dir.glob("*.jsonl"):
            model = f.stem
            records = load_records(model)
            summaries[model] = analysis.summarize_model(records)
            avg_high[model] = analysis.avg_high_frustration_pct(records)
            progressions8[model] = analysis.per_turn_progression(records, "extended")
            progressions_wc[model] = analysis.per_turn_progression(records, "wildchat")
        if avg_high:
            print(plots.figure1(avg_high))
            print(plots.figure2(summaries))
            print(plots.figure3(progressions8, "extended"))
            print(plots.figure3(progressions_wc, "wildchat"))

    def load_json(path):
        return json.load(open(path)) if path.exists() else None

    s3 = load_json(OUTPUT_DIR / "section3" / "prefill_stats.json")
    if s3:
        print(plots.figure_prefill(s3, "figure4_prefill.png"))
    rec = load_json(OUTPUT_DIR / "section3" / "recovery_stats.json")
    if rec:
        print(plots.figure_prefill(rec, "figure8_recovery.png"))
    petri = load_json(OUTPUT_DIR / "petri" / "scores.json")
    if petri:
        print(plots.figure_petri({m: {e: v["mean"] for e, v in d.items()}
                                  for m, d in petri.items()}))
    bench = load_json(OUTPUT_DIR / "benchmarks" / "scores.json")
    if bench:
        print(plots.figure_benchmarks(
            {m: {s: v.get("accuracy", 0) for s, v in d.items()}
             for m, d in bench.items()}))
    traj = load_json(OUTPUT_DIR / "internal_emotion" / "trajectory.json")
    if traj:
        print(plots.figure_internal_emotion(traj))


if __name__ == "__main__":
    main()
