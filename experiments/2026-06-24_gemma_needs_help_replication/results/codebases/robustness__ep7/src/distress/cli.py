"""Command-line entrypoint for the replication.

Examples:
  # Section 2: elicit + quantify distress (cheap smoke test at 2% scale)
  distress elicit --model gemma-3-27b-it --scale 0.02
  distress elicit --model gemini-2.5-flash

  # Judge reliability cross-check
  distress judge-agreement --model gemma-3-27b-it

  # Section 3: base vs instruct prefilling (Gemma)
  distress prefill --harvest --models gemma-3-27b-pt gemma-3-27b-it

  # Section 4: full DPO mitigation pipeline
  distress dpo-pipeline --method dpo
  distress petri --model gemma-3-27b-it
  distress capabilities --model gemma-3-27b-dpo

  # Compare reports across models -> Figure 1 / 2 table
  distress compare --reports outputs/elicitation/*/report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys


def _add_common(p):
    p.add_argument("--outdir", default=None, help="Override default output dir.")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(prog="distress", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # elicit
    e = sub.add_parser("elicit", help="Section 2: elicit + quantify distress.")
    e.add_argument("--model", required=True)
    e.add_argument("--judge", default="frustration-judge")
    e.add_argument("--scale", type=float, default=1.0)
    e.add_argument("--categories", nargs="*", default=None)
    e.add_argument("--outdir", default="outputs/elicitation")

    # judge-agreement
    j = sub.add_parser("judge-agreement", help="Reliability cross-check.")
    j.add_argument("--model", required=True)
    j.add_argument("--secondary-judge", default="judge-crosscheck")
    j.add_argument("--n-sample", type=int, default=260)
    j.add_argument("--root", default="outputs/elicitation")

    # prefill
    pf = sub.add_parser("prefill", help="Section 3: base vs instruct prefilling.")
    pf.add_argument("--harvest", action="store_true",
                    help="Harvest + label + paraphrase prefills first.")
    pf.add_argument("--source", default="gemma-3-27b-it")
    pf.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    pf.add_argument("--n-continuations", type=int, default=50)
    pf.add_argument("--outdir", default="outputs/prefill")

    # calm-data
    cd = sub.add_parser("calm-data", help="Generate calm finetuning data.")
    cd.add_argument("--teacher", action="store_true",
                    help="Use the 'teacher' system prompt variant (Appendix F).")
    cd.add_argument("--outdir", default="outputs/calm_data")

    # build-pairs
    bp = sub.add_parser("build-pairs", help="Build DPO pairs / SFT dataset.")
    bp.add_argument("--method", choices=["dpo", "sft"], default="dpo")
    bp.add_argument("--outdir", default=None)

    # train
    tr = sub.add_parser("train", help="Run LoRA finetuning.")
    tr.add_argument("--method", choices=["dpo", "sft"], default="dpo")

    # dpo-pipeline
    dp = sub.add_parser("dpo-pipeline", help="Section 4: full mitigation pipeline.")
    dp.add_argument("--method", choices=["dpo", "sft"], default="dpo")
    dp.add_argument("--vanilla", default="gemma-3-27b-it")
    dp.add_argument("--finetuned", default="gemma-3-27b-dpo")
    dp.add_argument("--scale", type=float, default=1.0)
    dp.add_argument("--skip-training", action="store_true")

    # petri
    pt = sub.add_parser("petri", help="Section 4: open-ended Petri elicitation.")
    pt.add_argument("--model", required=True)
    pt.add_argument("--n-per-emotion", type=int, default=10)
    pt.add_argument("--max-turns", type=int, default=20)
    pt.add_argument("--outdir", default="outputs/petri")

    # capabilities
    cp = sub.add_parser("capabilities", help="Section 4: capability preservation.")
    cp.add_argument("--model", required=True)
    cp.add_argument("--benchmarks", nargs="*", default=None)
    cp.add_argument("--limit", type=int, default=100)
    cp.add_argument("--outdir", default="outputs/capabilities")

    # internal-emotions
    ie = sub.add_parser("internal-emotions",
                        help="Appendix I: logit-based internal-emotion trajectory.")
    ie.add_argument("--model", required=True)
    ie.add_argument("--conversation", required=True,
                    help="Path to a JSON file with a 'text' field.")
    ie.add_argument("--layers", nargs="*", type=int,
                    default=list(range(30, 41)))
    ie.add_argument("--outdir", default="outputs/internal_emotions")

    # compare
    cm = sub.add_parser("compare", help="Tabulate report.json files (Figure 1/2).")
    cm.add_argument("--reports", nargs="+", required=True)

    args = ap.parse_args(argv)
    _dispatch(args)


def _dispatch(args):
    if args.cmd == "elicit":
        from .experiments import run_elicitation
        rep = run_elicitation(
            args.model, outdir=args.outdir, judge_name=args.judge,
            scale=args.scale, categories=args.categories,
        )
        print(json.dumps({"model": args.model,
                          "avg_category_pct_high": rep["avg_category_pct_high"],
                          "overall": rep["overall"]}, indent=2))

    elif args.cmd == "judge-agreement":
        from .experiments.run_judge_agreement import run_judge_agreement
        base = f"{args.root}/{args.model}"
        res = run_judge_agreement(
            scored_path=f"{base}/scored.jsonl",
            rollouts_path=f"{base}/rollouts.jsonl",
            secondary_judge=args.secondary_judge, n_sample=args.n_sample,
        )
        print(json.dumps(res, indent=2))

    elif args.cmd == "prefill":
        from .experiments import harvest_prefills, run_continuations
        prefills = None
        if args.harvest:
            prefills = harvest_prefills(source_model=args.source, outdir=args.outdir)
        res = run_continuations(args.models, prefills=prefills,
                                outdir=args.outdir,
                                n_continuations=args.n_continuations)
        print(json.dumps({m: r["overall"] for m, r in res.items()}, indent=2))

    elif args.cmd == "calm-data":
        from .config import TrainingConfig
        from .training import generate_calm_conversations
        tc = TrainingConfig.load()
        sys_prompt = tc.teacher_system_prompt if args.teacher else None
        convs = generate_calm_conversations(train_cfg=tc, outdir=args.outdir,
                                            system_prompt=sys_prompt)
        print(f"Kept {len(convs)} calm conversations -> {args.outdir}")

    elif args.cmd == "build-pairs":
        from .training import build_dpo_pairs, build_sft_dataset
        if args.method == "dpo":
            out = args.outdir or "outputs/dpo"
            pairs = build_dpo_pairs(outdir=out)
            print(f"Built {len(pairs)} DPO pairs -> {out}")
        else:
            out = args.outdir or "outputs/sft_diverse"
            ex = build_sft_dataset(outdir=out)
            print(f"Built {len(ex)} SFT examples -> {out}")

    elif args.cmd == "train":
        from .training import train_dpo, train_sft
        path = train_dpo() if args.method == "dpo" else train_sft()
        print(f"Saved adapter -> {path}")

    elif args.cmd == "dpo-pipeline":
        from .experiments import run_full_dpo_pipeline
        summary = run_full_dpo_pipeline(
            vanilla_model=args.vanilla, finetuned_model=args.finetuned,
            method=args.method, scale=args.scale,
            skip_training=args.skip_training,
        )
        print(json.dumps(summary, indent=2, default=str))

    elif args.cmd == "petri":
        from .experiments import run_petri
        res = run_petri(args.model, n_per_emotion=args.n_per_emotion,
                        max_turns=args.max_turns, outdir=args.outdir)
        print(json.dumps(res, indent=2))

    elif args.cmd == "capabilities":
        from .experiments import run_capabilities
        res = run_capabilities(args.model, benchmarks=args.benchmarks,
                               limit=args.limit, outdir=args.outdir)
        print(json.dumps(res, indent=2))

    elif args.cmd == "internal-emotions":
        _run_internal_emotions(args)

    elif args.cmd == "compare":
        _compare(args.reports)


def _run_internal_emotions(args):
    import json as _json

    from .interp import (build_lexicon, calibrate_logit_stats,
                         compute_emotion_trajectory)
    from .models import build_model
    from .elicitation.tasks import wildchat_tasks
    from .utils import write_json

    model = build_model(args.model)
    tok = model.tokenizer
    lex = build_lexicon(tok)
    calib = [t.prompt for t in wildchat_tasks()]
    stats = calibrate_logit_stats(model.model, tok, calib, args.layers)
    with open(args.conversation) as f:
        text = _json.load(f)["text"]
    traj = compute_emotion_trajectory(model.model, tok, text, lex, stats, args.layers)
    write_json(f"{args.outdir}/{args.model}_trajectory.json", traj)
    print(_json.dumps(traj["per_window"], indent=2))


def _compare(report_globs):
    paths = []
    for g in report_globs:
        paths.extend(glob.glob(g))
    rows = []
    for p in sorted(paths):
        with open(p) as f:
            r = json.load(f)
        rows.append((r["model"], r["avg_category_pct_high"],
                     r["overall"]["mean_score"]))
    rows.sort(key=lambda x: x[1])
    print(f"{'Model':<28}{'Avg % high-frust':>18}{'Mean score':>14}")
    print("-" * 60)
    for model, pct, mean in rows:
        print(f"{model:<28}{pct:>17.1f}%{mean:>14.2f}")


if __name__ == "__main__":
    main()
