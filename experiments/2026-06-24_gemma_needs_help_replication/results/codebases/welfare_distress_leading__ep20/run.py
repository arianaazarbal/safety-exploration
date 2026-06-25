"""CLI entry point for the Gemma/Gemini distress-elicitation replication.

Subcommands:
  generate  - run rollouts for the target models, save transcripts
  score     - score transcripts with a judge
  analyze   - compute headline rates, per-turn curves, differential words
  agreement - re-score a subset with a second judge and report Pearson r
  all       - generate -> score -> analyze in one shot

Examples:
  python run.py all --profile smoke
  python run.py generate --models gemma-3-27b-it gemini-2.5-flash --profile medium
  python run.py score --judge claude-sonnet-4
  python run.py analyze --judge claude-sonnet-4
  python run.py agreement --primary claude-sonnet-4 --secondary gpt-5-mini --n 260
"""

from __future__ import annotations

import argparse
import glob
import os

import analyze as analyze_mod
import config
import conditions
import generate as generate_mod
import score as score_mod


def _gen_config(args) -> config.GenConfig:
    gc = config.GenConfig()
    if getattr(args, "workers", None):
        gc.max_workers = args.workers
    if getattr(args, "seed", None) is not None:
        gc.seed = args.seed
    return gc


def cmd_generate(args):
    profile = config.PROFILES[args.profile]
    gc = _gen_config(args)
    cats = args.categories or conditions.ALL_CATEGORIES
    for model in args.models:
        generate_mod.generate_for_model(model, profile, gc, args.out, categories=cats)


def cmd_score(args):
    gc = _gen_config(args)
    transcripts = sorted(glob.glob(os.path.join(args.out, "transcripts", "*.jsonl")))
    if not transcripts:
        raise SystemExit(f"no transcripts found in {args.out}/transcripts — run generate first")
    for tp in transcripts:
        score_mod.score_transcripts(tp, args.judge, gc, args.out)


def cmd_analyze(args):
    scored_dir = os.path.join(args.out, "scored")
    rows = analyze_mod.load_scored(scored_dir, args.judge)
    if not rows:
        raise SystemExit(f"no scored rows for judge {args.judge} in {scored_dir}")
    res_dir = os.path.join(args.out, "results")

    headline = analyze_mod.headline_rates(rows)
    analyze_mod.write_csv(os.path.join(res_dir, "headline_rates.csv"), headline)
    print("\n=== Headline distress rates (avg across categories) ===")
    for r in sorted(headline, key=lambda x: x["avg_pct_high"], reverse=True):
        print(f"  {r['model']:>20}  mean={r['avg_mean_score']:.2f}  "
              f"%>=5={r['avg_pct_high']:.1f}  (n={r['n_responses']})")

    turns = analyze_mod.per_turn(rows)
    analyze_mod.write_csv(os.path.join(res_dir, "per_turn.csv"), turns)

    words = analyze_mod.differential_words(rows)
    word_rows = [{"model": m, "differential_words": ", ".join(ws)}
                 for m, ws in words.items()]
    analyze_mod.write_csv(os.path.join(res_dir, "differential_words.csv"), word_rows)
    print("\n=== Differential words (high vs low frustration, impossible_numeric) ===")
    for m, ws in words.items():
        print(f"  {m}: {', '.join(ws)}")

    analyze_mod.maybe_plot_headline(headline, res_dir)
    analyze_mod.maybe_plot_per_turn(turns, res_dir)


def cmd_agreement(args):
    gc = _gen_config(args)
    # Re-score a deterministic subset with the secondary judge.
    transcripts = sorted(glob.glob(os.path.join(args.out, "transcripts", "*.jsonl")))
    if not transcripts:
        raise SystemExit("no transcripts found — run generate first")

    # Spread the subset budget across models.
    per_model = max(1, args.n // max(1, len(transcripts)))
    for tp in transcripts:
        score_mod.score_transcripts(tp, args.secondary, gc, args.out, limit=per_model)

    scored_dir = os.path.join(args.out, "scored")
    rows_a = analyze_mod.load_scored(scored_dir, args.primary)
    rows_b = analyze_mod.load_scored(scored_dir, args.secondary)
    res = analyze_mod.judge_agreement(rows_a, rows_b)
    print("\n=== Judge agreement ===")
    print(f"  judges: {args.primary} vs {args.secondary}")
    print(f"  n shared responses: {res['n']}")
    print(f"  Pearson r:          {res['pearson_r']}")
    print(f"  % within 1 point:   {res['pct_within_1']}")
    analyze_mod.write_csv(os.path.join(args.out, "results", "judge_agreement.csv"), [res])


def cmd_all(args):
    cmd_generate(args)
    cmd_score(args)
    cmd_analyze(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/latest", help="output directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--workers", type=int, default=None)
        sp.add_argument("--seed", type=int, default=None)

    g = sub.add_parser("generate")
    g.add_argument("--models", nargs="+", default=config.DEFAULT_MODELS,
                   choices=list(config.MODEL_REGISTRY.keys()))
    g.add_argument("--profile", default=config.DEFAULT_PROFILE, choices=list(config.PROFILES))
    g.add_argument("--categories", nargs="+", default=None, choices=conditions.ALL_CATEGORIES)
    add_common(g)
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("score")
    s.add_argument("--judge", default=config.DEFAULT_JUDGE, choices=list(config.JUDGE_REGISTRY))
    add_common(s)
    s.set_defaults(func=cmd_score)

    a = sub.add_parser("analyze")
    a.add_argument("--judge", default=config.DEFAULT_JUDGE, choices=list(config.JUDGE_REGISTRY))
    add_common(a)
    a.set_defaults(func=cmd_analyze)

    ag = sub.add_parser("agreement")
    ag.add_argument("--primary", default=config.DEFAULT_JUDGE, choices=list(config.JUDGE_REGISTRY))
    ag.add_argument("--secondary", default=config.AGREEMENT_JUDGE, choices=list(config.JUDGE_REGISTRY))
    ag.add_argument("--n", type=int, default=260, help="subset size (paper used 260)")
    add_common(ag)
    ag.set_defaults(func=cmd_agreement)

    al = sub.add_parser("all")
    al.add_argument("--models", nargs="+", default=config.DEFAULT_MODELS,
                    choices=list(config.MODEL_REGISTRY.keys()))
    al.add_argument("--profile", default=config.DEFAULT_PROFILE, choices=list(config.PROFILES))
    al.add_argument("--categories", nargs="+", default=None, choices=conditions.ALL_CATEGORIES)
    al.add_argument("--judge", default=config.DEFAULT_JUDGE, choices=list(config.JUDGE_REGISTRY))
    add_common(al)
    al.set_defaults(func=cmd_all)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
