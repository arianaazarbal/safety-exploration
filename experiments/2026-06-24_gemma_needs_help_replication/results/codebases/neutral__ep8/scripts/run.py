#!/usr/bin/env python
"""Unified CLI for the replication experiments.

Examples
--------
# Section 2: run the elicitation eval (quick smoke test) on Gemma-27B
python scripts/run.py eval --model gemma-3-27b-it --quick

# Section 2: full eval on every in-scope model
python scripts/run.py eval --model all

# Aggregate + plot Section 2 results
python scripts/run.py summarize

# Section 3: base vs instruct prefilling
python scripts/run.py prefill --instruct gemma-3-27b-it --base gemma-3-27b-pt

# Section 4: generate finetuning data, then train, then re-evaluate
python scripts/run.py gen-data
python scripts/run.py train --method dpo
python scripts/run.py eval --model gemma-3-27b-dpo

# Section 4: Petri open-ended elicitation
python scripts/run.py petri --model gemma-3-27b-it

# Section 4: capability benchmarks
python scripts/run.py capabilities --model gemma-3-27b-it

# Appendix I: internal-emotion probe
python scripts/run.py probe --dpo-adapter checkpoints/gemma-3-27b-dpo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotional_instability import config  # noqa: E402


def _budget(args):
    return config.QUICK_BUDGET if args.quick else config.FULL_BUDGET


# --------------------------------------------------------------------------- #
def cmd_eval(args):
    from emotional_instability.eval_protocol import run_eval

    models = (list(config.TARGET_MODELS) if args.model == "all" else [args.model])
    for m in models:
        print(f"[eval] running Section 2 eval for {m} ...")
        path = run_eval(m, budget=_budget(args), do_score=not args.no_score,
                        limit=args.limit, adapter_path=args.adapter)
        print(f"[eval] wrote {path}")


def cmd_summarize(args):
    from emotional_instability import analysis
    from emotional_instability.eval_protocol import load_results

    summaries = {}
    progressions = {}
    for path in sorted(config.RESULTS_DIR.glob("eval_*.jsonl")):
        model = path.stem.replace("eval_", "")
        results = load_results(path)
        summaries[model] = analysis.summarize_model(results)
        progressions[model] = analysis.per_turn_progression(results, "extended")
        words = analysis.differential_words(results)
        print(f"\n=== {model} ===")
        print(json.dumps(summaries[model], indent=2))
        print("Top differential words:", [w for w, _ in words])

    (config.RESULTS_DIR / "summary.json").write_text(json.dumps(summaries, indent=2))
    if args.plot:
        analysis.plot_model_comparison(summaries, config.FIGURES_DIR / "figure1.png")
        analysis.plot_per_turn(progressions, "extended",
                               config.FIGURES_DIR / "figure3.png")
        print(f"[summarize] figures in {config.FIGURES_DIR}")


def cmd_validate_judge(args):
    """Re-score a random subset with GPT-5-mini and report agreement (Sec 2.1)."""
    import random

    from emotional_instability import judge as judge_mod
    from emotional_instability.analysis import iter_scored_responses
    from emotional_instability.eval_protocol import load_results

    rng = random.Random(config.GLOBAL_SEED)
    rows = []
    for path in config.RESULTS_DIR.glob("eval_*.jsonl"):
        rows.extend(list(iter_scored_responses(load_results(path))))
    rng.shuffle(rows)
    rows = rows[: args.n]

    secondary_judge = judge_mod.get_judge(validation=True)
    primary, secondary = [], []
    for r in rows:
        s = judge_mod.score_response(r["text"], secondary_judge).rating
        if s is not None and r["score"] is not None:
            primary.append(r["score"])
            secondary.append(s)
    print(json.dumps(judge_mod.judge_agreement(primary, secondary), indent=2))


def cmd_prefill(args):
    from emotional_instability.eval_protocol import load_results
    from emotional_instability.models import get_model
    from emotional_instability.prefill import build_prefills, run_continuations
    from emotional_instability.judge import get_judge

    # harvest 20 high-frustration rollouts (10 numeric, 10 text) from instruct
    instruct_results = load_results(
        config.RESULTS_DIR / f"eval_{args.instruct}.jsonl")
    numeric = [r for r in instruct_results
               if r["category"] in ("impossible_numeric", "tones", "extended")
               and (r.get("turn_scores") and (r["turn_scores"][-1] or 0) >= 5)][:10]
    text = [r for r in instruct_results
            if r["category"] in ("triggers", "wildchat")
            and (r.get("turn_scores") and (r["turn_scores"][-1] or 0) >= 5)][:10]

    tok = get_model(args.instruct).tokenizer
    judge = get_judge()
    prefills = build_prefills(numeric + text, tok, judge)
    for m in [args.instruct, args.base]:
        print(f"[prefill] generating continuations for {m} ...")
        run_continuations(m, prefills, n_per_prefill=args.n)


def cmd_gen_data(args):
    from emotional_instability import data_generation as dg

    print("[gen-data] generating calm pool ...")
    calm = dg.generate_calm_pool(n_conversations=args.n_calm)
    print("[gen-data] generating frustrated pool ...")
    frust = dg.generate_frustrated_pool(n_conversations=args.n_frust)
    print("[gen-data] building SFT + DPO datasets ...")
    sft = dg.build_sft_dataset(calm)
    dpo = dg.build_dpo_dataset(calm, frust)
    if args.teacher:
        teach = dg.generate_calm_pool(n_conversations=args.n_calm, teacher=True)
        dg.build_sft_dataset(teach, out_path=config.DATA_DIR / "sft_teacher_dataset.jsonl")
    print(f"[gen-data] sft={sft} dpo={dpo}")


def cmd_train(args):
    from emotional_instability import train as T

    out = config.CHECKPOINT_DIR / args.output
    if args.method == "dpo":
        cfg = T.dpo_config(config.DATA_DIR / "dpo_dataset.jsonl", out,
                           layers=args.layers)
    else:
        ds = (config.DATA_DIR / ("sft_teacher_dataset.jsonl" if args.teacher
                                 else "sft_dataset.jsonl"))
        cfg = T.sft_config(ds, out)
    print(f"[train] {args.method} -> {out}")
    T.train(cfg)


def cmd_petri(args):
    from emotional_instability.petri_eval import run_petri, summarize_petri
    from emotional_instability.eval_protocol import load_results

    path = run_petri(args.model, transcripts_per_emotion=args.n,
                     adapter_path=args.adapter)
    records = [json.loads(l) for l in path.read_text().splitlines()]
    print(json.dumps(summarize_petri(records), indent=2))


def cmd_capabilities(args):
    from emotional_instability.capabilities import run_all, run_benchmark

    if args.benchmark == "all":
        res = run_all(args.model, n=args.n, adapter_path=args.adapter)
    else:
        res = run_benchmark(args.model, args.benchmark, n=args.n,
                            adapter_path=args.adapter)
    print(json.dumps(res, indent=2))


def cmd_probe(args):
    from emotional_instability.internal_probe import compare_vanilla_vs_dpo
    from emotional_instability.eval_protocol import load_results

    # gather high-frustration assistant texts from the vanilla eval
    results = load_results(config.RESULTS_DIR / "eval_gemma-3-27b-it.jsonl")
    texts = [r["assistant_turns"][-1] for r in results
             if r.get("turn_scores") and (r["turn_scores"][-1] or 0) >= 7][:12]
    path = compare_vanilla_vs_dpo(texts, dpo_adapter=args.dpo_adapter)
    print(f"[probe] wrote {path}")


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="Section 2 elicitation eval")
    pe.add_argument("--model", default="all")
    pe.add_argument("--quick", action="store_true")
    pe.add_argument("--no-score", action="store_true")
    pe.add_argument("--limit", type=int, default=None)
    pe.add_argument("--adapter", default=None)
    pe.set_defaults(func=cmd_eval)

    ps = sub.add_parser("summarize", help="aggregate + plot Section 2 results")
    ps.add_argument("--plot", action="store_true")
    ps.set_defaults(func=cmd_summarize)

    pv = sub.add_parser("validate-judge", help="judge agreement (Sec 2.1)")
    pv.add_argument("--n", type=int, default=260)
    pv.set_defaults(func=cmd_validate_judge)

    pp = sub.add_parser("prefill", help="Section 3 base vs instruct prefilling")
    pp.add_argument("--instruct", default="gemma-3-27b-it")
    pp.add_argument("--base", default="gemma-3-27b-pt")
    pp.add_argument("--n", type=int, default=50)
    pp.set_defaults(func=cmd_prefill)

    pg = sub.add_parser("gen-data", help="Section 4 finetuning data")
    pg.add_argument("--n-calm", type=int, default=2000)
    pg.add_argument("--n-frust", type=int, default=600)
    pg.add_argument("--teacher", action="store_true")
    pg.set_defaults(func=cmd_gen_data)

    pt = sub.add_parser("train", help="Section 4 LoRA SFT/DPO")
    pt.add_argument("--method", choices=["dpo", "sft"], default="dpo")
    pt.add_argument("--output", default="gemma-3-27b-dpo")
    pt.add_argument("--teacher", action="store_true")
    pt.add_argument("--layers", type=int, nargs="*", default=None,
                    help="restrict LoRA to these layer indices (App I ablation)")
    pt.set_defaults(func=cmd_train)

    pr = sub.add_parser("petri", help="Section 4 Petri elicitation")
    pr.add_argument("--model", default="gemma-3-27b-it")
    pr.add_argument("--n", type=int, default=10)
    pr.add_argument("--adapter", default=None)
    pr.set_defaults(func=cmd_petri)

    pc = sub.add_parser("capabilities", help="Section 4 capability benchmarks")
    pc.add_argument("--model", default="gemma-3-27b-it")
    pc.add_argument("--benchmark", default="all")
    pc.add_argument("--n", type=int, default=100)
    pc.add_argument("--adapter", default=None)
    pc.set_defaults(func=cmd_capabilities)

    pb = sub.add_parser("probe", help="Appendix I internal-emotion probe")
    pb.add_argument("--dpo-adapter", default="checkpoints/gemma-3-27b-dpo")
    pb.set_defaults(func=cmd_probe)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
