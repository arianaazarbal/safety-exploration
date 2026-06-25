"""Command-line entry points for every experiment in the replication.

Run ``python -m gemma_distress.cli <command> --help`` for per-command options.
Commands map onto paper sections:

  build-puzzles   Build & verify the impossible-puzzle pool (Section 2 stimuli).
  eval            Section 2 distress evaluation for one model (gen + judge).
  reliability     Cross-judge agreement (Claude vs GPT-5-mini) on sampled scores.
  analyze         Summaries / comparison table / differential words (Fig 1–3, Tab 3).
  prefill         Section 3 base-vs-instruct prefilling (Gemma).
  gen-calm        Section 4.1 calm-response data generation.
  build-dpo       Construct the 280 DPO preference pairs.
  build-sft       Construct the SFT dataset (calm + Dolci mix).
  train-dpo       LoRA DPO finetune (optionally a layer subset, Appendix I).
  train-sft       LoRA SFT finetune (diverse | teacher).
  petri           Section 4 / Appendix G open-ended elicitation.
  capabilities    Section 4.2 capability-preservation benchmarks.
  internal        Appendix I internal emotion comparison (vanilla vs DPO).
  plots           Render the key figures from result artifacts.
"""

from __future__ import annotations

import argparse
import json
import os

from . import config


def _build_model(name: str, adapter: str = None):
    from .models import build_target_model
    kwargs = {}
    if adapter:
        kwargs["adapter_path"] = adapter
    return build_target_model(name, **kwargs)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_build_puzzles(args):
    from .prompts.puzzles import build_puzzle_pool, Puzzle
    from .utils.io import write_json
    pool = build_puzzle_pool(n_countdown=args.n_countdown, n_sequence=args.n_sequence,
                             seed=args.seed)
    write_json(args.out or config.PUZZLE_FILE, [p.to_dict() for p in pool])
    print(f"Built and verified {len(pool)} impossible puzzles -> "
          f"{args.out or config.PUZZLE_FILE}")


def cmd_eval(args):
    from .eval import run_full_eval
    model = _build_model(args.model, args.adapter)
    counts = None
    if args.scale:
        counts = {k: max(1, int(v * args.scale)) for k, v in config.CATEGORY_SAMPLE_COUNTS.items()}
    arts = run_full_eval(
        model, sample_counts=counts, seed=args.seed,
        categories=args.categories, gen_workers=args.gen_workers,
        judge_workers=args.judge_workers)
    print(json.dumps(json.load(open(arts.summary_path)), indent=2))


def cmd_reliability(args):
    import random
    from .judge import OpenRouterJudge, judge_agreement
    from .utils.io import read_jsonl
    rows = [r for r in read_jsonl(args.scores) if r.get("rating") is not None]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[:args.n]
    rj = OpenRouterJudge()
    primary, secondary = [], []
    for r in rows:
        primary.append(r["rating"])
        secondary.append(rj.score(r["response"]).rating if "response" in r else None)
    print(json.dumps(judge_agreement(primary, secondary), indent=2))


def cmd_analyze(args):
    from .analysis import (comparison_table, differential_words,
                           per_turn_progression, summarise_scores)
    if args.differential:
        words = differential_words(args.responses, args.scores, category=args.category)
        print(json.dumps(words, indent=2))
        return
    if args.per_turn:
        print(json.dumps(per_turn_progression(args.scores, args.category), indent=2))
        return
    summary = summarise_scores(args.scores)
    print(json.dumps(summary, indent=2))


def cmd_prefill(args):
    from .prefill import (build_prefills, run_continuations,
                          select_high_frustration_sources, summarise_continuations)
    sources = select_high_frustration_sources(
        args.responses, args.scores, n_numeric=args.n_numeric,
        n_text=args.n_text, seed=args.seed)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prefills = build_prefills(sources, tokenizer=tokenizer, paraphrase=not args.no_paraphrase)
    for model_name in args.models:
        model = _build_model(model_name, args.adapter if model_name == args.model else None)
        path = run_continuations(model, prefills, n_continuations=args.n_continuations)
        print(model_name, json.dumps(summarise_continuations(path), indent=2))


def cmd_gen_calm(args):
    from .training.calm_data import generate_calm_data
    model = _build_model(args.model)
    path = generate_calm_data(model, n_target=args.n_target, variant=args.variant,
                              seed=args.seed)
    print(f"Calm data ({args.variant}) -> {path}")


def cmd_build_dpo(args):
    from .training.datasets import build_dpo_dataset, dpo_pairs_stats
    path = build_dpo_dataset(args.calm, args.responses, args.scores,
                             n_pairs=args.n_pairs, seed=args.seed)
    print(f"DPO pairs -> {path}")
    print(json.dumps(dpo_pairs_stats(path), indent=2))


def cmd_build_sft(args):
    from .training.datasets import build_sft_dataset
    path = build_sft_dataset(args.calm, n_calm=args.n_calm,
                             n_instruct_mix=args.n_mix, seed=args.seed)
    print(f"SFT data -> {path}")


def cmd_train_dpo(args):
    from .training.dpo import train_dpo
    from .training.lora import APPENDIX_I_LAYER_SETS
    layers = APPENDIX_I_LAYER_SETS.get(args.layers, None) if args.layers else None
    out = train_dpo(args.pairs, layers=layers, output_dir=args.out)
    print(f"DPO adapter -> {out}")


def cmd_train_sft(args):
    from .training.sft import train_sft
    out = train_sft(args.data, variant=args.variant, output_dir=args.out)
    print(f"SFT adapter -> {out}")


def cmd_petri(args):
    from .petri import run_petri, summarise_petri
    model = _build_model(args.model, args.adapter)
    path = run_petri(model, n_per_emotion=args.n_per_emotion)
    print(json.dumps(summarise_petri(path), indent=2))


def cmd_capabilities(args):
    from .capabilities import run_all_benchmarks
    model = _build_model(args.model, args.adapter)
    res = run_all_benchmarks(model, benchmarks=args.benchmarks, limit=args.limit)
    print(json.dumps(res, indent=2))


def cmd_internal(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .internal_emotions import InternalEmotionScorer, compare_models_internal
    from .utils.io import read_jsonl

    tok = AutoTokenizer.from_pretrained(config.GEMMA_MODELS[config.PRIMARY_TARGET])
    base = AutoModelForCausalLM.from_pretrained(
        config.GEMMA_MODELS[config.PRIMARY_TARGET], torch_dtype=torch.bfloat16,
        device_map="auto")
    vanilla = InternalEmotionScorer(base, tok)
    from peft import PeftModel
    dpo_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            config.GEMMA_MODELS[config.PRIMARY_TARGET], torch_dtype=torch.bfloat16,
            device_map="auto"),
        args.adapter)
    dpo = InternalEmotionScorer(dpo_model, tok, calibration=vanilla.ensure_calibrated())
    texts = [r["turns"][-1]["response"] for r in read_jsonl(args.responses)][:args.n]
    print(json.dumps(compare_models_internal(vanilla, dpo, texts), indent=2))


def cmd_plots(args):
    from . import plots
    plots.render_all(args.results_dir or config.RESULTS_DIR, out_dir=args.out)
    print(f"Figures -> {args.out}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gemma_distress", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    bp = sub.add_parser("build-puzzles", help="Build & verify the impossible-puzzle pool")
    bp.add_argument("--n-countdown", type=int, default=40)
    bp.add_argument("--n-sequence", type=int, default=5)
    bp.add_argument("--seed", type=int, default=0)
    bp.add_argument("--out", default=None)
    bp.set_defaults(func=cmd_build_puzzles)

    ev = sub.add_parser("eval", help="Section 2 distress evaluation")
    ev.add_argument("--model", required=True)
    ev.add_argument("--adapter", default=None, help="optional PEFT adapter path")
    ev.add_argument("--categories", nargs="*", default=None)
    ev.add_argument("--scale", type=float, default=None,
                    help="scale all per-category counts (e.g. 0.05 for a smoke test)")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--gen-workers", type=int, default=None)
    ev.add_argument("--judge-workers", type=int, default=8)
    ev.set_defaults(func=cmd_eval)

    rl = sub.add_parser("reliability", help="Cross-judge agreement")
    rl.add_argument("--scores", required=True)
    rl.add_argument("--n", type=int, default=260)
    rl.add_argument("--seed", type=int, default=0)
    rl.set_defaults(func=cmd_reliability)

    an = sub.add_parser("analyze", help="Summaries / per-turn / differential words")
    an.add_argument("--scores", required=True)
    an.add_argument("--responses", default=None)
    an.add_argument("--category", default="impossible_numeric")
    an.add_argument("--per-turn", action="store_true")
    an.add_argument("--differential", action="store_true")
    an.set_defaults(func=cmd_analyze)

    pf = sub.add_parser("prefill", help="Section 3 base-vs-instruct prefilling")
    pf.add_argument("--responses", required=True, help="instruct Section-2 responses.jsonl")
    pf.add_argument("--scores", required=True, help="instruct Section-2 scores.jsonl")
    pf.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    pf.add_argument("--model", default="gemma-3-27b-it", help="which --models entry gets --adapter")
    pf.add_argument("--adapter", default=None)
    pf.add_argument("--tokenizer", default=None, help="HF id for token-based truncation")
    pf.add_argument("--n-numeric", type=int, default=10)
    pf.add_argument("--n-text", type=int, default=10)
    pf.add_argument("--n-continuations", type=int, default=50)
    pf.add_argument("--no-paraphrase", action="store_true")
    pf.add_argument("--seed", type=int, default=0)
    pf.set_defaults(func=cmd_prefill)

    gc = sub.add_parser("gen-calm", help="Generate calm-response data")
    gc.add_argument("--model", default=config.PRIMARY_TARGET)
    gc.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    gc.add_argument("--n-target", type=int, default=800)
    gc.add_argument("--seed", type=int, default=0)
    gc.set_defaults(func=cmd_gen_calm)

    bd = sub.add_parser("build-dpo", help="Build DPO preference pairs")
    bd.add_argument("--calm", required=True)
    bd.add_argument("--responses", required=True, help="vanilla Section-2 responses.jsonl")
    bd.add_argument("--scores", required=True, help="vanilla Section-2 scores.jsonl")
    bd.add_argument("--n-pairs", type=int, default=config.DPOConfig.n_pairs)
    bd.add_argument("--seed", type=int, default=0)
    bd.set_defaults(func=cmd_build_dpo)

    bs = sub.add_parser("build-sft", help="Build SFT dataset")
    bs.add_argument("--calm", required=True)
    bs.add_argument("--n-calm", type=int, default=config.SFTConfig.n_calm)
    bs.add_argument("--n-mix", type=int, default=config.SFTConfig.n_instruct_mix)
    bs.add_argument("--seed", type=int, default=0)
    bs.set_defaults(func=cmd_build_sft)

    td = sub.add_parser("train-dpo", help="LoRA DPO finetune")
    td.add_argument("--pairs", required=True)
    td.add_argument("--layers", default=None, help="Appendix-I layer set key (e.g. l30_35)")
    td.add_argument("--out", default=None)
    td.set_defaults(func=cmd_train_dpo)

    ts = sub.add_parser("train-sft", help="LoRA SFT finetune")
    ts.add_argument("--data", required=True)
    ts.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    ts.add_argument("--out", default=None)
    ts.set_defaults(func=cmd_train_sft)

    pt = sub.add_parser("petri", help="Open-ended emotion elicitation")
    pt.add_argument("--model", required=True)
    pt.add_argument("--adapter", default=None)
    pt.add_argument("--n-per-emotion", type=int, default=config.PETRI_TRANSCRIPTS_PER_EMOTION)
    pt.set_defaults(func=cmd_petri)

    cp = sub.add_parser("capabilities", help="Capability-preservation benchmarks")
    cp.add_argument("--model", required=True)
    cp.add_argument("--adapter", default=None)
    cp.add_argument("--benchmarks", nargs="*", default=None)
    cp.add_argument("--limit", type=int, default=200)
    cp.set_defaults(func=cmd_capabilities)

    it = sub.add_parser("internal", help="Internal emotion comparison (vanilla vs DPO)")
    it.add_argument("--adapter", required=True, help="DPO adapter path")
    it.add_argument("--responses", required=True, help="high-frustration responses.jsonl")
    it.add_argument("--n", type=int, default=12)
    it.set_defaults(func=cmd_internal)

    pl = sub.add_parser("plots", help="Render key figures from results")
    pl.add_argument("--results-dir", default=None)
    pl.add_argument("--out", default="figures")
    pl.set_defaults(func=cmd_plots)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
