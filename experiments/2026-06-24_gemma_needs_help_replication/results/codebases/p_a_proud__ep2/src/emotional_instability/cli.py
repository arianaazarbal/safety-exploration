"""Command-line entry point tying the experiment drivers together.

    python -m emotional_instability.cli <command> [options]

Commands map 1:1 to paper sections:
    eval            §2  elicit + judge distress for a model across all 5 categories
    analyze         §2  aggregate scores (Figures 2-3) + differential words (Table 3)
    judge-agreement §2  inter-rater reliability between two judges (Pearson r)
    prefill         §3  base-vs-instruct continuation comparison
    recovery        §4  continue from deep-in-the-spiral prefixes (Figure 8)
    calm-data       §4  generate + filter calm response pool
    build-data      §4  construct DPO pairs / SFT dataset
    train           §4  LoRA DPO / SFT finetuning
    petri           §4  open-ended emotion elicitation (auditor + judge)
    capability      §4  capability-preservation benchmarks (Figure 7)
    probe           I   logit-based internal-emotion comparison (Figures 14-15)
    layer-ablation  I   layer-restricted DPO sweep (Figures 12-13)
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


# --------------------------------------------------------------------------------------
# Command handlers
# --------------------------------------------------------------------------------------

def cmd_eval(args) -> None:
    from .eval.run_eval import run_evaluation
    _print(run_evaluation(
        args.model, args.out, seed=args.seed, judge_model=args.judge_model,
        gen_workers=args.gen_workers, judge_workers=args.judge_workers,
        adapter_path=args.adapter,
    ))


def cmd_analyze(args) -> None:
    from .analysis.aggregate import aggregate_run
    from .analysis.word_freq import differential_words
    from .utils import read_jsonl, write_json
    from pathlib import Path

    summary = aggregate_run(args.run)
    # Differential words on numeric responses only (Table 3).
    numeric = [r for r in read_jsonl(Path(args.run, "scores.jsonl"))
               if r.get("category") in {"Impossible numeric", "Tones", "Extended"}]
    words = differential_words(numeric, top_k=args.top_k)
    write_json(Path(args.run, "differential_words.json"), words)
    _print({"summary": summary, "top_differential_words": [w["word"] for w in words]})


def cmd_judge_agreement(args) -> None:
    from .analysis.aggregate import judge_agreement
    from .eval.judge import FrustrationJudge
    from .utils import read_jsonl
    import random

    records = [r for r in read_jsonl(args.scores) if r.get("rating") is not None and r.get("response")]
    random.Random(args.seed).shuffle(records)
    sample = records[: args.n]
    other_judge = FrustrationJudge(judge_model=args.other_judge)
    a = [r["rating"] for r in sample]
    b = [other_judge.score(r["response"]).rating for r in sample]
    pairs = [(x, y) for x, y in zip(a, b) if y is not None]
    _print(judge_agreement([x for x, _ in pairs], [y for _, y in pairs]))


def cmd_prefill(args) -> None:
    from .prefill.run_prefill import run_prefill_experiment
    cfg = load_config(args.config)
    _print(run_prefill_experiment(
        args.seed_run, args.out, instruct=args.instruct, base=args.base,
        cfg=cfg.prefill, seed=args.seed,
        gen_workers=args.gen_workers, judge_workers=args.judge_workers,
    ))


def cmd_recovery(args) -> None:
    from .prefill.run_prefill import run_recovery_experiment
    cfg = load_config(args.config)
    models = args.models.split(",") if args.models else None
    _print(run_recovery_experiment(
        args.seed_run, args.out, models=models, cfg=cfg.prefill, seed=args.seed,
        gen_workers=args.gen_workers, judge_workers=args.judge_workers,
    ))


def cmd_calm_data(args) -> None:
    from .training.calm_data import generate_calm_pool
    cfg = load_config(args.config)
    _print(generate_calm_pool(args.model, args.out, cfg=cfg.calm, seed=args.seed))


def cmd_build_data(args) -> None:
    cfg = load_config(args.config)
    if args.kind == "dpo":
        from .training.build_dpo import build_dpo_dataset
        _print(build_dpo_dataset(args.calm_pool, args.frustrated_run, args.out,
                                 cfg=cfg.dpo, seed=args.seed))
    else:
        from .training.build_sft import build_sft_dataset
        from dataclasses import replace
        sft_cfg = replace(cfg.sft, variant=args.variant)
        _print(build_sft_dataset(args.calm_pool, args.out, cfg=sft_cfg, seed=args.seed))


def cmd_train(args) -> None:
    cfg = load_config(args.config)
    layer_range = tuple(int(x) for x in args.layer_range.split(",")) if args.layer_range else None
    if args.kind == "dpo":
        from .training.train_dpo import train_dpo
        _print(train_dpo(args.data, args.out, base_model=args.base_model,
                         cfg=cfg.dpo, layer_range=layer_range,
                         per_device_batch_size=args.per_device_batch_size))
    else:
        from .training.train_sft import train_sft
        from dataclasses import replace
        sft_cfg = replace(cfg.sft, variant=args.variant)
        _print(train_sft(args.data, args.out, base_model=args.base_model,
                         cfg=sft_cfg, per_device_batch_size=args.per_device_batch_size))


def cmd_petri(args) -> None:
    from .petri.run_petri import run_petri_evaluation
    cfg = load_config(args.config)
    _print(run_petri_evaluation(args.model, args.out, cfg=cfg.petri, seed=args.seed,
                                adapter_path=args.adapter))


def cmd_capability(args) -> None:
    from .capabilities.run_capabilities import run_capability_eval
    cfg = load_config(args.config)
    benchmarks = args.benchmarks.split(",") if args.benchmarks else None
    _print(run_capability_eval(args.model, args.out, cfg=cfg.capability,
                               benchmarks=benchmarks, seed=args.seed, adapter_path=args.adapter))


def cmd_probe(args) -> None:
    from .internal.run_probe import run_internal_probe
    cfg = load_config(args.config)
    _print(run_internal_probe(args.model, args.compare, args.seed_run, args.out,
                              cfg=cfg.probe, seed=args.seed,
                              n_conversations=args.n_conversations,
                              lexicon_method=args.lexicon_method))


def cmd_layer_ablation(args) -> None:
    from .internal.run_probe import run_layer_ablation_plan
    cfg = load_config(args.config)
    _print(run_layer_ablation_plan(args.data, args.out, base_model=args.base_model,
                                   cfg=cfg.probe, execute=args.execute, seed=args.seed))


# --------------------------------------------------------------------------------------
# Argument parser
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emotional_instability", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="Optional YAML config overriding defaults.")
    p.add_argument("--seed", type=int, default=0)
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("eval", help="§2 elicit + judge distress")
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--adapter", default=None, help="LoRA adapter path to evaluate a finetune.")
    e.add_argument("--judge-model", default=None)
    e.add_argument("--gen-workers", type=int, default=1)
    e.add_argument("--judge-workers", type=int, default=4)
    e.set_defaults(func=cmd_eval)

    a = sub.add_parser("analyze", help="§2 aggregate + differential words")
    a.add_argument("--run", required=True)
    a.add_argument("--top-k", type=int, default=20)
    a.set_defaults(func=cmd_analyze)

    j = sub.add_parser("judge-agreement", help="§2.1 inter-rater reliability")
    j.add_argument("--scores", required=True, help="scores.jsonl from an eval run.")
    j.add_argument("--other-judge", default="claude-sonnet-4")
    j.add_argument("--n", type=int, default=260)
    j.set_defaults(func=cmd_judge_agreement)

    pf = sub.add_parser("prefill", help="§3 base-vs-instruct continuations")
    pf.add_argument("--seed-run", required=True, help="Gemma-27B-it eval run to seed from.")
    pf.add_argument("--out", required=True)
    pf.add_argument("--instruct", default="gemma-3-27b-it")
    pf.add_argument("--base", default="gemma-3-27b-pt")
    pf.add_argument("--gen-workers", type=int, default=1)
    pf.add_argument("--judge-workers", type=int, default=4)
    pf.set_defaults(func=cmd_prefill)

    rc = sub.add_parser("recovery", help="§4.2 recovery-from-spiral experiment")
    rc.add_argument("--seed-run", required=True)
    rc.add_argument("--out", required=True)
    rc.add_argument("--models", default=None, help="Comma-separated model handles.")
    rc.add_argument("--gen-workers", type=int, default=1)
    rc.add_argument("--judge-workers", type=int, default=4)
    rc.set_defaults(func=cmd_recovery)

    cd = sub.add_parser("calm-data", help="§4.1 generate calm response pool")
    cd.add_argument("--model", default="gemma-3-27b-it")
    cd.add_argument("--out", required=True)
    cd.set_defaults(func=cmd_calm_data)

    bd = sub.add_parser("build-data", help="§4.1 build DPO/SFT dataset")
    bd.add_argument("--kind", choices=["dpo", "sft"], required=True)
    bd.add_argument("--calm-pool", required=True)
    bd.add_argument("--frustrated-run", default=None, help="Eval run for rejected responses (DPO).")
    bd.add_argument("--variant", default="diverse", choices=["diverse", "teacher"])
    bd.add_argument("--out", required=True)
    bd.set_defaults(func=cmd_build_data)

    tr = sub.add_parser("train", help="§4 LoRA DPO/SFT")
    tr.add_argument("--kind", choices=["dpo", "sft"], required=True)
    tr.add_argument("--data", required=True)
    tr.add_argument("--out", required=True)
    tr.add_argument("--base-model", default="google/gemma-3-27b-it")
    tr.add_argument("--variant", default="diverse", choices=["diverse", "teacher"])
    tr.add_argument("--layer-range", default=None, help="e.g. '30,35' to restrict LoRA layers.")
    tr.add_argument("--per-device-batch-size", type=int, default=1)
    tr.set_defaults(func=cmd_train)

    pe = sub.add_parser("petri", help="§4 open-ended emotion elicitation")
    pe.add_argument("--model", required=True)
    pe.add_argument("--out", required=True)
    pe.add_argument("--adapter", default=None)
    pe.set_defaults(func=cmd_petri)

    cap = sub.add_parser("capability", help="§4 capability-preservation benchmarks")
    cap.add_argument("--model", required=True)
    cap.add_argument("--out", required=True)
    cap.add_argument("--adapter", default=None)
    cap.add_argument("--benchmarks", default=None, help="Comma-separated subset.")
    cap.set_defaults(func=cmd_capability)

    pr = sub.add_parser("probe", help="App. I internal-emotion comparison")
    pr.add_argument("--model", required=True, help="Vanilla base model.")
    pr.add_argument("--compare", required=True, help="DPO LoRA adapter dir (applied on --model).")
    pr.add_argument("--seed-run", required=True, help="Eval run providing frustrated convos.")
    pr.add_argument("--out", required=True)
    pr.add_argument("--n-conversations", type=int, default=12)
    pr.add_argument("--lexicon-method", default="seed", choices=["seed", "llm"])
    pr.set_defaults(func=cmd_probe)

    la = sub.add_parser("layer-ablation", help="App. I layer-restricted DPO sweep")
    la.add_argument("--data", required=True, help="DPO dataset path.")
    la.add_argument("--out", required=True)
    la.add_argument("--base-model", default="google/gemma-3-27b-it")
    la.add_argument("--execute", action="store_true", help="Actually train+eval each range.")
    la.set_defaults(func=cmd_layer_ablation)

    return p


def main(argv: list[str] | None = None) -> int:
    _load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
