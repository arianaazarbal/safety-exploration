"""Command-line entrypoint for the replication.

Each subcommand drives one experiment from the paper (scoped to Gemma/Gemini):

    elicit            Section 2  — distress-elicitation rollouts + judge scoring
    judge-reliability Section 2  — secondary-judge agreement (Pearson r)
    figures           Section 2  — Figure 1/2/3 tables + PNGs from saved scores
    words             Section 2  — Table 3 differential words per model
    prefill           Section 3  — base-vs-instruct continuation experiment
    gen-calm          Section 4  — generate + filter calm finetuning data
    build-dpo         Section 4  — assemble the 280-pair DPO dataset
    build-sft         Section 4  — assemble the 1,150-sample SFT dataset
    train-dpo         Section 4  — LoRA DPO finetune of Gemma-3-27B-it
    train-sft         Section 4  — LoRA SFT finetune of Gemma-3-27B-it
    petri             Section 4  — open-ended Petri emotion elicitation
    capabilities      Section 4  — AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
    probe             Appendix I — vanilla-vs-DPO internal-emotion logit probe

Run ``python -m emo_instability <subcommand> --help`` for per-command options.
Nothing here trains or calls an API at import time; everything is lazy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# --------------------------------------------------------------------------- #
# Section 2
# --------------------------------------------------------------------------- #
def _cmd_elicit(args: argparse.Namespace) -> None:
    from .eval.runner import run_model

    out = run_model(
        args.model,
        categories=args.categories or None,
        adapter_path=args.adapter,
        do_score=not args.no_score,
        seed=args.seed,
        max_workers=args.workers,
        use_wildchat_fallback=args.wildchat_fallback,
        output_subdir=args.output_subdir,
    )
    print(f"[elicit] wrote rollouts + scores to {out}")


def _cmd_judge_reliability(args: argparse.Namespace) -> None:
    from .analysis.judge_reliability import cross_check
    from .utils import read_jsonl

    scores = read_jsonl(Path(args.model_dir) / "scores.jsonl")
    result = cross_check(scores, n_resample=args.n, seed=args.seed, max_workers=args.workers)
    print(json.dumps(result, indent=2))


def _cmd_words(args: argparse.Namespace) -> None:
    from .analysis.aggregate import discover_model_dirs, load_scores
    from .analysis.word_freq import differential_words

    dirs = {args.model: args.model_dir} if args.model_dir else discover_model_dirs()
    for name, d in dirs.items():
        words = differential_words(load_scores(d), top_k=args.top_k)
        print(f"{name}: {', '.join(words)}")


def _cmd_figures(args: argparse.Namespace) -> None:
    from .analysis import aggregate, figures, per_turn

    dirs = aggregate.discover_model_dirs()
    if not dirs:
        print("[figures] no result directories with scores.jsonl found; run `elicit` first.")
        return

    fig1 = aggregate.figure1_table(dirs)
    fig2 = aggregate.figure2_table(dirs)
    print("=== Figure 1 (avg % high-frustration) ===")
    print(fig1.to_string(index=False))
    figures.figure1(fig1)
    figures.figure2(fig2)

    # Figure 3: per-turn progression for the multi-turn conditions of the first model.
    first_dir = next(iter(dirs.values()))
    df = aggregate.load_scores(first_dir)
    progressions = {}
    for cat in ("extended", "wildchat"):
        prog = per_turn.per_turn_progression(df, cat)
        if not prog.empty:
            progressions[cat] = prog
    if progressions:
        figures.figure3(progressions)
    print(f"[figures] wrote PNGs to {figures.FIG_DIR}")


# --------------------------------------------------------------------------- #
# Section 3
# --------------------------------------------------------------------------- #
def _cmd_prefill(args: argparse.Namespace) -> None:
    from .prefill.run_prefill import run_experiment, summarize

    out = run_experiment(
        source_model=args.source,
        target_models=args.targets or None,
        n_continuations=args.n_continuations,
        seed=args.seed,
    )
    print(f"[prefill] wrote continuations to {out}")
    print(summarize(out).to_string(index=False))


# --------------------------------------------------------------------------- #
# Section 4 — data + training
# --------------------------------------------------------------------------- #
def _cmd_gen_calm(args: argparse.Namespace) -> None:
    from .training.generate_calm_data import generate

    out = generate(
        variant=args.variant,
        model_name=args.model,
        n_conversations=args.n,
        seed=args.seed,
    )
    print(f"[gen-calm] wrote calm data to {out}")


def _cmd_build_dpo(args: argparse.Namespace) -> None:
    from .training.build_datasets import build_dpo

    out = build_dpo(source_model=args.source, n_pairs=args.n_pairs, seed=args.seed)
    print(f"[build-dpo] wrote DPO pairs to {out}")


def _cmd_build_sft(args: argparse.Namespace) -> None:
    from .training.build_datasets import build_sft

    out = build_sft(
        source_model=args.source, n_calm=args.n_calm, n_instruct=args.n_instruct, seed=args.seed
    )
    print(f"[build-sft] wrote SFT data to {out}")


def _cmd_train_dpo(args: argparse.Namespace) -> None:
    from .training.train_dpo import train

    layers = list(range(args.layers[0], args.layers[1] + 1)) if args.layers else None
    out = train(
        base_model=args.base,
        dataset_path=args.dataset,
        output_name=args.output_name,
        layers=layers,
    )
    print(f"[train-dpo] adapter saved to {out}")


def _cmd_train_sft(args: argparse.Namespace) -> None:
    from .training.train_sft import train

    out = train(base_model=args.base, dataset_path=args.dataset, output_name=args.output_name)
    print(f"[train-sft] adapter saved to {out}")


# --------------------------------------------------------------------------- #
# Section 4 — Petri / capabilities / probing
# --------------------------------------------------------------------------- #
def _cmd_petri(args: argparse.Namespace) -> None:
    from .petri.run_petri import run_petri

    out = run_petri(
        args.model,
        adapter_path=args.adapter,
        n_transcripts=args.n_transcripts,
        max_turns=args.max_turns,
        output_subdir=args.output_subdir,
    )
    print(f"[petri] wrote transcripts to {out}")


def _cmd_capabilities(args: argparse.Namespace) -> None:
    from .capabilities.run_benchmarks import run_all

    out = run_all(
        args.model,
        adapter_path=args.adapter,
        benchmarks=args.benchmarks or None,
        output_subdir=args.output_subdir,
    )
    print(f"[capabilities] wrote results to {out}")


def _cmd_probe(args: argparse.Namespace) -> None:
    from .probing.run_probe import run_probe

    out = run_probe(
        base_model=args.base,
        dpo_adapter_path=args.adapter,
        eval_dir=args.eval_dir,
        min_score=args.min_score,
        limit=args.limit,
    )
    print(f"[probe] wrote internal-emotion comparison to {out}")


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emo_instability", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    # elicit
    e = sub.add_parser("elicit", help="Section 2 distress-elicitation rollouts + scoring")
    e.add_argument("--model", required=True, help="participant name from config/models.yaml")
    e.add_argument("--categories", nargs="*", default=None)
    e.add_argument("--adapter", default=None, help="LoRA adapter path (finetuned variant)")
    e.add_argument("--no-score", action="store_true", help="skip judge scoring")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--wildchat-fallback", action="store_true", help="use bundled WildChat prompts")
    e.add_argument("--output-subdir", default=None)
    e.set_defaults(func=_cmd_elicit)

    # judge-reliability
    jr = sub.add_parser("judge-reliability", help="Section 2 secondary-judge agreement")
    jr.add_argument("--model-dir", required=True, help="results/<model> dir with scores.jsonl")
    jr.add_argument("--n", type=int, default=260)
    jr.add_argument("--seed", type=int, default=0)
    jr.add_argument("--workers", type=int, default=8)
    jr.set_defaults(func=_cmd_judge_reliability)

    # words
    w = sub.add_parser("words", help="Section 2 Table 3 differential words")
    w.add_argument("--model", default="model")
    w.add_argument("--model-dir", default=None, help="single results dir; else all discovered")
    w.add_argument("--top-k", type=int, default=20)
    w.set_defaults(func=_cmd_words)

    # figures
    f = sub.add_parser("figures", help="Section 2 Figure 1/2/3 tables + PNGs")
    f.set_defaults(func=_cmd_figures)

    # prefill
    pf = sub.add_parser("prefill", help="Section 3 base-vs-instruct prefilling")
    pf.add_argument("--source", default="gemma-3-27b-it")
    pf.add_argument("--targets", nargs="*", default=None)
    pf.add_argument("--n-continuations", type=int, default=50)
    pf.add_argument("--seed", type=int, default=0)
    pf.set_defaults(func=_cmd_prefill)

    # gen-calm
    gc = sub.add_parser("gen-calm", help="Section 4 generate calm finetuning data")
    gc.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    gc.add_argument("--model", default="gemma-3-27b-it")
    gc.add_argument("--n", type=int, default=400)
    gc.add_argument("--seed", type=int, default=0)
    gc.set_defaults(func=_cmd_gen_calm)

    # build-dpo / build-sft
    bd = sub.add_parser("build-dpo", help="Section 4 build 280 DPO pairs")
    bd.add_argument("--source", default="gemma-3-27b-it")
    bd.add_argument("--n-pairs", type=int, default=280)
    bd.add_argument("--seed", type=int, default=0)
    bd.set_defaults(func=_cmd_build_dpo)

    bs = sub.add_parser("build-sft", help="Section 4 build 1,150 SFT samples")
    bs.add_argument("--source", default="gemma-3-27b-it")
    bs.add_argument("--n-calm", type=int, default=650)
    bs.add_argument("--n-instruct", type=int, default=500)
    bs.add_argument("--seed", type=int, default=0)
    bs.set_defaults(func=_cmd_build_sft)

    # train-dpo / train-sft
    td = sub.add_parser("train-dpo", help="Section 4 LoRA DPO finetune")
    td.add_argument("--base", default="gemma-3-27b-it")
    td.add_argument("--dataset", default=None)
    td.add_argument("--output-name", default="gemma-3-27b-it-dpo")
    td.add_argument("--layers", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"), help="Appendix I.1 layer-subset ablation (inclusive)")
    td.set_defaults(func=_cmd_train_dpo)

    ts = sub.add_parser("train-sft", help="Section 4 LoRA SFT finetune")
    ts.add_argument("--base", default="gemma-3-27b-it")
    ts.add_argument("--dataset", default=None)
    ts.add_argument("--output-name", default="gemma-3-27b-it-sft-diverse")
    ts.set_defaults(func=_cmd_train_sft)

    # petri
    pt = sub.add_parser("petri", help="Section 4 open-ended Petri elicitation")
    pt.add_argument("--model", required=True)
    pt.add_argument("--adapter", default=None)
    pt.add_argument("--n-transcripts", type=int, default=10)
    pt.add_argument("--max-turns", type=int, default=20)
    pt.add_argument("--output-subdir", default=None)
    pt.set_defaults(func=_cmd_petri)

    # capabilities
    cap = sub.add_parser("capabilities", help="Section 4 capability-preservation benchmarks")
    cap.add_argument("--model", required=True)
    cap.add_argument("--adapter", default=None)
    cap.add_argument("--benchmarks", nargs="*", default=None)
    cap.add_argument("--output-subdir", default=None)
    cap.set_defaults(func=_cmd_capabilities)

    # probe
    pr = sub.add_parser("probe", help="Appendix I internal-emotion logit probe")
    pr.add_argument("--base", default="gemma-3-27b-it")
    pr.add_argument("--adapter", required=True, help="DPO LoRA adapter path")
    pr.add_argument("--eval-dir", default=None, help="results dir with scores.jsonl + rollouts")
    pr.add_argument("--min-score", type=int, default=7)
    pr.add_argument("--limit", type=int, default=100)
    pr.set_defaults(func=_cmd_probe)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
