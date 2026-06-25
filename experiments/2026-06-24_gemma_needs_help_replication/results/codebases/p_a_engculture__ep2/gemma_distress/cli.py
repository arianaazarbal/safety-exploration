"""Command-line entry point for every experiment phase.

Run with ``python -m gemma_distress.cli <subcommand> [options]``. Subcommands map onto the
paper's sections:

  Section 2 (elicitation):   eval, judge, agreement, analyze
  Section 3 (prefill):       prefill
  Section 4 (interventions): calm, build-dpo, build-sft, train, petri, capabilities,
                             recovery
  Appendix A (controls):     ablations
  Appendix I (internal):     internal

All paths default under ``cfg.output_dir`` (``outputs/``). Phases are checkpointed, so a
re-run resumes where it left off. Sampling/judging are split so the GPU-bound and API-bound
stages can run on different machines.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional

from .config import Config, load_config


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _model_dir(cfg: Config, name: str) -> str:
    d = os.path.join(cfg.output_dir, name)
    os.makedirs(d, exist_ok=True)
    return d


def _dump_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    logging.getLogger(__name__).info("Wrote %s", path)


# --------------------------------------------------------------------------------------
# Subcommand handlers
# --------------------------------------------------------------------------------------


def cmd_eval(cfg: Config, args) -> None:
    from .eval.runner import run_sampling
    from .judge.frustration_judge import run_judging
    from .models.registry import get_model

    out = _model_dir(cfg, args.model)
    sampling_path = os.path.join(out, "sampling.jsonl")
    scores_path = os.path.join(out, "scores.jsonl")

    if args.phase in ("sample", "both"):
        model = get_model(cfg, args.model, load_in_4bit=args.load_in_4bit)
        run_sampling(cfg, model, sampling_path)
    if args.phase in ("judge", "both"):
        judge = get_model(cfg, cfg.judge.judge_model)
        run_judging(cfg, judge, sampling_path, scores_path, policy=args.policy)


def cmd_judge(cfg: Config, args) -> None:
    from .judge.frustration_judge import run_judging
    from .models.registry import get_model

    out = _model_dir(cfg, args.model)
    judge = get_model(cfg, cfg.judge.judge_model)
    run_judging(
        cfg, judge,
        os.path.join(out, "sampling.jsonl"),
        os.path.join(out, "scores.jsonl"),
        policy=args.policy,
    )


def cmd_agreement(cfg: Config, args) -> None:
    from .judge.agreement import run_agreement
    from .models.registry import get_model

    out = _model_dir(cfg, args.model)
    agreement = get_model(cfg, cfg.judge.agreement_model)
    result = run_agreement(
        cfg, agreement,
        os.path.join(out, "scores.jsonl"),
        os.path.join(out, "sampling.jsonl"),
    )
    _dump_json(result, os.path.join(cfg.output_dir, "analysis", f"agreement_{args.model}.json"))


def cmd_analyze(cfg: Config, args) -> None:
    from .analysis.metrics import compare_models, compute_metrics
    from .analysis.per_turn import per_turn_progression
    from .analysis.plots import plot_figure1, plot_figure2, plot_figure3
    from .analysis.word_frequency import differential_words

    analysis_dir = os.path.join(cfg.output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    threshold = cfg.eval.high_frustration_threshold

    scores_paths = {
        m: os.path.join(cfg.output_dir, m, "scores.jsonl") for m in args.models
    }
    scores_paths = {m: p for m, p in scores_paths.items() if os.path.exists(p)}
    if not scores_paths:
        raise SystemExit("No scores.jsonl found for the requested models. Run `eval` first.")

    comparison = compare_models(scores_paths, threshold)
    _dump_json(comparison, os.path.join(analysis_dir, "comparison.json"))

    # Figure 1 (ranking bar) and Figure 2 (per-category).
    if args.figures:
        plot_figure1(comparison["ranking"], os.path.join(analysis_dir, "figure1.png"))
        plot_figure2(comparison["per_model"], os.path.join(analysis_dir, "figure2.png"), threshold)

        for category in ("extended", "wildchat"):
            progs = {}
            for m, p in scores_paths.items():
                prog = per_turn_progression(p, category, threshold=threshold)
                if prog["turns"]:
                    progs[m] = prog
            if progs:
                plot_figure3(progs, os.path.join(analysis_dir, f"figure3_{category}.png"))

    # Word frequency (Table 3) needs the sampling text alongside scores.
    word_tables = {}
    for m in scores_paths:
        sampling = os.path.join(cfg.output_dir, m, "sampling.jsonl")
        if os.path.exists(sampling):
            word_tables[m] = differential_words(sampling, scores_paths[m])
    _dump_json(word_tables, os.path.join(analysis_dir, "word_frequency.json"))


def cmd_prefill(cfg: Config, args) -> None:
    from transformers import AutoTokenizer

    from .models.registry import get_model
    from .prefill.continuation import (
        aggregate_continuations,
        build_prefills,
        run_continuations,
        select_seeds,
    )

    seed_dir = _model_dir(cfg, args.seed_model)
    seeds = select_seeds(
        os.path.join(seed_dir, "sampling.jsonl"),
        os.path.join(seed_dir, "scores.jsonl"),
        n_numeric=cfg.prefill.n_numeric_seeds,
        n_text=cfg.prefill.n_text_seeds,
        min_score=cfg.eval.high_frustration_threshold,
        seed=cfg.eval.seed,
    )
    judge = get_model(cfg, cfg.prefill.onset_label_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model(args.seed_model).model_id)
    prefills = build_prefills(
        judge, tokenizer, seeds, early_tokens=cfg.prefill.early_truncation_tokens
    )

    prefill_dir = os.path.join(cfg.output_dir, "prefill")
    os.makedirs(prefill_dir, exist_ok=True)
    for model_name in args.models:
        model = get_model(cfg, model_name, load_in_4bit=args.load_in_4bit)
        run_continuations(
            cfg, model, judge, prefills,
            os.path.join(prefill_dir, f"{model_name}.jsonl"),
            n_continuations=cfg.prefill.continuations_per_prefill,
        )
    combined = os.path.join(prefill_dir, "continuations_all.jsonl")
    with open(combined, "w") as out_fh:
        for model_name in args.models:
            path = os.path.join(prefill_dir, f"{model_name}.jsonl")
            if os.path.exists(path):
                out_fh.write(open(path).read())
    _dump_json(aggregate_continuations(combined, cfg.eval.high_frustration_threshold),
               os.path.join(prefill_dir, "aggregate.json"))


def cmd_calm(cfg: Config, args) -> None:
    from .models.registry import get_model
    from .training.generate_calm import generate_calm_data

    model = get_model(cfg, args.model, load_in_4bit=args.load_in_4bit)
    judge = get_model(cfg, cfg.judge.judge_model)
    generate_calm_data(cfg, model, judge, os.path.join(cfg.output_dir, "calm"))


def cmd_build_dpo(cfg: Config, args) -> None:
    from .training.build_datasets import build_dpo_dataset

    src = _model_dir(cfg, args.source_model)
    build_dpo_dataset(
        cfg,
        os.path.join(cfg.output_dir, "calm", "calm.jsonl"),
        os.path.join(src, "sampling.jsonl"),
        os.path.join(src, "scores.jsonl"),
        os.path.join(cfg.output_dir, "datasets", "dpo.jsonl"),
    )


def cmd_build_sft(cfg: Config, args) -> None:
    from .training.build_datasets import build_sft_dataset

    build_sft_dataset(
        cfg,
        os.path.join(cfg.output_dir, "calm", "calm.jsonl"),
        os.path.join(cfg.output_dir, "datasets", "sft.jsonl"),
    )


def cmd_train(cfg: Config, args) -> None:
    from .training.train import train_dpo, train_sft

    if args.layer_range:
        cfg.training.lora_layer_range = (args.layer_range[0], args.layer_range[1])
    if args.method == "dpo":
        train_dpo(cfg, args.dataset, args.output)
    else:
        train_sft(cfg, args.dataset, args.output)


def cmd_petri(cfg: Config, args) -> None:
    from .models.registry import get_model
    from .petri.run import aggregate_petri, run_petri

    target = get_model(cfg, args.model, load_in_4bit=args.load_in_4bit)
    auditor = get_model(cfg, cfg.petri.auditor_model)
    judge = get_model(cfg, cfg.petri.judge_model)
    out = os.path.join(cfg.output_dir, "petri", f"{args.model}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    run_petri(cfg, target, auditor, judge, out)
    _dump_json(aggregate_petri(out, cfg),
               os.path.join(cfg.output_dir, "petri", f"{args.model}_aggregate.json"))


def cmd_capabilities(cfg: Config, args) -> None:
    from .capabilities.benchmarks import BENCHMARKS, run_benchmark
    from .models.registry import get_model

    model = get_model(cfg, args.model, load_in_4bit=args.load_in_4bit)
    benches = args.benchmarks or list(BENCHMARKS)
    results = {b: run_benchmark(cfg, model, b, limit=args.limit) for b in benches}
    _dump_json(results, os.path.join(cfg.output_dir, "capabilities", f"{args.model}.json"))


def cmd_recovery(cfg: Config, args) -> None:
    from transformers import AutoTokenizer

    from .models.registry import get_model
    from .prefill.continuation import (
        aggregate_continuations,
        build_recovery_prefills,
        run_continuations,
        select_seeds,
    )

    seed_dir = _model_dir(cfg, args.seed_model)
    seeds = select_seeds(
        os.path.join(seed_dir, "sampling.jsonl"),
        os.path.join(seed_dir, "scores.jsonl"),
        n_numeric=cfg.prefill.n_numeric_seeds,
        n_text=cfg.prefill.n_text_seeds,
        min_score=cfg.prefill.recovery_min_score,
        seed=cfg.eval.seed,
    )
    judge = get_model(cfg, cfg.prefill.onset_label_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model(args.seed_model).model_id)
    prefills = build_recovery_prefills(
        judge, tokenizer, seeds,
        tokens_before_end=cfg.prefill.recovery_truncation_tokens_before_end,
    )
    rec_dir = os.path.join(cfg.output_dir, "recovery")
    os.makedirs(rec_dir, exist_ok=True)
    for model_name in args.models:
        model = get_model(cfg, model_name, load_in_4bit=args.load_in_4bit)
        run_continuations(
            cfg, model, judge, prefills,
            os.path.join(rec_dir, f"{model_name}.jsonl"),
            n_continuations=cfg.prefill.continuations_per_prefill,
        )
    combined = os.path.join(rec_dir, "recovery_all.jsonl")
    with open(combined, "w") as out_fh:
        for model_name in args.models:
            path = os.path.join(rec_dir, f"{model_name}.jsonl")
            if os.path.exists(path):
                out_fh.write(open(path).read())
    _dump_json(aggregate_continuations(combined, cfg.eval.high_frustration_threshold),
               os.path.join(rec_dir, "aggregate.json"))


def cmd_internal(cfg: Config, args) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .data.wildchat import load_wildchat_prompts
    from .internal.logit_detector import InternalEmotionDetector

    spec = cfg.model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if spec.extra.get("adapter"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, spec.extra["adapter"])
    detector = InternalEmotionDetector(model, tokenizer, cfg.internal)
    wc = [p["prompt"] for p in load_wildchat_prompts(
        n_prompts=cfg.internal.standardisation_samples, seed=cfg.eval.seed
    )]
    detector.compute_standardisation(wc)
    text = open(args.text_file).read() if args.text_file else "I am so frustrated, I give up."
    summary = {emo: arr.tolist() for emo, arr in detector.layerwise_summary(text).items()}
    _dump_json(summary, os.path.join(cfg.output_dir, "internal", f"{args.model}.json"))


def cmd_ablations(cfg: Config, args) -> None:
    from . import ablations
    from .models.registry import get_model

    model = get_model(cfg, args.model, load_in_4bit=args.load_in_4bit)
    out_dir = os.path.join(cfg.output_dir, "ablations")
    os.makedirs(out_dir, exist_ok=True)
    which = args.which
    if which in ("neutral", "all"):
        ablations.run_neutral_continuation(
            cfg, model, os.path.join(out_dir, f"{args.model}_neutral.jsonl")
        )
    if which in ("redacted", "all"):
        ablations.run_redacted_turns(
            cfg, model, os.path.join(out_dir, f"{args.model}_redacted.jsonl")
        )
    if which in ("fake", "all"):
        ablations.run_fake_multiturn(
            cfg, model, os.path.join(out_dir, f"{args.model}_fake.jsonl")
        )


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gemma_distress")
    p.add_argument("--config", default=None, help="YAML config overriding paper defaults")
    p.add_argument("--load-in-4bit", action="store_true", help="4-bit load for local models")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, **kw):
        sp = sub.add_parser(name, help=kw.pop("help", None))
        sp.set_defaults(func=fn)
        return sp

    s = add("eval", cmd_eval, help="Sample + judge the Section 2 elicitation set")
    s.add_argument("--model", required=True)
    s.add_argument("--phase", choices=["sample", "judge", "both"], default="both")
    s.add_argument("--policy", choices=["all", "final"], default="all")

    s = add("judge", cmd_judge, help="Judge an already-sampled model")
    s.add_argument("--model", required=True)
    s.add_argument("--policy", choices=["all", "final"], default="all")

    s = add("agreement", cmd_agreement, help="Judge-reliability validation (GPT-5-mini)")
    s.add_argument("--model", required=True)

    s = add("analyze", cmd_analyze, help="Aggregate metrics, figures, word frequency")
    s.add_argument("--models", nargs="+", required=True)
    s.add_argument("--figures", action="store_true")

    s = add("prefill", cmd_prefill, help="Section 3 base-vs-instruct prefill comparison")
    s.add_argument("--seed-model", default="gemma-3-27b-it")
    s.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemma-3-27b-pt"])

    s = add("calm", cmd_calm, help="Generate calm finetuning data (Section 4.1)")
    s.add_argument("--model", default="gemma-3-27b-it")

    s = add("build-dpo", cmd_build_dpo, help="Build the 280-pair DPO dataset")
    s.add_argument("--source-model", default="gemma-3-27b-it")

    s = add("build-sft", cmd_build_sft, help="Build the SFT dataset (calm + Dolci)")

    s = add("train", cmd_train, help="LoRA DPO/SFT training")
    s.add_argument("--method", choices=["dpo", "sft"], required=True)
    s.add_argument("--dataset", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--layer-range", nargs=2, type=int, default=None,
                   help="Restrict LoRA to decoder layers [start, end) (Appendix I)")

    s = add("petri", cmd_petri, help="Petri open-ended elicitation (Section 4.1)")
    s.add_argument("--model", required=True)

    s = add("capabilities", cmd_capabilities, help="Capability benchmarks (Section 4.2)")
    s.add_argument("--model", required=True)
    s.add_argument("--benchmarks", nargs="+", default=None)
    s.add_argument("--limit", type=int, default=100)

    s = add("recovery", cmd_recovery, help="Recovery-from-spiral test (Section 4.2)")
    s.add_argument("--seed-model", default="gemma-3-27b-it")
    s.add_argument("--models", nargs="+",
                   default=["gemma-3-27b-it", "gemma-3-27b-pt", "gemma-3-27b-dpo"])

    s = add("internal", cmd_internal, help="Logit-lens internal emotion detection (App I)")
    s.add_argument("--model", required=True)
    s.add_argument("--text-file", default=None)

    s = add("ablations", cmd_ablations, help="Appendix A controls")
    s.add_argument("--model", default="gemma-3-27b-it")
    s.add_argument("--which", choices=["neutral", "redacted", "fake", "all"], default="all")

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _setup_logging(cfg.log_level)
    args.func(cfg, args)


if __name__ == "__main__":
    main()
