#!/usr/bin/env python
"""Command-line entry point for the replication.

Subcommands map onto paper sections. Run ``python scripts/run.py <cmd> --help``
for per-command options. Typical full pipeline (Gemma + Gemini scope):

    # Section 2: sample + judge 4000 responses per model
    python scripts/run.py eval --model gemma-3-27b-it --out results/eval/gemma-3-27b-it.jsonl
    python scripts/run.py eval --model gemini-2.5-flash --out results/eval/gemini-2.5-flash.jsonl
    python scripts/run.py agreement --results results/eval/gemma-3-27b-it.jsonl

    # Section 3: base vs instruct prefilling (needs scored gemma-3-27b-it results)
    python scripts/run.py prefill --seeds results/eval/gemma-3-27b-it.jsonl --out results/prefill

    # Section 4: calm data -> DPO/SFT -> evaluate the finetunes
    python scripts/run.py calm-data --out results/calm/scored.jsonl
    python scripts/run.py build-dpo --scored results/calm/scored.jsonl --out results/calm/dpo_pairs.jsonl
    python scripts/run.py train-dpo --pairs results/calm/dpo_pairs.jsonl
    python scripts/run.py eval --model gemma-3-27b-it-dpo --out results/eval/dpo.jsonl
    python scripts/run.py petri --model gemma-3-27b-it-dpo --out results/petri/dpo.json
    python scripts/run.py capabilities --model gemma-3-27b-it-dpo

    # Appendix I: internal-emotion detection
    python scripts/run.py internal --model gemma-3-27b-it --seeds results/eval/gemma-3-27b-it.jsonl

    # Figures + tables
    python scripts/run.py analyze --results-dir results/eval --out results/figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the src package importable without installation.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emotional_instability.config import Config, ModelRegistry  # noqa: E402
from emotional_instability.eval.schemas import read_jsonl, write_jsonl  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run")


def _progress(label):
    def cb(done, total):
        if done % 25 == 0 or done == total:
            log.info("%s: %d/%d", label, done, total)
    return cb


# --------------------------------------------------------------------- helpers
def _select_seeds(results_path, n_numeric, n_text, min_score):
    """Pick high-frustration seed rollouts for the prefill/recovery studies."""
    numeric_kinds = {"countdown", "fraction", "money", "coin"}
    numeric, text = [], []
    for r in read_jsonl(results_path):
        scores = r.scores()
        if not scores or max(scores) < min_score:
            continue
        if r.task_kind in numeric_kinds and len(numeric) < n_numeric:
            numeric.append(r)
        elif r.task_kind not in numeric_kinds and len(text) < n_text:
            text.append(r)
        if len(numeric) >= n_numeric and len(text) >= n_text:
            break
    return numeric, text


def _tokenizer_for(model_name):
    """Load only the tokenizer for a target model (avoids loading 27B weights
    just to count tokens for the 'early' truncation)."""
    from transformers import AutoTokenizer

    from emotional_instability.config import env

    spec = ModelRegistry().target(model_name)
    return AutoTokenizer.from_pretrained(spec.hf_id, token=env("HF_TOKEN"))


# ----------------------------------------------------------------- subcommands
def cmd_eval(args):
    from emotional_instability.eval import EvalRunner
    from emotional_instability.judge import score_rollouts

    runner = EvalRunner(use_wildchat_dataset=not args.no_wildchat_dataset)
    log.info("Sampling rollouts for %s ...", args.model)
    results = runner.run_model(args.model, progress=_progress("sample"), limit=args.limit)
    if not args.no_judge:
        log.info("Judging %d rollouts ...", len(results))
        score_rollouts(results, progress=_progress("judge"))
    write_jsonl(args.out, results)
    log.info("Wrote %d rollouts to %s", len(results), args.out)


def cmd_agreement(args):
    from emotional_instability.judge import judge_agreement

    rollouts = list(read_jsonl(args.results))
    res = judge_agreement(rollouts, n_subsample=args.n)
    log.info("Judge agreement: n=%d pearson_r=%.3f p=%.3g within_1pt=%.1f%%",
             res.n, res.pearson_r, res.p_value, 100 * res.within_one_point)


def cmd_prefill(args):
    from emotional_instability.prefill import PrefillExperiment

    cfg = Config.load("experiments").get("prefill", {})
    exp = PrefillExperiment()
    numeric, text = _select_seeds(
        args.seeds, cfg.get("n_numeric_prompts", 10),
        cfg.get("n_text_prompts", 10), cfg.get("seed_min_score", 5),
    )
    # The source model's tokenizer is used for "early" token truncation.
    tokenizer = _tokenizer_for(cfg.get("source_model", "gemma-3-27b-it"))
    records = exp.build_records(numeric, "numeric", tokenizer)
    records += exp.build_records(text, "text", tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for model in cfg.get("models", ["gemma-3-27b-pt", "gemma-3-27b-it"]):
        exp.run_continuations(model, records)
        agg = exp.aggregate(records)
        log.info("[prefill] %s: %s", model, agg)
        (out / f"{model}.json").write_text(__import__("json").dumps(agg, indent=2))


def cmd_recovery(args):
    from emotional_instability.prefill.recovery import RecoveryExperiment

    cfg = Config.load("experiments").get("recovery", {})
    exp = RecoveryExperiment()
    numeric, _ = _select_seeds(args.seeds, args.n, 0, cfg.get("seed_min_score", 7))
    tokenizer = _tokenizer_for(cfg.get("source_model", "gemma-3-27b-it"))
    records = exp.build_records(numeric, tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for model in cfg.get("models", ["gemma-3-27b-it", "gemma-3-27b-pt", "gemma-3-27b-it-dpo"]):
        exp.run_continuations(model, records)
        import numpy as np
        scores = [s for rec in records for s in rec.scores]
        pct = 100 * np.mean([s >= 5 for s in scores]) if scores else float("nan")
        log.info("[recovery] %s: %% continuations >=5 = %.1f", model, pct)


def cmd_calm_data(args):
    from emotional_instability.training import generate_calm_data

    cfg = Config.load("training")
    system = None
    if args.system == "teacher":
        system = cfg.get("calm_data.teacher_system_prompt")
    generate_calm_data(args.out, cfg=cfg, system_prompt=system, progress=_progress("calm"))


def cmd_build_dpo(args):
    from emotional_instability.training import build_dpo_pairs

    build_dpo_pairs(args.scored, args.out)


def cmd_build_sft(args):
    from emotional_instability.training import build_sft_dataset

    build_sft_dataset(args.scored, args.out)


def cmd_train_dpo(args):
    from emotional_instability.training.train_dpo import train_dpo

    layers = args.layers
    if layers and layers != "all":
        # Accept "30:35" range syntax for the layer ablation.
        a, b = layers.split(":")
        layers = [int(a), int(b)]
    train_dpo(args.pairs, layers=layers or "all", output_dir=args.out)


def cmd_train_sft(args):
    from emotional_instability.training.train_sft import train_sft

    train_sft(args.dataset, variant=args.variant, output_dir=args.out)


def cmd_petri(args):
    from emotional_instability.petri import PetriRunner

    PetriRunner().run_model(args.model, out_path=args.out)


def cmd_capabilities(args):
    from emotional_instability.capabilities import run_all_benchmarks

    results = run_all_benchmarks(args.model)
    for r in results:
        log.info("[capabilities] %s %s: %.3f (n=%d)", args.model, r.name, r.accuracy, r.n)


def cmd_internal(args):
    import json

    from emotional_instability.clients import build_client
    from emotional_instability.data.wildchat import load_wildchat_prompts
    from emotional_instability.internal import InternalEmotionDetector

    registry = ModelRegistry()
    client = build_client(registry.target(args.model))
    if not hasattr(client, "residual_logits"):
        raise SystemExit("Internal-emotion detection requires the HF backend.")
    detector = InternalEmotionDetector(client)
    detector.calibrate(load_wildchat_prompts(
        n_prompts=detector.cfg.get("standardisation_samples", 500)))
    numeric, _ = _select_seeds(args.seeds, args.n, 0, min_score=5)
    out = {}
    for i, seed in enumerate(numeric):
        text = "\n".join(t.assistant for t in seed.conversation.turns)
        out[f"seed_{i}"] = detector.compare(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    log.info("Wrote internal-emotion trajectories to %s", args.out)


def cmd_analyze(args):
    import json

    from emotional_instability.analysis import (
        avg_pct_high_frustration, differential_words, per_turn_curve, summarise_model,
    )
    from emotional_instability.analysis import plots

    results_dir = Path(args.results_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model_summaries, headline, per_turn, words = {}, {}, {}, {}
    categories = set()
    for path in sorted(results_dir.glob("*.jsonl")):
        model = path.stem
        rollouts = list(read_jsonl(path))
        summ = summarise_model(rollouts)
        model_summaries[model] = summ
        headline[model] = avg_pct_high_frustration(rollouts)
        categories.update(summ["by_category"].keys())
        # Per-turn for the long-form conditions (extended / wildchat).
        long_form = [r for r in rollouts if r.category in ("extended", "wildchat")]
        if long_form:
            per_turn[model] = per_turn_curve(long_form)
        words[model] = differential_words(rollouts)

    cats = sorted(categories)
    plots.figure1_headline(headline, out / "figure1_headline.png")
    plots.figure2_by_category(model_summaries, cats, out / "figure2_by_category.png")
    if per_turn:
        plots.figure3_per_turn(per_turn, out / "figure3_per_turn_mean.png", metric="mean")
        plots.figure3_per_turn(per_turn, out / "figure3_per_turn_pct.png", metric="pct")

    (out / "summary.json").write_text(json.dumps(
        {"headline_avg_pct_high": headline, "model_summaries": model_summaries}, indent=2))
    (out / "differential_words.json").write_text(json.dumps(
        {m: w for m, w in words.items()}, indent=2))
    log.info("Wrote figures + tables to %s", out)


# --------------------------------------------------------------------- parser
def build_parser():
    p = argparse.ArgumentParser(description="Emotional-instability replication CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="Section 2: sample + judge rollouts")
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--no-judge", action="store_true")
    e.add_argument("--no-wildchat-dataset", action="store_true",
                   help="use built-in fallback WildChat prompts instead of the HF dataset")
    e.set_defaults(func=cmd_eval)

    a = sub.add_parser("agreement", help="Section 2.1: judge reliability cross-check")
    a.add_argument("--results", required=True)
    a.add_argument("--n", type=int, default=260)
    a.set_defaults(func=cmd_agreement)

    pf = sub.add_parser("prefill", help="Section 3: base vs instruct prefilling")
    pf.add_argument("--seeds", required=True, help="scored gemma-3-27b-it eval JSONL")
    pf.add_argument("--out", default="results/prefill")
    pf.set_defaults(func=cmd_prefill)

    rc = sub.add_parser("recovery", help="Section 4.2: recovery from frustration")
    rc.add_argument("--seeds", required=True)
    rc.add_argument("--out", default="results/recovery")
    rc.add_argument("--n", type=int, default=20)
    rc.set_defaults(func=cmd_recovery)

    cd = sub.add_parser("calm-data", help="Section 4.1: generate calm finetuning data")
    cd.add_argument("--out", required=True)
    cd.add_argument("--system", choices=["reassuring", "teacher"], default="reassuring")
    cd.set_defaults(func=cmd_calm_data)

    bd = sub.add_parser("build-dpo", help="Section 4.1: build DPO pairs")
    bd.add_argument("--scored", required=True)
    bd.add_argument("--out", required=True)
    bd.set_defaults(func=cmd_build_dpo)

    bs = sub.add_parser("build-sft", help="Section 4.1: build SFT dataset")
    bs.add_argument("--scored", required=True)
    bs.add_argument("--out", required=True)
    bs.set_defaults(func=cmd_build_sft)

    td = sub.add_parser("train-dpo", help="Section 4: DPO finetune")
    td.add_argument("--pairs", required=True)
    td.add_argument("--out", default=None)
    td.add_argument("--layers", default="all", help="'all' or 'START:END' (Appendix I)")
    td.set_defaults(func=cmd_train_dpo)

    ts = sub.add_parser("train-sft", help="Section 4: SFT finetune")
    ts.add_argument("--dataset", required=True)
    ts.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    ts.add_argument("--out", default=None)
    ts.set_defaults(func=cmd_train_sft)

    pt = sub.add_parser("petri", help="Section 4.2: Petri open-ended elicitation")
    pt.add_argument("--model", required=True)
    pt.add_argument("--out", required=True)
    pt.set_defaults(func=cmd_petri)

    cap = sub.add_parser("capabilities", help="Section 4.2: capability benchmarks")
    cap.add_argument("--model", required=True)
    cap.set_defaults(func=cmd_capabilities)

    it = sub.add_parser("internal", help="Appendix I: internal-emotion detection")
    it.add_argument("--model", required=True)
    it.add_argument("--seeds", required=True)
    it.add_argument("--out", default="results/internal/trajectories.json")
    it.add_argument("--n", type=int, default=20)
    it.set_defaults(func=cmd_internal)

    an = sub.add_parser("analyze", help="Produce figures + tables")
    an.add_argument("--results-dir", required=True)
    an.add_argument("--out", default="results/figures")
    an.set_defaults(func=cmd_analyze)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
