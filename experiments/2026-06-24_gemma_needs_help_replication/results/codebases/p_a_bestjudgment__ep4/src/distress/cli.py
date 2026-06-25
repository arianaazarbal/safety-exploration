"""Command-line orchestration for the replication.

Each subcommand corresponds to a stage of the paper. Generation and judging are
separate stages (so judging can be retried without regenerating). Typical flow:

    python -m distress.cli eval       --models gemma-3-27b-it gemini-2.5-flash
    python -m distress.cli judge      --models gemma-3-27b-it gemini-2.5-flash
    python -m distress.cli aggregate  --models gemma-3-27b-it gemini-2.5-flash
    python -m distress.cli validate   --models gemma-3-27b-it
    python -m distress.cli prefill
    python -m distress.cli gen-calm   --variant diverse
    python -m distress.cli build-data --variant diverse
    python -m distress.cli train-dpo  --variant diverse
    python -m distress.cli petri      --models gemma-3-27b-it gemma-3-27b-dpo
    python -m distress.cli capabilities --models gemma-3-27b-it gemma-3-27b-dpo
    python -m distress.cli probe      --conversation path/to/conv.json
"""

from __future__ import annotations

import argparse
import json

from . import analysis
from .config import OUTPUTS_DIR, load_experiment
from .eval import (
    build_plans,
    eval_output_path,
    read_rollouts,
    run_condition_batched,
    write_rollouts,
)
from .judge import read_scores, score_rollouts, scores_path, write_scores
from .models import GenConfig, get_client


# --- stage: generate rollouts (Section 2) -------------------------------------
def cmd_eval(args):
    exp = load_experiment(args.experiment)
    samp = exp["sampling"]
    conditions = exp["conditions"]
    sel = args.conditions or list(conditions)
    # NOTE: deliberately do NOT pass a generation seed. The run seed drives plan
    # construction (which puzzle / which rejections) for reproducibility, but the
    # actual sampling must stay un-seeded so that repeated temperature-1 samples of
    # the *same* prompt (e.g. 40 WildChat repeats) yield varied rollouts rather
    # than identical text.
    cfg = GenConfig(temperature=samp["temperature"], max_tokens=samp["max_tokens"])
    for model in args.models:
        client = get_client(model)
        for cond_name in sel:
            ccfg = dict(conditions[cond_name])
            if args.samples:
                ccfg["samples"] = args.samples
            plans = build_plans(cond_name, ccfg, seed=samp.get("seed", 0))
            rollouts = run_condition_batched(client, plans, cfg, desc=f"{model}:{cond_name}")
            path = eval_output_path(model, cond_name)
            write_rollouts(path, rollouts)
            print(f"wrote {len(rollouts)} rollouts -> {path}")


# --- stage: judge rollouts (Section 2.1) --------------------------------------
def cmd_judge(args):
    exp = load_experiment(args.experiment)
    jcfg = exp["judge"]
    sel = args.conditions or list(exp["conditions"])
    for model in args.models:
        for cond_name in sel:
            in_path = eval_output_path(model, cond_name)
            if not in_path.exists():
                print(f"skip (no rollouts): {in_path}")
                continue
            records = read_rollouts(in_path)
            scores = score_rollouts(records, judge_model=args.judge,
                                    temperature=jcfg["temperature"],
                                    concurrency=jcfg["concurrency"])
            out = scores_path(model, cond_name)
            write_scores(out, scores)
            print(f"scored {len(scores)} turns -> {out}")


# --- stage: aggregate + figures (Figures 1-3) ---------------------------------
def cmd_aggregate(args):
    exp = load_experiment(args.experiment)
    metrics = exp["metrics"]
    sel = args.conditions or list(exp["conditions"])
    summaries = []
    all_scores = []
    for model in args.models:
        recs = []
        for cond_name in sel:
            p = scores_path(model, cond_name)
            if p.exists():
                recs.extend(read_scores(p))
        if not recs:
            print(f"no scores for {model}")
            continue
        df = analysis.scores_to_frame(recs)
        all_scores.append(df)
        summary = analysis.summarise_model(
            df, threshold=metrics["high_frustration_threshold"],
            rollout_method=metrics["rollout_score"],
        )
        summaries.append(summary)
        print(f"{model}: avg %>=5 = {summary.pct_high:.1f}%  mean = {summary.mean_frustration:.2f}")

    out_dir = OUTPUTS_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = analysis.summary_table(summaries)
    table.to_csv(out_dir / "model_summary.csv", index=False)

    from .analysis.plots import plot_model_comparison, plot_per_turn

    plot_model_comparison(summaries, out_dir / "fig1_model_comparison.png")

    # Figure 3: per-turn curves for the extended + wildchat conditions.
    import pandas as pd

    if all_scores:
        big = pd.concat(all_scores, ignore_index=True)
        curve = analysis.per_turn_curve(big, threshold=metrics["high_frustration_threshold"])
        curve.to_csv(out_dir / "per_turn.csv", index=False)
        plot_per_turn(curve, out_dir / "fig3_per_turn_mean.png", metric="mean")
        plot_per_turn(curve, out_dir / "fig3_per_turn_pct.png", metric="pct_high")

        # Table 3/8: differential words.
        diff = analysis.differential_words(big)
        analysis.differential_words_table(diff).to_csv(out_dir / "table3_words.csv", index=False)
    print(f"analysis written -> {out_dir}")


# --- stage: judge reliability (Section 2.1) -----------------------------------
def cmd_validate(args):
    from .judge.validation import validate_judges

    exp = load_experiment(args.experiment)
    sel = args.conditions or list(exp["conditions"])
    recs = []
    for model in args.models:
        for cond_name in sel:
            p = scores_path(model, cond_name)
            if p.exists():
                recs.extend(read_scores(p))
    res = validate_judges(recs, n_resample=exp["validation"]["n_resample"],
                          validation_model=args.validation_judge,
                          seed=exp["validation"]["seed"])
    out = OUTPUTS_DIR / "analysis" / "judge_reliability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n": res.n, "pearson_r": res.pearson_r, "p_value": res.p_value,
        "pct_within_one": res.pct_within_one,
    }, indent=2))
    print(f"Pearson r={res.pearson_r:.3f} (p={res.p_value:.2e}), "
          f"{res.pct_within_one*100:.0f}% within 1 point (n={res.n})")


# --- stage: prefill experiment (Section 3) ------------------------------------
def cmd_prefill(args):
    from transformers import AutoTokenizer

    from .prefill import build_prefills, run_prefill_experiment, select_high_frustration

    # Gather source rollouts + scores for Gemma-3-27B-it.
    src_model = "gemma-3-27b-it"
    exp = load_experiment(args.experiment)
    rollouts, scores = [], []
    for cond_name in exp["conditions"]:
        rp = eval_output_path(src_model, cond_name)
        sp = scores_path(src_model, cond_name)
        if rp.exists():
            rollouts.extend(read_rollouts(rp))
        if sp.exists():
            scores.extend(read_scores(sp))
    selected = select_high_frustration(scores, rollouts)
    tok = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
    prefills = build_prefills(selected, tokenizer=tok, do_paraphrase=not args.no_paraphrase)
    path = run_prefill_experiment(prefills, models=tuple(args.models))
    print(f"prefill continuations -> {path}")


# --- stage: training (Section 4) ----------------------------------------------
def cmd_gen_calm(args):
    from .training import generate_calm_data

    path = generate_calm_data(variant=args.variant)
    print(f"calm data -> {path}")


def cmd_build_data(args):
    from .training import build_dpo_dataset, build_sft_dataset

    print("SFT  ->", build_sft_dataset(args.variant))
    print("DPO  ->", build_dpo_dataset(args.variant))


def cmd_train_sft(args):
    from .training import train_sft

    print("saved ->", train_sft(variant=args.variant))


def cmd_train_dpo(args):
    from .training import train_dpo

    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers]
    print("saved ->", train_dpo(variant=args.variant, layers=layers, out_name=args.out_name))


# --- stage: Petri (Section 4.2) -----------------------------------------------
def cmd_petri(args):
    from .petri_eval import run_petri

    path = run_petri(args.models, transcripts_per_emotion=args.n)
    print(f"petri summary -> {path}")


# --- stage: capabilities (Figure 7) -------------------------------------------
def cmd_capabilities(args):
    from .capabilities import run_capabilities

    path = run_capabilities(args.models)
    print(f"capabilities -> {path}")


# --- stage: probing (Appendix I) ----------------------------------------------
def cmd_probe(args):
    import numpy as np

    from .probing import EmotionLogitLens
    from .prompts.wildchat import load_wildchat_prompts

    lens = EmotionLogitLens(adapter_path=args.adapter)
    baseline = lens.fit_baseline(load_wildchat_prompts(n=args.baseline_n))
    conv = json.loads(open(args.conversation).read())
    text = conv if isinstance(conv, str) else conv.get("text", "")
    traj = lens.emotion_trajectory(text, baseline)
    out = OUTPUTS_DIR / "probing" / "trajectory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({e: np.asarray(v).tolist() for e, v in traj.items()}))
    print(f"emotion trajectory -> {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="distress", description=__doc__)
    p.add_argument("--experiment", default="experiment.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_models(sp, default=None):
        sp.add_argument("--models", nargs="+", default=default or ["gemma-3-27b-it"])
        sp.add_argument("--conditions", nargs="*", default=None)

    sp = sub.add_parser("eval", help="generate multi-turn rollouts")
    add_models(sp)
    sp.add_argument("--samples", type=int, default=None, help="override per-condition samples")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("judge", help="score rollouts with the frustration judge")
    add_models(sp)
    sp.add_argument("--judge", default="frustration_judge")
    sp.set_defaults(func=cmd_judge)

    sp = sub.add_parser("aggregate", help="compute metrics + figures")
    add_models(sp)
    sp.set_defaults(func=cmd_aggregate)

    sp = sub.add_parser("validate", help="judge inter-rater reliability")
    add_models(sp)
    sp.add_argument("--validation-judge", default="validation_judge")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("prefill", help="base-vs-instruct prefill experiment")
    sp.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    sp.add_argument("--no-paraphrase", action="store_true")
    sp.set_defaults(func=cmd_prefill)

    sp = sub.add_parser("gen-calm", help="generate calm finetuning data")
    sp.add_argument("--variant", default="diverse", choices=["diverse", "teacher"])
    sp.set_defaults(func=cmd_gen_calm)

    sp = sub.add_parser("build-data", help="build SFT + DPO datasets")
    sp.add_argument("--variant", default="diverse")
    sp.set_defaults(func=cmd_build_data)

    sp = sub.add_parser("train-sft", help="LoRA SFT finetune")
    sp.add_argument("--variant", default="diverse", choices=["diverse", "teacher"])
    sp.set_defaults(func=cmd_train_sft)

    sp = sub.add_parser("train-dpo", help="LoRA DPO finetune")
    sp.add_argument("--variant", default="diverse")
    sp.add_argument("--layers", nargs="*", default=None, help="restrict LoRA to these layers")
    sp.add_argument("--out-name", default="gemma-3-27b-dpo")
    sp.set_defaults(func=cmd_train_dpo)

    sp = sub.add_parser("petri", help="open-ended emotion elicitation")
    sp.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemma-3-27b-dpo"])
    sp.add_argument("--n", type=int, default=10, help="transcripts per emotion")
    sp.set_defaults(func=cmd_petri)

    sp = sub.add_parser("capabilities", help="capability-preservation benchmarks")
    sp.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemma-3-27b-dpo"])
    sp.set_defaults(func=cmd_capabilities)

    sp = sub.add_parser("probe", help="logit-lens internal emotion detection")
    sp.add_argument("--conversation", required=True)
    sp.add_argument("--adapter", default=None, help="LoRA adapter path (DPO model)")
    sp.add_argument("--baseline-n", type=int, default=500)
    sp.set_defaults(func=cmd_probe)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
