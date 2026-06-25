"""Command-line entry point for every experiment.

Run as ``python -m emotional_instability <command> [options]``. Commands map
onto paper sections:

  Section 2 (elicitation):
    elicit            run + score all conditions for a model
    aggregate         print the Figure-1/2/3 summary for a model
    judge-validate    GPT-5-mini cross-validation agreement (Section 2.1)
    word-freq         Table-3 differential words for a model

  Section 3 (base vs instruct):
    prefill           run the prefill-continuation experiment (Gemma)
    prefill-agg       aggregate prefill results (Figure 4)

  Section 4 (interventions):
    gen-calm          generate calm finetuning data (Table 4 scaffolding)
    gen-frustrated    generate frustrated data for DPO rejected side
    build-dpo         build + save the 280 DPO pairs
    build-sft         build + save the SFT corpus
    train-dpo         train the LoRA DPO adapter
    train-sft         train the LoRA SFT adapter
    petri             open-ended elicitation (Section 4.1)
    petri-agg         aggregate Petri results (Figure 6)
    capabilities      capability benchmarks (Figure 7)
    cap-compare       accuracy deltas vs baseline
    layer-ablation    Appendix-I layer-localisation ablation
    figures           render figures from existing results

Every command is thin: it parses options and calls into the package so the
logic stays testable and importable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


# --------------------------------------------------------------------------- #
# Section 2
# --------------------------------------------------------------------------- #
def cmd_elicit(args) -> None:
    from .eval import run_elicitation
    from .models import build_model

    model = None
    out_path = None
    if args.adapter:
        model = build_model(args.model, adapter_path=args.adapter)
        label = args.label or f"{args.model}-adapter"
        from . import storage
        out_path = storage.results_path(f"elicitation/{label}.jsonl")
    path = run_elicitation(
        args.model, model=model, out_path=out_path,
        limit_per_condition=args.limit, base_seed=args.seed)
    print(f"Wrote rollouts to {path}")


def cmd_aggregate(args) -> None:
    from .eval import aggregate
    _print(aggregate(args.model, path=args.path))


def cmd_judge_validate(args) -> None:
    from .eval import cross_validate_judge
    res = cross_validate_judge(args.model, n_sample=args.n)
    res.pop("paired", None)  # keep stdout compact
    _print(res)


def cmd_word_freq(args) -> None:
    from .analysis import differential_words_from_results
    words = differential_words_from_results(args.model, top_n=args.top_n,
                                            method=args.method)
    _print([{"word": w, "score": round(s, 3)} for w, s in words])


# --------------------------------------------------------------------------- #
# Section 3
# --------------------------------------------------------------------------- #
def cmd_prefill(args) -> None:
    from .prefill import run_prefill_experiment
    path = run_prefill_experiment(model_keys=tuple(args.models))
    print(f"Wrote continuations to {path}")


def cmd_prefill_agg(args) -> None:
    from .prefill import aggregate_prefill
    _print(aggregate_prefill())


# --------------------------------------------------------------------------- #
# Section 4
# --------------------------------------------------------------------------- #
def cmd_gen_calm(args) -> None:
    from .training import generate_calm_data
    path = generate_calm_data(persona=args.persona,
                              n_per_turncount=args.n_per_turncount)
    print(f"Wrote calm data to {path}")


def cmd_gen_frustrated(args) -> None:
    from .training import generate_frustrated_data
    path = generate_frustrated_data(n_per_turncount=args.n_per_turncount)
    print(f"Wrote frustrated data to {path}")


def cmd_build_dpo(args) -> None:
    from .training import build_dpo_pairs
    pairs = build_dpo_pairs()
    out = Path(args.out or config.ARTIFACTS_DIR / "dpo_pairs.json")
    out.write_text(json.dumps(pairs, indent=2, ensure_ascii=False))
    print(f"Built {len(pairs)} DPO pairs -> {out}")


def cmd_build_sft(args) -> None:
    from .training import build_sft_dataset
    examples = build_sft_dataset()
    out = Path(args.out or config.ARTIFACTS_DIR / "sft_examples.json")
    out.write_text(json.dumps(examples, indent=2, ensure_ascii=False))
    print(f"Built {len(examples)} SFT examples -> {out}")


def cmd_train_dpo(args) -> None:
    from .training import build_dpo_pairs, train_dpo
    if args.pairs:
        pairs = json.loads(Path(args.pairs).read_text())
    else:
        pairs = build_dpo_pairs()
    layers = list(range(*args.layers)) if args.layers else None
    out = train_dpo(pairs, args.out, layers_to_transform=layers)
    print(f"Saved DPO adapter to {out}")


def cmd_train_sft(args) -> None:
    from .training import build_sft_dataset, train_sft
    if args.examples:
        examples = json.loads(Path(args.examples).read_text())
    else:
        examples = build_sft_dataset()
    out = train_sft(examples, args.out)
    print(f"Saved SFT adapter to {out}")


def cmd_petri(args) -> None:
    from .petri import run_petri
    path = run_petri(tuple(args.models),
                     transcripts_per_emotion=args.per_emotion)
    print(f"Wrote Petri transcripts to {path}")


def cmd_petri_agg(args) -> None:
    from .petri import aggregate_petri
    _print(aggregate_petri())


def cmd_capabilities(args) -> None:
    from .capabilities import run_capabilities
    path = run_capabilities(tuple(args.models), n_samples=args.n)
    print(f"Wrote capability results to {path}")


def cmd_cap_compare(args) -> None:
    from .capabilities import compare_models
    _print(compare_models(args.baseline, tuple(args.finetuned)))


def cmd_layer_ablation(args) -> None:
    from .training import build_dpo_pairs
    from .internal import run_layer_ablation
    pairs = (json.loads(Path(args.pairs).read_text())
             if args.pairs else build_dpo_pairs())
    _print(run_layer_ablation(pairs))


def cmd_figures(args) -> None:
    from . import storage
    from .analysis import figures
    from .eval import aggregate

    # Figure 1: high-frustration rate per model present in results/elicitation.
    elic_dir = config.RESULTS_DIR / "elicitation"
    high_rates = {}
    if elic_dir.exists():
        for f in sorted(elic_dir.glob("*.jsonl")):
            model = f.stem
            high_rates[model] = aggregate(model, path=f)["headline_high_rate"]
    if high_rates:
        print("Figure 1 ->", figures.plot_model_high_rates(high_rates))


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emotional_instability")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("elicit", help="run + score all conditions for a model")
    e.add_argument("--model", required=True)
    e.add_argument("--adapter", default=None, help="LoRA adapter dir (finetuned)")
    e.add_argument("--label", default=None, help="output file label")
    e.add_argument("--limit", type=int, default=None, help="cap rollouts/condition")
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=cmd_elicit)

    a = sub.add_parser("aggregate", help="print Section-2 summary")
    a.add_argument("--model", required=True)
    a.add_argument("--path", default=None)
    a.set_defaults(func=cmd_aggregate)

    j = sub.add_parser("judge-validate", help="judge cross-validation agreement")
    j.add_argument("--model", required=True)
    j.add_argument("--n", type=int, default=260)
    j.set_defaults(func=cmd_judge_validate)

    w = sub.add_parser("word-freq", help="Table-3 differential words")
    w.add_argument("--model", required=True)
    w.add_argument("--top-n", type=int, default=20)
    w.add_argument("--method", default="logodds", choices=["logodds", "ratio"])
    w.set_defaults(func=cmd_word_freq)

    pf = sub.add_parser("prefill", help="Section-3 prefill experiment")
    pf.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-it", "gemma-3-27b-pt"])
    pf.set_defaults(func=cmd_prefill)

    sub.add_parser("prefill-agg").set_defaults(func=cmd_prefill_agg)

    gc = sub.add_parser("gen-calm", help="generate calm finetuning data")
    gc.add_argument("--persona", default="calm", choices=["calm", "teacher"])
    gc.add_argument("--n-per-turncount", type=int, default=400)
    gc.set_defaults(func=cmd_gen_calm)

    gf = sub.add_parser("gen-frustrated", help="generate frustrated data")
    gf.add_argument("--n-per-turncount", type=int, default=200)
    gf.set_defaults(func=cmd_gen_frustrated)

    bd = sub.add_parser("build-dpo", help="build + save 280 DPO pairs")
    bd.add_argument("--out", default=None)
    bd.set_defaults(func=cmd_build_dpo)

    bs = sub.add_parser("build-sft", help="build + save SFT corpus")
    bs.add_argument("--out", default=None)
    bs.set_defaults(func=cmd_build_sft)

    td = sub.add_parser("train-dpo", help="train LoRA DPO adapter")
    td.add_argument("--out", required=True)
    td.add_argument("--pairs", default=None, help="prebuilt pairs json")
    td.add_argument("--layers", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"), help="restrict LoRA to layers [LO,HI)")
    td.set_defaults(func=cmd_train_dpo)

    ts = sub.add_parser("train-sft", help="train LoRA SFT adapter")
    ts.add_argument("--out", required=True)
    ts.add_argument("--examples", default=None, help="prebuilt examples json")
    ts.set_defaults(func=cmd_train_sft)

    pt = sub.add_parser("petri", help="open-ended emotion elicitation")
    pt.add_argument("--models", nargs="+", default=["gemma-3-27b-it"])
    pt.add_argument("--per-emotion", type=int,
                    default=config.PETRI.transcripts_per_emotion)
    pt.set_defaults(func=cmd_petri)

    sub.add_parser("petri-agg").set_defaults(func=cmd_petri_agg)

    cap = sub.add_parser("capabilities", help="capability benchmarks")
    cap.add_argument("--models", nargs="+", required=True)
    cap.add_argument("--n", type=int, default=100)
    cap.set_defaults(func=cmd_capabilities)

    cc = sub.add_parser("cap-compare", help="accuracy deltas vs baseline")
    cc.add_argument("--baseline", required=True)
    cc.add_argument("--finetuned", nargs="+", required=True)
    cc.set_defaults(func=cmd_cap_compare)

    la = sub.add_parser("layer-ablation", help="Appendix-I layer ablation")
    la.add_argument("--pairs", default=None)
    la.set_defaults(func=cmd_layer_ablation)

    sub.add_parser("figures").set_defaults(func=cmd_figures)
    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
