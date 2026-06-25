"""Command-line entry point: `python -m gnh.cli <command> [...]`.

Commands mirror the paper's experiments:
  eval          Section 2 propensity eval for a model
  ablations     Appendix A control ablations (Gemma)
  reliability   Section 2.1 judge-agreement check
  prefill       Section 3 base-vs-instruct continuation experiment (Gemma)
  recovery      Section 4.2 recovery-from-frustration experiment (Gemma)
  gen-calm      Section 4.1 generate calm/frustrated response pools (Gemma)
  build-data    Build DPO (280 pairs) and SFT datasets
  train         LoRA DPO/SFT finetune (Gemma)
  petri         Appendix G open-ended elicitation
  capabilities  Figure 7 capability-preservation benchmarks
  analyze       Print headline metrics / tables from result files
  figures       Render figures from result files
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config


def _add_common(p):
    p.add_argument("--profile", default=None, help="full | smoke | reduced")
    p.add_argument("--seed", type=int, default=0)


def cmd_eval(a):
    from .eval.runner import run_model_eval

    profile = config.get_profile(a.profile)
    path = run_model_eval(
        a.model, profile=profile, seed=a.seed, backend_key=a.backend,
        judge_workers=a.judge_workers, gen_workers=a.gen_workers,
    )
    print(f"wrote {path}")


def cmd_ablations(a):
    from .eval.ablations import run_ablations

    print(f"wrote {run_ablations(a.model, n_conversations=a.n, seed=a.seed)}")


def cmd_reliability(a):
    from .eval.reliability import reliability_check

    res = reliability_check(Path(a.eval_jsonl), n=a.n, seed=a.seed)
    print(json.dumps(res, indent=2))


def cmd_prefill(a):
    from .prefill import run_prefill_experiment

    print(f"wrote {run_prefill_experiment(n_each=a.n_each, n_cont=a.n_cont, seed=a.seed)}")


def cmd_recovery(a):
    from .prefill import run_recovery_experiment

    print(f"wrote {run_recovery_experiment(n_each=a.n_each, n_cont=a.n_cont, seed=a.seed)}")


def cmd_gen_calm(a):
    from .training import generate_response_pool

    paths = generate_response_pool(n_per_count=a.n_per_count, seed=a.seed)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


def cmd_build_data(a):
    from .training import build_dpo_dataset, build_sft_dataset

    if a.kind in ("dpo", "both"):
        print(f"DPO -> {build_dpo_dataset(seed=a.seed)}")
    if a.kind in ("sft", "both"):
        print(f"SFT -> {build_sft_dataset(seed=a.seed)}")


def cmd_train(a):
    from .training.train import train_dpo, train_sft

    out = Path(a.output_dir or (config.ADAPTER_DIR / a.method))
    layers = config.LAYER_ABLATIONS.get(a.layers) if a.layers else None
    if a.method == "dpo":
        data = a.dataset or (config.DATA_DIR / "dpo_pairs.jsonl")
        print(f"adapter -> {train_dpo(Path(data), out, layers=layers, load_in_4bit=a.load_in_4bit)}")
    else:
        data = a.dataset or (config.DATA_DIR / "sft_dataset.jsonl")
        print(f"adapter -> {train_sft(Path(data), out, load_in_4bit=a.load_in_4bit)}")


def cmd_petri(a):
    from .petri import run_petri

    print(f"wrote {run_petri(a.models, n_per_emotion=a.n, seed=a.seed)}")


def cmd_capabilities(a):
    from .capabilities import run_capabilities

    print(f"wrote {run_capabilities(a.models, n_per_benchmark=a.n)}")


def cmd_analyze(a):
    from .analysis import (
        category_summary, differential_words, headline_metric, load_eval,
        model_comparison_table,
    )

    if a.compare:
        paths = {Path(p).stem.replace("eval_", ""): p for p in a.compare}
        print(model_comparison_table(paths).to_string(index=False))
        return
    df = load_eval(a.eval_jsonl)
    print("Headline:", json.dumps(headline_metric(df), indent=2))
    print("\nPer category:\n", category_summary(df).to_string())
    if a.words:
        print("\nDifferential words:")
        for w, e in differential_words(df):
            print(f"  {w:>16}  {e:.2f}x")


def cmd_figures(a):
    from .analysis import model_comparison_table
    from .analysis.plots import (
        plot_category_breakdown, plot_model_comparison, plot_per_turn,
    )

    paths = {Path(p).stem.replace("eval_", "").split("_")[0]: p for p in a.eval}
    table = model_comparison_table(paths)
    print(plot_model_comparison(table))
    print(plot_category_breakdown(paths))
    for cond in ("extended_8turn", "wildchat_5turn"):
        print(plot_per_turn(paths, cond))
    if a.petri:
        from .analysis.plots import plot_petri
        print(plot_petri(a.petri))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gnh", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("eval"); _add_common(e)
    e.add_argument("model"); e.add_argument("--backend", default=None)
    e.add_argument("--gen-workers", type=int, default=None, dest="gen_workers")
    e.add_argument("--judge-workers", type=int, default=8, dest="judge_workers")
    e.set_defaults(func=cmd_eval)

    ab = sub.add_parser("ablations"); _add_common(ab)
    ab.add_argument("--model", default="gemma-3-27b-it"); ab.add_argument("--n", type=int, default=40)
    ab.set_defaults(func=cmd_ablations)

    rel = sub.add_parser("reliability"); _add_common(rel)
    rel.add_argument("eval_jsonl"); rel.add_argument("--n", type=int, default=260)
    rel.set_defaults(func=cmd_reliability)

    pf = sub.add_parser("prefill"); _add_common(pf)
    pf.add_argument("--n-each", type=int, default=10, dest="n_each")
    pf.add_argument("--n-cont", type=int, default=50, dest="n_cont")
    pf.set_defaults(func=cmd_prefill)

    rc = sub.add_parser("recovery"); _add_common(rc)
    rc.add_argument("--n-each", type=int, default=10, dest="n_each")
    rc.add_argument("--n-cont", type=int, default=50, dest="n_cont")
    rc.set_defaults(func=cmd_recovery)

    gc = sub.add_parser("gen-calm"); _add_common(gc)
    gc.add_argument("--n-per-count", type=int, default=400, dest="n_per_count")
    gc.set_defaults(func=cmd_gen_calm)

    bd = sub.add_parser("build-data"); _add_common(bd)
    bd.add_argument("--kind", choices=["dpo", "sft", "both"], default="both")
    bd.set_defaults(func=cmd_build_data)

    tr = sub.add_parser("train")
    tr.add_argument("--method", choices=["dpo", "sft"], required=True)
    tr.add_argument("--dataset", default=None); tr.add_argument("--output-dir", default=None, dest="output_dir")
    tr.add_argument("--layers", default=None, help="layer-ablation key, e.g. 30-35")
    tr.add_argument("--load-in-4bit", action="store_true", dest="load_in_4bit")
    tr.set_defaults(func=cmd_train)

    pt = sub.add_parser("petri"); _add_common(pt)
    pt.add_argument("models", nargs="+"); pt.add_argument("--n", type=int, default=10)
    pt.set_defaults(func=cmd_petri)

    cap = sub.add_parser("capabilities")
    cap.add_argument("models", nargs="+"); cap.add_argument("--n", type=int, default=100)
    cap.set_defaults(func=cmd_capabilities)

    an = sub.add_parser("analyze")
    an.add_argument("eval_jsonl", nargs="?"); an.add_argument("--compare", nargs="+")
    an.add_argument("--words", action="store_true")
    an.set_defaults(func=cmd_analyze)

    fg = sub.add_parser("figures")
    fg.add_argument("eval", nargs="+"); fg.add_argument("--petri", default=None)
    fg.set_defaults(func=cmd_figures)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
