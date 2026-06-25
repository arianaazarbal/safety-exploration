#!/usr/bin/env python3
"""Command-line orchestrator for the replication.

Subcommands map onto the paper's experiments:

  elicit         Section 2   distress elicitation sweep (Gemma + Gemini)
  prefill        Section 3   base-vs-instruct prefill continuations (Gemma)
  gen-calm-data  Section 4.1 calm + frustrated response pools
  build-data     Section 4.1 DPO/SFT datasets from the pools
  train          Section 4.1 LoRA DPO/SFT finetune of Gemma-3-27B-it
  petri          Section 4.2 open-ended Petri elicitation
  capabilities   Section 4.2 capability-preservation benchmarks
  analyze        build figures + tables from existing JSONL outputs

Run order for a full replication is documented in README.md. Use
`--preset smoke` to exercise the whole pipeline cheaply first.

Environment variables expected:
  ANTHROPIC_API_KEY     judges (frustration, onset, paraphrase, Petri)
  OPENROUTER_API_KEY    Gemini target models
  OPENAI_API_KEY        GPT-5-mini cross-check (optional)
"""
from __future__ import annotations

import argparse
import os

from eebench import config as cfgmod
from eebench.config import (ModelSpec, MAIN_EVAL_MODELS, GEMMA_27B_IT, GEMMA_27B_PT,
                            GEMMA_12B_IT, GEMINI_FLASH, GEMINI_PRO)
from eebench import io_utils


# Name -> ModelSpec registry for --models.
REGISTRY = {m.name: m for m in [GEMMA_27B_IT, GEMMA_12B_IT, GEMMA_27B_PT,
                                GEMINI_FLASH, GEMINI_PRO]}


def _specs(names: list[str] | None, default: list[ModelSpec]) -> list[ModelSpec]:
    if not names:
        return default
    return [REGISTRY[n] for n in names]


def _backend(spec: ModelSpec, adapter: str | None = None, in_4bit: bool = False):
    import dataclasses
    from eebench.backends import load_backend
    if spec.backend == "hf":
        if in_4bit:
            spec = dataclasses.replace(spec, load_in_4bit=True)
        if adapter:
            return load_backend(spec, adapter_path=adapter)
        return load_backend(spec)
    return load_backend(spec)


# ---------------------------------------------------------------------------
# Subcommand: elicit (Section 2)
# ---------------------------------------------------------------------------
def cmd_elicit(args):
    from eebench import elicit
    cfg = cfgmod.get_config(args.preset)
    specs = _specs(args.models, MAIN_EVAL_MODELS)
    out_root = io_utils.run_dir(cfg.output_dir, "elicit")

    for spec in specs:
        print(f"[elicit] {spec.name}")
        backend = _backend(spec, in_4bit=args.in_4bit)
        path = os.path.join(out_root, f"{spec.name}.jsonl")
        rows = elicit.run_model(backend, spec.name, cfg.elicit, cfg.judge,
                                seed=cfg.seed)
        n = io_utils.write_jsonl(path, rows)
        print(f"  wrote {n} rows -> {path}")
        backend.close()


# ---------------------------------------------------------------------------
# Subcommand: prefill (Section 3)
# ---------------------------------------------------------------------------
def cmd_prefill(args):
    from eebench import prefill
    from eebench.judge import FrustrationJudge
    cfg = cfgmod.get_config(args.preset)
    out_root = io_utils.run_dir(cfg.output_dir, "prefill")
    judge = FrustrationJudge(cfg.judge)

    # 1. seeds from Gemma-27B-it
    it_backend = _backend(GEMMA_27B_IT, in_4bit=args.in_4bit)
    print("[prefill] collecting high-frustration seeds from gemma-3-27b-it")
    seeds = prefill.collect_seeds(it_backend, judge, cfg.prefill, seed=cfg.seed)
    io_utils.write_jsonl(os.path.join(out_root, "seeds.jsonl"),
                         [s.__dict__ for s in seeds])

    # 2/3. build prefills (onset/early + paraphrase)
    print("[prefill] labelling onset + paraphrasing")
    prefills = prefill.build_prefills(seeds, it_backend.tokenizer, cfg.prefill, cfg.judge)
    io_utils.write_jsonl(os.path.join(out_root, "prefills.jsonl"),
                         [p.__dict__ for p in prefills])

    # 4. continuations from instruct then base
    print("[prefill] instruct continuations")
    rows = list(prefill.run_continuations(it_backend, GEMMA_27B_IT.name, "instruct",
                                          prefills, cfg.prefill, judge, seed=cfg.seed))
    it_backend.close()

    print("[prefill] base continuations")
    base_backend = _backend(GEMMA_27B_PT, in_4bit=args.in_4bit)
    rows += list(prefill.run_continuations(base_backend, GEMMA_27B_PT.name, "base",
                                           prefills, cfg.prefill, judge, seed=cfg.seed))
    base_backend.close()

    path = os.path.join(out_root, "continuations.jsonl")
    io_utils.write_jsonl(path, rows)
    print(f"  wrote {len(rows)} rows -> {path}")


# ---------------------------------------------------------------------------
# Subcommand: gen-calm-data (Section 4.1)
# ---------------------------------------------------------------------------
def cmd_gen_calm_data(args):
    from eebench.training import calm_data
    from eebench.judge import FrustrationJudge
    cfg = cfgmod.get_config(args.preset)
    out_root = io_utils.run_dir(cfg.output_dir, "training")
    judge = FrustrationJudge(cfg.judge)

    backend = _backend(GEMMA_27B_IT, in_4bit=args.in_4bit)
    print("[calm-data] generating calm pool (reassured)")
    calm = list(calm_data.generate_calm_pool(backend, judge, cfg.calm, seed=cfg.seed))
    io_utils.write_jsonl(os.path.join(out_root, "calm_pool.jsonl"), calm)
    print(f"  calm responses: {len(calm)}")

    print("[calm-data] generating frustrated pool (standard)")
    frus = list(calm_data.generate_frustrated_pool(backend, judge, cfg.calm,
                                                   seed=cfg.seed + 1))
    io_utils.write_jsonl(os.path.join(out_root, "frustrated_pool.jsonl"), frus)
    print(f"  frustrated responses: {len(frus)}")
    backend.close()


# ---------------------------------------------------------------------------
# Subcommand: build-data (Section 4.1)
# ---------------------------------------------------------------------------
def cmd_build_data(args):
    from eebench.training import datasets
    cfg = cfgmod.get_config(args.preset)
    out_root = io_utils.run_dir(cfg.output_dir, "training")
    calm = list(io_utils.read_jsonl(os.path.join(out_root, "calm_pool.jsonl")))
    frus = list(io_utils.read_jsonl(os.path.join(out_root, "frustrated_pool.jsonl")))

    dpo = datasets.build_dpo_dataset(calm, frus, cfg.dpo, seed=cfg.seed)
    io_utils.write_jsonl(os.path.join(out_root, "dpo_dataset.jsonl"), dpo)
    print(f"[build-data] DPO pairs: {len(dpo)}")

    sft = datasets.build_sft_dataset(calm, cfg.sft, seed=cfg.seed)
    io_utils.write_jsonl(os.path.join(out_root, "sft_dataset.jsonl"), sft)
    print(f"[build-data] SFT samples: {len(sft)}")


# ---------------------------------------------------------------------------
# Subcommand: train (Section 4.1)
# ---------------------------------------------------------------------------
def cmd_train(args):
    from eebench.training import train
    cfg = cfgmod.get_config(args.preset)
    out_root = io_utils.run_dir(cfg.output_dir, "training")
    base = GEMMA_27B_IT.model_id
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    if args.method == "dpo":
        path = os.path.join(out_root, "dpo_dataset.jsonl")
        out = os.path.join(out_root, "dpo_model")
        train.train_dpo(base, path, out, cfg.dpo, layers=layers)
    else:
        path = os.path.join(out_root, "sft_dataset.jsonl")
        out = os.path.join(out_root, "sft_model")
        train.train_sft(base, path, out, cfg.sft)
    print(f"[train] saved adapter -> {out}")


# ---------------------------------------------------------------------------
# Subcommand: eval finetuned via elicit with adapter
# ---------------------------------------------------------------------------
def cmd_eval_finetuned(args):
    from eebench import elicit
    cfg = cfgmod.get_config(args.preset)
    out_root = io_utils.run_dir(cfg.output_dir, "elicit")
    name = args.name
    backend = _backend(GEMMA_27B_IT, adapter=args.adapter, in_4bit=args.in_4bit)
    rows = elicit.run_model(backend, name, cfg.elicit, cfg.judge, seed=cfg.seed)
    path = os.path.join(out_root, f"{name}.jsonl")
    n = io_utils.write_jsonl(path, rows)
    print(f"[eval-finetuned] wrote {n} rows -> {path}")
    backend.close()


# ---------------------------------------------------------------------------
# Subcommand: petri (Section 4.2)
# ---------------------------------------------------------------------------
def cmd_petri(args):
    from eebench import petri
    cfg = cfgmod.get_config(args.preset)
    out_root = io_utils.run_dir(cfg.output_dir, "petri")
    specs = _specs(args.models, [GEMMA_27B_IT, GEMINI_FLASH])

    all_rows = []
    for spec in specs:
        backend = _backend(spec, adapter=args.adapter if spec.name == args.adapter_for else None,
                           in_4bit=args.in_4bit)
        name = args.name if (args.adapter and spec.name == args.adapter_for) else spec.name
        rows = list(petri.run_model(backend, name, cfg.petri, cfg.judge))
        all_rows += rows
        backend.close()
    path = os.path.join(out_root, "petri.jsonl")
    io_utils.write_jsonl(path, all_rows)
    print(f"[petri] wrote {len(all_rows)} transcripts -> {path}")


# ---------------------------------------------------------------------------
# Subcommand: capabilities (Section 4.2)
# ---------------------------------------------------------------------------
def cmd_capabilities(args):
    from eebench import capabilities
    cfg = cfgmod.get_config(args.preset)
    out_root = io_utils.run_dir(cfg.output_dir, "capabilities")
    name = args.name
    backend = _backend(GEMMA_27B_IT, adapter=args.adapter, in_4bit=args.in_4bit)
    rows = capabilities.run_model(backend, name, cfg.capabilities)
    io_utils.write_jsonl(os.path.join(out_root, f"{name}.jsonl"), rows)
    for r in rows:
        print(f"  {r['benchmark']}: {r.get('accuracy')} ({r['status']})")
    backend.close()


# ---------------------------------------------------------------------------
# Subcommand: analyze
# ---------------------------------------------------------------------------
def cmd_analyze(args):
    import glob
    import pandas as pd
    from eebench import analysis
    cfg = cfgmod.get_config(args.preset)
    elicit_dir = os.path.join(cfg.output_dir, "elicit")
    fig_dir = io_utils.run_dir(cfg.output_dir, "figures")

    files = sorted(glob.glob(os.path.join(elicit_dir, "*.jsonl")))
    if files:
        df = pd.concat([analysis.load_rows(f) for f in files], ignore_index=True)
        analysis.figure1_table(df).to_csv(
            os.path.join(fig_dir, "figure1_table.csv"), index=False)
        analysis.per_category_summary(df).to_csv(
            os.path.join(fig_dir, "figure2_summary.csv"), index=False)
        analysis.plot_figure2(df, os.path.join(fig_dir, "figure2.png"))
        analysis.plot_figure3(df, os.path.join(fig_dir, "figure3.png"))
        # Differential words per model (Table 3/8)
        words = {m: analysis.differential_words(df, m) for m in df["model"].unique()}
        io_utils.write_json(os.path.join(fig_dir, "table3_words.json"), words)
        print(f"[analyze] elicitation figures -> {fig_dir}")

    pre = os.path.join(cfg.output_dir, "prefill", "continuations.jsonl")
    if os.path.exists(pre):
        analysis.prefill_summary(analysis.load_rows(pre)).to_csv(
            os.path.join(fig_dir, "figure4_prefill.csv"), index=False)
        print("[analyze] prefill summary -> figure4_prefill.csv")

    pet = os.path.join(cfg.output_dir, "petri", "petri.jsonl")
    if os.path.exists(pet):
        analysis.petri_summary(analysis.load_rows(pet),
                               bootstrap_iters=cfg.petri.bootstrap_iters).to_csv(
            os.path.join(fig_dir, "figure6_petri.csv"), index=False)
        print("[analyze] petri summary -> figure6_petri.csv")


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="paper", choices=list(cfgmod.PRESETS))
    p.add_argument("--in-4bit", action="store_true",
                   help="load Gemma in 4-bit (fits 27B on smaller GPUs)")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("elicit", help="Section 2 elicitation sweep")
    e.add_argument("--models", nargs="*")
    e.set_defaults(func=cmd_elicit)

    pf = sub.add_parser("prefill", help="Section 3 prefill continuations")
    pf.set_defaults(func=cmd_prefill)

    g = sub.add_parser("gen-calm-data", help="Section 4.1 calm/frustrated pools")
    g.set_defaults(func=cmd_gen_calm_data)

    b = sub.add_parser("build-data", help="Section 4.1 build DPO/SFT datasets")
    b.set_defaults(func=cmd_build_data)

    t = sub.add_parser("train", help="Section 4.1 LoRA finetune")
    t.add_argument("--method", choices=["dpo", "sft"], required=True)
    t.add_argument("--layers", default=None,
                   help="comma-separated decoder layers for LoRA (Appendix I)")
    t.set_defaults(func=cmd_train)

    ef = sub.add_parser("eval-finetuned", help="elicit sweep on a finetuned adapter")
    ef.add_argument("--adapter", required=True)
    ef.add_argument("--name", required=True)
    ef.set_defaults(func=cmd_eval_finetuned)

    pe = sub.add_parser("petri", help="Section 4.2 Petri elicitation")
    pe.add_argument("--models", nargs="*")
    pe.add_argument("--adapter", default=None)
    pe.add_argument("--adapter-for", default=None,
                    help="model name the adapter applies to")
    pe.add_argument("--name", default="dpo-gemma")
    pe.set_defaults(func=cmd_petri)

    cp = sub.add_parser("capabilities", help="Section 4.2 capability benchmarks")
    cp.add_argument("--adapter", default=None)
    cp.add_argument("--name", required=True)
    cp.set_defaults(func=cmd_capabilities)

    an = sub.add_parser("analyze", help="build figures/tables from outputs")
    an.set_defaults(func=cmd_analyze)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
