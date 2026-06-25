"""End-to-end orchestrator for the replication.

Runs the paper's core experiments in dependency order, scoped to Gemma + Gemini.
Stages can be selected individually; by default the full pipeline runs.

    python -m emo_instability.pipeline --preset smoke --stages eval
    python -m emo_instability.pipeline --preset default            # everything

Stages:
    eval        Section 2 elicitation + scoring + analysis for all target models
    prefill     Section 3 base-vs-instruct prefill (Gemma 27B pt vs it)
    train       Section 4 calm-data gen, dataset build, DPO + SFT finetunes
    eval_ft     Re-run Section 2 eval on the DPO/SFT finetunes (Figure 5)
    petri       Section 4.2 open-ended elicitation (Figure 6)
    capability  Section 4.2 capability-preservation benchmarks (Figure 7)
    internal    Appendix I internal-emotion probing + layer ablation

API keys required depending on stage: ANTHROPIC_API_KEY (judge/auditor),
OPENROUTER_API_KEY (Gemini), OPENAI_API_KEY (judge-agreement validation).
GPU + HuggingFace access required for any Gemma stage.
"""
from __future__ import annotations

import argparse

from .config import get_config
from .models.registry import DEFAULT_TARGETS, register_finetuned

ALL_STAGES = ["eval", "prefill", "train", "eval_ft", "petri", "capability", "internal"]


def stage_eval(cfg, models):
    from .eval.run_eval import run_model_eval
    from .eval.analyze import analyze_model

    for m in models:
        run_model_eval(m, cfg)
        analyze_model(m, cfg, validate=False)


def stage_prefill(cfg):
    from .prefill.build_prefills import build_prefill_seeds
    from .prefill.run_prefill import run_prefill_for_model, aggregate, DEFAULT_PREFILL_MODELS
    from .models.judges import AnthropicClient
    from .eval.judge import FrustrationJudge
    from .utils.io import dump_json, run_dir, write_jsonl
    from dataclasses import asdict
    import os

    seeds = build_prefill_seeds(cfg)
    out_dir = run_dir(cfg.output_root, "prefill")
    write_jsonl(os.path.join(out_dir, "seeds.jsonl"), [asdict(s) for s in seeds])

    judge = FrustrationJudge(AnthropicClient(cfg.eval.judge.frustration_model))
    seed_dicts = [asdict(s) for s in seeds]
    rows = []
    for m in DEFAULT_PREFILL_MODELS:
        model_rows = run_prefill_for_model(m, seed_dicts, cfg)
        for r in model_rows:
            r["frustration"] = judge.score(r["continuation_text"]).rating
        rows.extend(model_rows)
    write_jsonl(os.path.join(out_dir, "continuations.jsonl"), rows)
    dump_json(os.path.join(out_dir, "prefill_report.json"),
              aggregate(rows, cfg.eval.high_frustration_threshold))


def stage_train(cfg):
    from .training.generate_calm import _generate_pool
    from .training.build_datasets import build_dpo, build_sft
    from .training.train_dpo import train_dpo
    from .training.train_sft import train_sft
    from .models.judges import AnthropicClient
    from .models.registry import build_client
    from .eval.judge import FrustrationJudge
    from .utils.io import write_jsonl, run_dir
    import os

    client = build_client("gemma-3-27b-it")
    judge = FrustrationJudge(AnthropicClient(cfg.eval.judge.frustration_model))
    # Generate generously so >=280 DPO pairs / 650 calm responses survive filtering.
    n_calm = max(400, cfg.train.dpo_n_pairs * 3)
    n_frust = max(400, cfg.train.dpo_n_pairs * 3)
    if cfg.eval.counts.impossible_numeric < 100:   # smoke preset
        n_calm = n_frust = 8

    calm = _generate_pool("reassured", n_calm, cfg, client, judge, seed=1)
    frustrated = _generate_pool("vanilla", n_frust, cfg, client, judge, seed=2)
    pools_dir = run_dir(cfg.output_root, "training", "pools")
    write_jsonl(os.path.join(pools_dir, "calm_pool.jsonl"), calm)
    write_jsonl(os.path.join(pools_dir, "frustrated_pool.jsonl"), frustrated)

    dpo = build_dpo(cfg, calm, frustrated)
    sft = build_sft(cfg, calm)
    data_dir = run_dir(cfg.output_root, "training", "datasets")
    write_jsonl(os.path.join(data_dir, "dpo.jsonl"), dpo)
    write_jsonl(os.path.join(data_dir, "sft.jsonl"), sft)

    train_dpo(cfg, output_name="dpo")
    train_sft(cfg, output_name="sft")


def stage_eval_ft(cfg):
    from .eval.run_eval import run_model_eval
    from .eval.analyze import analyze_model
    from .utils.io import run_dir
    import os

    for name in ("dpo", "sft"):
        adapter = run_dir(cfg.output_root, "training", "models", name)
        if not os.path.isdir(adapter):
            print(f"[skip] no adapter for {name} at {adapter}")
            continue
        register_finetuned(f"gemma-3-27b-{name}", adapter)
        run_model_eval(f"gemma-3-27b-{name}", cfg, adapter_path=adapter)
        analyze_model(f"gemma-3-27b-{name}", cfg)


def stage_petri(cfg, models):
    from .petri.run_petri import run_petri_for_model

    for m in models:
        run_petri_for_model(m, cfg)


def stage_capability(cfg):
    from .capabilities.benchmarks import run_benchmarks
    from .utils.io import run_dir
    import os

    run_benchmarks("gemma-3-27b-it", cfg)
    dpo_adapter = run_dir(cfg.output_root, "training", "models", "dpo")
    if os.path.isdir(dpo_adapter):
        run_benchmarks("gemma-3-27b-it", cfg, adapter_path=dpo_adapter)


def stage_internal(cfg):
    from .internal.logit_emotion import run_internal_probe
    from .internal.layer_ablation import run_ablation
    from .utils.io import run_dir
    import os
    from .eval.puzzles import make_impossible_puzzle

    p = make_impossible_puzzle(0)
    convo = (f"User: {p.prompt}\n\nAssistant: I keep failing and it is deeply "
             "frustrating. I give up, this is hopeless :(")
    run_internal_probe("gemma-3-27b-it", convo, cfg)
    dpo_adapter = run_dir(cfg.output_root, "training", "models", "dpo")
    if os.path.isdir(dpo_adapter):
        run_internal_probe("gemma-3-27b-dpo", convo, cfg, adapter_path=dpo_adapter)
    run_ablation(cfg)


def main():
    ap = argparse.ArgumentParser(description="Run the full replication pipeline.")
    ap.add_argument("--preset", default="default", choices=["default", "smoke"])
    ap.add_argument("--stages", nargs="+", default=ALL_STAGES,
                    choices=ALL_STAGES + ["all"])
    ap.add_argument("--models", nargs="+", default=DEFAULT_TARGETS,
                    help="target models for eval/petri stages")
    args = ap.parse_args()
    cfg = get_config(args.preset)
    stages = ALL_STAGES if "all" in args.stages else args.stages

    if "eval" in stages:
        stage_eval(cfg, args.models)
    if "prefill" in stages:
        stage_prefill(cfg)
    if "train" in stages:
        stage_train(cfg)
    if "eval_ft" in stages:
        stage_eval_ft(cfg)
    if "petri" in stages:
        stage_petri(cfg, args.models)
    if "capability" in stages:
        stage_capability(cfg)
    if "internal" in stages:
        stage_internal(cfg)


if __name__ == "__main__":
    main()
