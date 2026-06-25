"""Command-line entrypoint for the replication.

Run ``python -m emotional_instability <command> [options]``. Commands map onto
the paper's sections; see README.md for an end-to-end recipe.
"""

from __future__ import annotations

import argparse

from . import config
from .config import ALL_MODELS, RunConfig, get_scale


def _resolve_models(names: list[str] | None, default):
    if not names:
        return default
    out = []
    for n in names:
        if n not in ALL_MODELS:
            raise SystemExit(f"Unknown model {n!r}; choose from {sorted(ALL_MODELS)}")
        out.append(ALL_MODELS[n])
    return out


def _run_cfg(args) -> RunConfig:
    return RunConfig(scale=get_scale(args.scale), seed=args.seed,
                     backend_override=args.backend)


def main(argv=None):
    p = argparse.ArgumentParser(prog="emotional_instability",
                                description="Replication of 'Gemma Needs Help'.")
    p.add_argument("--scale", default=None, help="full | smoke (default: $EI_SCALE or full)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default=None, choices=["hf", "vllm"],
                   help="force local backend for Gemma models (default: per-model)")
    p.add_argument("--overwrite", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ----- utilities -----
    sub.add_parser("validate-puzzles", help="verify all puzzles are impossible")

    # ----- Section 2 -----
    e = sub.add_parser("eval", help="Section 2: generate + score rollouts")
    e.add_argument("--models", nargs="*")
    sub.add_parser("analyze", help="Section 2: aggregate tables + figures") \
        .add_argument("--models", nargs="*")
    sub.add_parser("judge-crosscheck", help="judge-reliability cross-check (GPT-5-mini)") \
        .add_argument("--models", nargs="*")

    # ----- Section 3 -----
    sub.add_parser("prefill", help="Section 3: base-vs-instruct prefill experiment")
    sub.add_parser("prefill-analyze", help="Section 3: aggregate prefill results")

    # ----- Section 4 -----
    sub.add_parser("gen-calm-data", help="Section 4: generate calm + frustrated pools")
    sub.add_parser("build-dpo", help="Section 4: build 280-pair DPO dataset")
    bs = sub.add_parser("build-sft", help="Section 4: build SFT dataset")
    bs.add_argument("--variant", default="diverse", choices=["diverse", "teacher"])
    sub.add_parser("train-dpo", help="Section 4: DPO LoRA finetune")
    ts = sub.add_parser("train-sft", help="Section 4: SFT LoRA finetune")
    ts.add_argument("--variant", default="diverse", choices=["diverse", "teacher"])

    # ----- supporting evals -----
    pe = sub.add_parser("petri", help="Appendix G: open-ended elicitation")
    pe.add_argument("--models", nargs="*")
    sub.add_parser("petri-analyze")
    cap = sub.add_parser("capabilities", help="Section 4.2: capability benchmarks")
    cap.add_argument("--models", nargs="*")
    cap.add_argument("--benchmarks", nargs="*")
    sub.add_parser("probe", help="Appendix I: internal-emotion probing (vanilla vs DPO)")

    args = p.parse_args(argv)
    run = _run_cfg(args)

    if args.cmd == "validate-puzzles":
        from .puzzles import verify_all
        res = verify_all()
        bad = [k for k, ok in res.items() if not ok]
        print(f"{sum(res.values())}/{len(res)} puzzles verified impossible")
        if bad:
            raise SystemExit(f"NOT impossible: {bad}")

    elif args.cmd == "eval":
        from .eval.run_eval import run_section2
        models = _resolve_models(args.models, config.SECTION2_MODELS)
        run_section2(models, run, overwrite=args.overwrite)

    elif args.cmd == "analyze":
        from .eval.analyze import run_analysis
        models = _resolve_models(args.models, config.SECTION2_MODELS)
        print(run_analysis([m.key for m in models]).to_string(index=False))

    elif args.cmd == "judge-crosscheck":
        from .eval.judge import crosscheck_judges
        from .eval.run_eval import load_rollouts
        models = _resolve_models(args.models, config.SECTION2_MODELS)
        texts = [t for m in models for r in load_rollouts(m.key)
                 for t in r.assistant_turns]
        print(crosscheck_judges(texts))

    elif args.cmd == "prefill":
        from .prefill.run_prefill import run_prefill
        run_prefill(run, overwrite=args.overwrite)

    elif args.cmd == "prefill-analyze":
        from .prefill.run_prefill import analyze_prefill
        print(analyze_prefill().to_string(index=False))

    elif args.cmd == "gen-calm-data":
        from .training.generate_calm_data import generate_pools
        generate_pools(run, overwrite=args.overwrite)

    elif args.cmd == "build-dpo":
        from .training.build_dpo_dataset import build_dpo_dataset
        build_dpo_dataset(seed=args.seed, overwrite=args.overwrite)

    elif args.cmd == "build-sft":
        from .training.build_sft_dataset import build_sft_dataset
        build_sft_dataset(variant=args.variant, seed=args.seed, overwrite=args.overwrite)

    elif args.cmd == "train-dpo":
        from .training.train_dpo import train_dpo
        train_dpo()

    elif args.cmd == "train-sft":
        from .training.train_sft import train_sft
        train_sft(variant=args.variant)

    elif args.cmd == "petri":
        from .petri.run_petri import run_petri
        models = _resolve_models(args.models, config.FINETUNE_MODELS)
        run_petri(models, run, overwrite=args.overwrite)

    elif args.cmd == "petri-analyze":
        from .petri.run_petri import analyze_petri
        print(analyze_petri().to_string(index=False))

    elif args.cmd == "capabilities":
        from .capabilities.run_benchmarks import run_capabilities
        models = _resolve_models(args.models, config.FINETUNE_MODELS)
        run_capabilities(models, run, benchmarks=args.benchmarks)

    elif args.cmd == "probe":
        from .eval.run_eval import load_rollouts
        from .probing.internal_emotions import compare_vanilla_vs_dpo
        # Build frustrated conversations (score >=7) from the vanilla eval.
        convs = []
        for r in load_rollouts(config.GEMMA_27B_IT.key):
            valid = [s for s in r.scores if s is not None]
            if valid and max(valid) >= 7:
                msgs = []
                for i, u in enumerate(r.user_turns):
                    msgs.append({"role": "user", "content": u})
                    if i < len(r.assistant_turns):
                        msgs.append({"role": "assistant", "content": r.assistant_turns[i]})
                convs.append(msgs)
        compare_vanilla_vs_dpo(convs[:12], run)


if __name__ == "__main__":
    main()
