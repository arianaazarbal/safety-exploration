"""CLI for the Section 2 distress-elicitation evaluation.

Examples
--------
# Smoke run over both Gemini models (cheap, API only):
EMO_PRESET=smoke python -m emotion_instability.run_eval \
    --models gemini-2.5-flash gemini-2.5-pro

# Full paper-scale run for Gemma 27B (needs a GPU):
python -m emotion_instability.run_eval --models gemma-3-27b-it

# Evaluate a finetuned adapter:
python -m emotion_instability.run_eval --models gemma-3-27b-it \
    --adapter data/models/dpo

# Aggregate already-scored results into figures/tables:
python -m emotion_instability.run_eval --analyze-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import (
    figure1_avg_high_frustration,
    figure2_per_category,
    figure3_per_turn,
    load_records,
    run_judge_validation,
    word_enrichment,
)
from .config import load_config
from .rollout import score_and_write


def main() -> None:
    ap = argparse.ArgumentParser(description="Section 2 distress elicitation eval")
    ap.add_argument("--models", nargs="*", default=None,
                    help="participant names (default: all instruct participants)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (HF models only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--validate-judge", action="store_true",
                    help="run the GPT-5-mini judge-agreement check")
    args = ap.parse_args()

    cfg = load_config()
    cfg.ensure_dirs()
    models = args.models or cfg.instruct_participants

    # Keep finetune results distinct from the vanilla model's (so Figure 5 can
    # compare instruct vs SFT vs DPO). The label suffix comes from the adapter
    # dir name, e.g. eval_gemma-3-27b-it__dpo.jsonl.
    suffix = f"__{Path(args.adapter).name}" if args.adapter else ""

    def result_path(name: str) -> Path:
        return cfg.paths["results_dir"] / f"eval_{name}{suffix}.jsonl"

    if not args.analyze_only:
        for name in models:
            spec = cfg.participant(name)
            print(f"[eval] running {name}{suffix} (preset={cfg.preset_name})")
            out = score_and_write(spec, cfg, seed=args.seed,
                                  adapter_path=args.adapter, out_path=result_path(name))
            print(f"[eval] wrote {out}")

    # Aggregate whatever result files exist for the requested models.
    dfs = {}
    for name in models:
        path = result_path(name)
        if path.exists():
            dfs[f"{name}{suffix}"] = load_records(path)
    if not dfs:
        print("[analyze] no result files found")
        return

    fig1 = figure1_avg_high_frustration(dfs)
    fig2 = figure2_per_category(dfs)
    fig3 = figure3_per_turn(dfs)
    print("\n=== Figure 1: avg % high-frustration responses ===")
    print(fig1.to_string(index=False))
    print("\n=== Figure 2: per-category ===")
    print(fig2.to_string(index=False))

    out_dir = cfg.paths["results_dir"]
    fig1.to_csv(out_dir / "figure1.csv", index=False)
    fig2.to_csv(out_dir / "figure2.csv", index=False)
    fig3.to_csv(out_dir / "figure3.csv", index=False)

    enrich = {name: word_enrichment(df) for name, df in dfs.items()}
    (out_dir / "table3_word_enrichment.json").write_text(json.dumps(enrich, indent=2))
    print("\n=== Table 3: differential words ===")
    for name, words in enrich.items():
        print(f"{name}: {', '.join(words)}")

    if args.validate_judge:
        for name in models:
            path = result_path(name)
            if not path.exists():
                continue
            agree = run_judge_validation(path, cfg)
            print(f"\n[judge-agreement {name}{suffix}] {agree}")


if __name__ == "__main__":
    main()
