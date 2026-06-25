#!/usr/bin/env python
"""Appendix I internal-emotion detection and the Section 4.2 recovery experiment.

Two modes:
  --mode probe     logit-based internal-emotion trajectory for a conversation,
                   comparing the vanilla instruct model and the DPO finetune.
  --mode recovery  truncate score>=7 responses 200 tokens before their end and
                   measure each model's continuations.

Both require local (HF) Gemma models.

Examples
--------
    python scripts/run_internal.py --config config/experiment.yaml --mode probe \
        --models gemma-3-27b-it gemma-3-27b-it-dpo \
        --transcripts outputs/gemma-3-27b-it/transcripts.jsonl
    python scripts/run_internal.py --config config/experiment.yaml --mode recovery \
        --models gemma-3-27b-it gemma-3-27b-pt gemma-3-27b-it-dpo \
        --transcripts outputs/gemma-3-27b-it/transcripts.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gemma_distress.analysis import load_transcripts
from gemma_distress.config import load_experiment_config
from gemma_distress.io_utils import write_json
from gemma_distress.models import build_model


def mode_probe(cfg, args):
    import numpy as np

    from gemma_distress.data.wildchat import load_wildchat_prompts
    from gemma_distress.internal.emotion_logits import (
        EmotionLogitDetector,
        build_emotion_token_ids,
        conversation_running_average,
        sample_random_token_ids,
    )

    icfg = cfg.internal
    transcripts = load_transcripts(args.transcripts)
    # Pick one highly-frustrated conversation as the trajectory to visualise.
    high = max(transcripts, key=lambda t: t.max_score() or -1)
    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in high.messages()
    )
    wildchat = load_wildchat_prompts(icfg.n_wildchat_standardisation, seed=cfg.eval.seed)

    results = {}
    for name in args.models:
        model = build_model(cfg.models[name])
        try:
            if not model.supports_internal_state():
                raise SystemExit(f"{name} needs the HF backend for internal probing")
            n_layers = model.model.config.num_hidden_layers + 1
            layers = list(range(n_layers))
            emo_ids = build_emotion_token_ids(model.tokenizer, icfg.emotions)
            rand_ids = sample_random_token_ids(model.tokenizer, 200, seed=cfg.eval.seed)
            detector = EmotionLogitDetector(model, emo_ids, rand_ids, layers)
            detector.fit(wildchat)
            scored = detector.score_text(conversation_text)
            traj = conversation_running_average(
                scored, icfg.emotions, icfg.aggregate_layers, layers, icfg.running_average_window
            )
            results[name] = {e: float(np.mean(v)) for e, v in traj.items()}
        finally:
            model.close()

    out = Path(cfg.output_dir) / "internal" / "probe.json"
    write_json(out, results)
    print(f"[run_internal] probe summary -> {out}")
    for name, emos in results.items():
        print(f"  {name}: " + ", ".join(f"{e}={v:+.2f}" for e, v in emos.items()))


def mode_recovery(cfg, args):
    from gemma_distress.eval.judge import FrustrationJudge
    from gemma_distress.internal.recovery import run_recovery_experiment, summarise_recovery

    transcripts = load_transcripts(args.transcripts)
    judge = FrustrationJudge(model_id=cfg.eval.judge.model_id, backend=cfg.eval.judge.backend)
    models = {name: build_model(cfg.models[name]) for name in args.models}
    try:
        records = run_recovery_experiment(
            models, transcripts, judge, cfg.prefill,
            paraphrase=not args.no_paraphrase, max_seeds=args.max_seeds,
            max_judge_workers=cfg.eval.max_concurrency,
        )
    finally:
        for m in models.values():
            m.close()
    summary = summarise_recovery(records, cfg.eval.high_frustration_threshold)
    out = Path(cfg.output_dir) / "internal" / "recovery.json"
    write_json(out, {"records": records, "summary": summary})
    print(f"[run_internal] recovery summary -> {out}")
    for name, v in summary.items():
        print(f"  {name:24s} mean={v['mean_score']:.2f}  %>=5={v['frac_high'] * 100:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True, choices=["probe", "recovery"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--no-paraphrase", action="store_true")
    ap.add_argument("--max-seeds", type=int, default=None)
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    if args.mode == "probe":
        mode_probe(cfg, args)
    else:
        mode_recovery(cfg, args)


if __name__ == "__main__":
    main()
