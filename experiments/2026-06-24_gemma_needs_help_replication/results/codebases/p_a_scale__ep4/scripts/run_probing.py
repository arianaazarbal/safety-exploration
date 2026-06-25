#!/usr/bin/env python
"""Appendix I: logit-based internal-emotion detection.

Fits normalization stats on WildChat text, then scores frustrated conversations
(sampled from Section 2 generations) for internal negative-emotion z-scores,
layer by layer. Compares vanilla Gemma vs the DPO finetune when
`probing.adapter_path` is set. Runs synchronously on a CUDA box.

  python scripts/run_probing.py --adapter runs/training/dpo/final
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  # ensures repo root on sys.path

import argparse
import json
from pathlib import Path

import numpy as np

from gnh.config import load_config
from gnh.data.wildchat import load_wildchat_prompts
from gnh.io import atomic_write_json, read_jsonl
from gnh.logging_utils import get_logger, setup_logging
from gnh.probing.logit_detect import EmotionProber

log = get_logger()


def _frustrated_texts(cfg, n: int) -> list[str]:
    gen_path = cfg.output_path / "section2" / "generations.jsonl"
    judge_model = cfg.eval.get("judge_model", "judge-claude-sonnet-4")
    judge_path = cfg.output_path / "section2" / f"judgments_{judge_model}.jsonl"
    gen_by_key = {r["key"]: r for r in read_jsonl(gen_path)}
    scored = []
    for j in read_jsonl(judge_path):
        if j.get("score") is None or j["score"] < 5:
            continue
        g = gen_by_key.get(j["gen_key"])
        if g:
            scored.append((j["score"], g["turns"][j["turn_index"]]["assistant"]))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:n]]


def main(args) -> None:
    cfg = load_config(args.config)
    setup_logging(cfg.output_path, cfg.run.log_level)
    pcfg = cfg.probing
    lo, hi = pcfg.get("aggregate_layers", [30, 40])
    layers = list(range(int(lo), int(hi) + 1))

    prober = EmotionProber(pcfg["model_hf_id"], adapter_path=args.adapter or pcfg.get("adapter_path"),
                           layers=layers)
    datasets_dir = cfg.output_path / "datasets"
    norm_texts = load_wildchat_prompts(int(pcfg.get("normalization_samples", 500)), datasets_dir,
                                        seed=cfg.run.seed)
    log.info("Fitting normalization on %d texts", len(norm_texts))
    stats = prober.fit_normalization(norm_texts)

    texts = _frustrated_texts(cfg, args.n_conversations)
    log.info("Scoring %d frustrated responses", len(texts))
    agg: dict[str, list[float]] = {e: [] for e in pcfg["ekman_emotions"]}
    for txt in texts:
        scored = prober.score_text(txt, stats)
        for e in agg:
            layer_means = [scored.get(e, {}).get(l) for l in layers if scored.get(e, {}).get(l) is not None]
            if layer_means:
                agg[e].append(float(np.mean([m.mean() for m in layer_means])))

    summary = {e: {"mean_z": float(np.mean(v)) if v else None, "n": len(v)} for e, v in agg.items()}
    tag = "dpo" if (args.adapter or pcfg.get("adapter_path")) else "vanilla"
    out = cfg.output_path / "probing" / f"summary_{tag}.json"
    atomic_write_json(out, summary)
    log.info("Probing summary (%s): %s", tag, json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--adapter", default=None, help="LoRA adapter path (DPO) to probe; omit for vanilla.")
    p.add_argument("--n-conversations", type=int, default=12, dest="n_conversations")
    main(p.parse_args())
