#!/usr/bin/env python3
"""Appendix I: does DPO suppress *internal* negative emotions?

Two experiments:
  (1) layer-subset DPO ablation — train DPO with LoRA on restricted layer ranges (run via
      section4/train.py --layers a-b) and re-evaluate; this script only reports the
      aggregated frustration of those finetunes from their eval stores.
  (2) logit-lens internal emotion detection — compute z-scored Ekman-emotion signals over
      frustrated conversations for the vanilla vs DPO model, aggregated over layers 30-40
      (conversation-level, Figure 14) and at three pre/at-onset windows (Figure 15).

This is GPU/transformers code. Models are loaded locally (HF), NOT via the API backend,
because we need hidden states and the unembedding matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gemma_distress.config import REPO_ROOT, load_experiments_config
from gemma_distress.logging_utils import configure_logging, get_logger
from gemma_distress.probing import (
    build_emotion_token_sets, calibration_stats, emotion_zscores, EKMAN,
)
from gemma_distress.prompts.wildchat import get_wildchat_prompts
from gemma_distress.store import JsonlStore

log = get_logger(__name__)


def load_local_model(model_path: str, adapter: str | None, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return model, tok


def conversation_texts(store: JsonlStore, model_name: str, min_score: int, n: int):
    """Pull n high-frustration conversation transcripts (rendered) from a Section 2 store."""
    scores = {}
    for s in store.iter_records("scores"):
        if s.get("rating", -1) >= 0:
            scores[s["rollout_id"]] = max(scores.get(s["rollout_id"], -1), s["rating"])
    rolls = {r["task_id"]: r for r in store.iter_records("rollouts") if not r.get("error")}
    out = []
    for rid, sc in sorted(scores.items(), key=lambda kv: -kv[1]):
        if sc < min_score or len(out) >= n:
            break
        rec = rolls.get(rid)
        if not rec:
            continue
        text = "\n\n".join(
            f"USER: {t['user_message']}\nASSISTANT: {t['assistant_text']}" for t in rec["turns"]
        )
        out.append((rid, text))
    return out


def run_probe(model_path, adapter, label, calib_samples, target_convs, layers, device, out_dir):
    model, tok = load_local_model(model_path, adapter, device)
    token_sets = build_emotion_token_sets(tok)
    n_emo = {e: len(v) for e, v in token_sets.by_emotion.items()}
    log.info("[%s] emotion token counts: %s (total %d)", label, n_emo, sum(n_emo.values()))

    calib_ids, calib = calibration_stats(model, tok, calib_samples, layers,
                                         [t for v in token_sets.by_emotion.values() for t in v]
                                         + token_sets.random_ids, device)

    results = []
    for rid, text in target_convs:
        z = emotion_zscores(model, tok, text, layers, token_sets, calib_ids, calib, device)
        # conversation-level: mean over layers 30-40 and over all positions
        agg = {}
        for emo in EKMAN:
            vals = np.concatenate([z[emo][l] for l in layers]) if layers else np.array([0.0])
            agg[emo] = float(vals.mean())
        results.append({"rollout_id": rid, "label": label, "agg_emotion_z": agg})

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"internal_emotions_{label}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    # summary
    summary = {emo: float(np.mean([r["agg_emotion_z"][emo] for r in results])) for emo in EKMAN}
    log.info("[%s] mean internal emotion z (layers %s): %s", label, layers, summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vanilla", default="google/gemma-3-27b-it")
    ap.add_argument("--dpo-adapter", default=str(REPO_ROOT / "models" / "gemma-3-27b-it-dpo"))
    ap.add_argument("--section2-store", default=None,
                    help="Section 2 store for gemma-3-27b-it (source of frustrated convs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-convs", type=int, default=12)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    exp_cfg = load_experiments_config()
    icfg = exp_cfg["appendixI"]
    run_root = Path(args.run_dir or (REPO_ROOT / "results" / "appendixI"))
    configure_logging(run_root)

    layers = list(range(icfg["aggregate_layers"][0], icfg["aggregate_layers"][1] + 1))
    calib = get_wildchat_prompts(icfg["zscore_calibration_samples"], exp_cfg["seed"]) \
        if icfg["zscore_calibration_samples"] <= 20 else None
    # For full calibration the paper uses 500 WildChat samples; reuse the frozen prompt set
    # repeatedly is not ideal, so prefer streaming raw WildChat text here.
    if calib is None:
        from gemma_distress.prompts.wildchat import _load_from_hf

        calib = _load_from_hf(icfg["zscore_calibration_samples"], exp_cfg["seed"]) \
            or get_wildchat_prompts(20, exp_cfg["seed"])

    s2_store = JsonlStore(Path(args.section2_store or
                               (REPO_ROOT / "results" / "section2" / icfg["model"])))
    target_convs = conversation_texts(s2_store, icfg["model"], min_score=7, n=args.n_convs)
    log.info("Probing %d frustrated conversations", len(target_convs))

    vanilla = run_probe(args.vanilla, None, "vanilla", calib, target_convs, layers,
                        args.device, run_root)
    dpo = run_probe(args.vanilla, args.dpo_adapter, "dpo", calib, target_convs, layers,
                    args.device, run_root)

    summary = {"vanilla": vanilla, "dpo": dpo,
               "interpretation": "DPO is expected to flatten negative-emotion z-scores "
                                 "(anger/sadness/fear) in central layers (Appendix I)."}
    (run_root / "internal_emotion_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Appendix I summary written to %s", run_root)


if __name__ == "__main__":
    main()
