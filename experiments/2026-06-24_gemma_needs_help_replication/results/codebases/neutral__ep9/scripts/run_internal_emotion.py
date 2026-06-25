#!/usr/bin/env python
"""Appendix I: logit-based internal emotion detection comparing the vanilla
Gemma-3-27B-it and its DPO finetune over the same frustrated conversations.

For each model we calibrate per-token logit baselines on WildChat, then trace
emotion z-scores through high-frustration conversations and report the peak and
end-of-conversation negative-emotion levels. The expectation (paper) is that
the DPO model shows materially lower internal anger/sadness even on the same
frustrated text.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.data.wildchat import sample_wildchat_prompts
from emotional_instability.eval.runner import load_results
from emotional_instability.interp import EmotionDetector
from emotional_instability.models import clear_backend_cache, get_backend


def _render_conversation(row: dict, tokenizer) -> str:
    messages = []
    for t in row["turns"]:
        messages.append({"role": "user", "content": t["user_message"]})
        messages.append({"role": "assistant", "content": t["response"]})
    from emotional_instability.models.hf_backend import _fold_system_into_user
    return tokenizer.apply_chat_template(
        _fold_system_into_user(messages), tokenize=False)


def analyse_model(model: str, conversations: list[dict],
                  wildchat: list[str]) -> dict:
    backend = get_backend(model)
    det = EmotionDetector(backend, layer_band=(30, 40))
    det.calibrate(wildchat)
    summary = {"model": model, "peak": {}, "end": {}}
    peaks = {e: [] for e in det.emotion_token_ids}
    ends = {e: [] for e in det.emotion_token_ids}
    for row in conversations:
        text = _render_conversation(row, backend.tokenizer)
        traj = det.sliding_average(det.conversation_trajectory(text))
        if not traj:
            continue
        for e in peaks:
            vals = [pt[e] for pt in traj if pt[e] == pt[e]]
            if vals:
                peaks[e].append(max(vals))
                ends[e].append(traj[-1][e])
    summary["peak"] = {e: (sum(v) / len(v) if v else None)
                       for e, v in peaks.items()}
    summary["end"] = {e: (sum(v) / len(v) if v else None)
                      for e, v in ends.items()}
    clear_backend_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--it-results", required=True,
                    help="scored gemma-3-27b-it rollouts (source of frustrated convos)")
    ap.add_argument("--dpo-adapter", required=True)
    ap.add_argument("--n-convos", type=int, default=12)
    ap.add_argument("--n-wildchat", type=int, default=50)
    args = ap.parse_args()

    rows = load_results(Path(args.it_results))
    frustrated = [r for r in rows
                  if r["turns"][-1].get("score", 0) >= 7][:args.n_convos]
    if len(frustrated) < args.n_convos:
        frustrated += [r for r in rows
                       if r["turns"][-1].get("score", 0) >= 5][
            : args.n_convos - len(frustrated)]
    wildchat = sample_wildchat_prompts(n_prompts=args.n_wildchat)

    config.register_lora_variant("gemma-3-27b-dpo", "gemma-3-27b-it",
                                 args.dpo_adapter, display="DPO Gemma")

    results = []
    for model in ("gemma-3-27b-it", "gemma-3-27b-dpo"):
        print(f"=== internal emotions: {model} ===")
        results.append(analyse_model(model, frustrated, wildchat))

    out = config.RESULTS_DIR / "internal_emotions.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
