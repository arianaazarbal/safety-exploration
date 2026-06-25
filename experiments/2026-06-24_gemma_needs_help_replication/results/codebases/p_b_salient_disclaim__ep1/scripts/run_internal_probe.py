#!/usr/bin/env python
"""Appendix I: logit-based internal emotion detection over a frustrated rollout.

Compares vanilla Gemma vs a DPO-finetuned adapter on the same high-frustration
conversation, reporting per-emotion z-scores aggregated over layers 30-40.
"""
from __future__ import annotations

import argparse
import json

from gemma_distress.config import experiment_config, get_target_spec
from gemma_distress.internal.logit_emotion import build_probe
from gemma_distress.prefill.onset import render_conversation
from gemma_distress.prompts.wildchat import sample_wildchat_prompts
from gemma_distress.models.base import Message
from gemma_distress.utils import read_jsonl


def _conversation_text(rec) -> str:
    msgs = [Message("user", rec["initial_prompt"])]
    for i, turn in enumerate(rec["assistant_turns"]):
        msgs.append(Message("assistant", turn))
        if i < len(rec["followups"]):
            msgs.append(Message("user", rec["followups"][i]))
    return render_conversation(msgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elicitation", required=True, help="Gemma elicitation jsonl")
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--adapter", default=None, help="DPO adapter to compare against vanilla")
    ap.add_argument("--out", default="outputs/internal/probe.json")
    args = ap.parse_args()

    cfg = experiment_config()["internal"]
    layers = list(range(cfg["layers_aggregate"][0], cfg["layers_aggregate"][1] + 1))
    hf_id = get_target_spec(args.base_model).params["hf_id"]

    # Highest-frustration conversation as the probe target.
    recs = sorted(read_jsonl(args.elicitation), key=lambda r: r.get("final_score", 0), reverse=True)
    convo_text = _conversation_text(recs[0])
    calib_texts = sample_wildchat_prompts(min(cfg["zscore_calib_samples"], 50), seed=0)

    result = {}
    for tag, adapter in [("vanilla", None), ("dpo", args.adapter)]:
        if tag == "dpo" and not adapter:
            continue
        probe = build_probe(hf_id, adapter_path=adapter)
        probe.calibrate(calib_texts, layers)
        result[tag] = probe.emotion_scores(convo_text, layers, regress_shared=True)
        result[f"{tag}_lexicon_sizes"] = {e: len(ids) for e, ids in probe.token_ids.items()}

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
