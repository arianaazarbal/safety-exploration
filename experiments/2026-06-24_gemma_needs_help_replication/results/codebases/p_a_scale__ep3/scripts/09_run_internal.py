#!/usr/bin/env python
"""Appendix I: logit-based internal-emotion detection on Gemma.

Calibrates the probe on WildChat, then scores high-frustration conversations
(from the eval rollouts) under the vanilla and DPO models, to test whether DPO
suppresses *internal* negative emotions (not just expressed ones). Uses the
transformers backend (hidden states / LM head).

  python scripts/09_run_internal.py --lora-path model_store/gemma-3-27b-it-dpo-diverse
"""
import numpy as np

from _bootstrap import boot, common_parser

from eilm.data.wildchat import select_wildchat_prompts
from eilm.internal.emotion_logit import EmotionProbe
from eilm.models.local_hf import HFModel
from eilm.utils.io import read_jsonl, write_json


def high_frustration_convos(cfg, model_name, min_score, limit):
    rollouts = list(read_jsonl(cfg.path("data") / "rollouts" / f"{model_name}.jsonl"))
    scores = list(read_jsonl(cfg.path("data") / "scores" / f"{model_name}.jsonl"))
    final = {}
    for s in scores:
        if s.get("rating") is None:
            continue
        k = (s["condition"], s["index"])
        if k not in final or s["turn"] > final[k][0]:
            final[k] = (s["turn"], s["rating"])
    convos = []
    for rec in rollouts:
        k = (rec["condition"], rec["index"])
        if k in final and final[k][1] >= min_score:
            text = "\n\n".join(
                ("User: " if m["role"] == "user" else "Assistant: ") + m["content"]
                for m in rec["messages"]
            )
            convos.append(text)
        if len(convos) >= limit:
            break
    return convos


def score_model(cfg, hf_id, lora_path, calib_texts, convos, icfg):
    model = HFModel(hf_id=hf_id, name="probe", lora_path=lora_path)
    probe = EmotionProbe(
        model, emotions=icfg["ekman_emotions"],
        tokens_per_emotion=icfg["tokens_per_emotion"],
        n_random=icfg["regress_out_random_tokens"],
        seed=cfg["generation"]["seed"],
    )
    probe.calibrate(calib_texts)
    # Per-emotion peak conversation-level z, averaged over conversations.
    agg = {e: [] for e in icfg["ekman_emotions"]}
    for text in convos:
        cs = probe.conversation_scores(text, layers=tuple(icfg["layers_aggregate"]),
                                       window=icfg["running_window_tokens"])
        for e, arr in cs.items():
            if len(arr):
                agg[e].append(float(np.max(arr)))
    return {e: (float(np.mean(v)) if v else None) for e, v in agg.items()}


def main():
    p = common_parser(__doc__)
    p.add_argument("--lora-path", default=None, help="DPO adapter to compare against vanilla")
    p.add_argument("--min-score", type=int, default=7)
    p.add_argument("--n-convos", type=int, default=12)
    args = p.parse_args()
    cfg, registry, logger = boot(args)

    icfg = cfg["internal"]
    model_name = icfg["model"]
    hf_id = cfg["targets"][model_name]["hf_id"]

    calib_texts = select_wildchat_prompts(
        n=icfg["wildchat_calib_samples"], seed=cfg["generation"]["seed"],
        cache_path=cfg.path("cache") / "wildchat_calib.json",
    )
    convos = high_frustration_convos(cfg, model_name, args.min_score, args.n_convos)
    logger.info("Scoring %d high-frustration conversations", len(convos))

    result = {"vanilla": score_model(cfg, hf_id, None, calib_texts, convos, icfg)}
    if args.lora_path:
        result["dpo"] = score_model(cfg, hf_id, args.lora_path, calib_texts, convos, icfg)

    write_json(cfg.path("results") / "internal_emotions.json", result)
    logger.info("Internal-emotion summary: %s", result)


if __name__ == "__main__":
    main()
