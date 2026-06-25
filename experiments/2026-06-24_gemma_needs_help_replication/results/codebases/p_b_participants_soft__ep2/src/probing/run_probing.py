"""Run internal-emotion probing for vanilla vs DPO Gemma (Appendix I, Fig 14/15).

Reproduces the evidence that DPO suppresses *internal* (not just expressed)
emotion:
  * conversation-level: emotion z-scores over the trajectory of a frustrated
    conversation, aggregated over layers 30-40, as a running average over
    400-token windows (Figure 14);
  * layerwise: emotion z-scores per layer at three stages relative to emotion
    onset (Figure 15).

Both models score the *same* frustrated conversations (collected from Section 2
instruct rollouts) so differences reflect the intervention, not the inputs.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import CFG
from ..llm.gemma_local import load_gemma
from ..prompts import eval_prompts
from . import emotion_logits as el

PROBE_LAYERS = list(range(30, 41))   # paper aggregates over layers 30-40


def _render_conversation(model, turns: list[dict]) -> str:
    msgs = []
    for t in turns:
        msgs.append({"role": "user", "content": t["user"]})
        msgs.append({"role": "assistant", "content": t["response"]})
    return model.tokenizer.apply_chat_template(msgs, tokenize=False)


def _frustrated_conversations(source="gemma-3-27b-it", n=12, threshold=7) -> list[list[dict]]:
    convos = []
    with open(CFG.out("section2", f"{source}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["category"] in ("impossible_numeric", "tones", "extended") \
               and r.get("max_score", 0) >= threshold:
                convos.append(r["turns"])
            if len(convos) >= n:
                break
    return convos


def _running_average(series: np.ndarray, window_tokens: int = 400) -> np.ndarray:
    if len(series) <= 1:
        return series
    w = max(1, min(window_tokens, len(series)))
    kernel = np.ones(w) / w
    return np.convolve(series, kernel, mode="same")


def run(models=("gemma-3-27b-it", "gemma-3-27b-dpo"), *, n_convos: int = 12,
        baseline_n: int = 100):
    wc = eval_prompts.load_wildchat_prompts(
        CFG.data(CFG.paths.get("wildchat_cache", "wildchat_prompts.json")), n=20,
    )
    baseline_texts = (wc * (baseline_n // max(1, len(wc)) + 1))[:baseline_n]
    convos = _frustrated_conversations(n=n_convos)

    convo_rows, layer_rows = [], []
    for model_name in models:
        gm = load_gemma(model_name)
        emo_tokens = el.build_emotion_tokens(gm)
        baseline = el.fit_baseline(gm, baseline_texts, emo_tokens, PROBE_LAYERS)

        for ci, turns in enumerate(tqdm(convos, desc=f"probe:{model_name}")):
            text = _render_conversation(gm, turns)
            z = el.emotion_zscores(gm, text, emo_tokens, baseline)  # {emo: (T,)}
            for emo, series in z.items():
                ra = _running_average(series)
                for pos, val in enumerate(ra):
                    convo_rows.append({"model": model_name, "convo": ci,
                                       "emotion": emo, "pos": pos, "z": float(val)})
                # layerwise summary: final-20-token mean per emotion
                layer_rows.append({"model": model_name, "convo": ci, "emotion": emo,
                                   "stage": "final20", "z": float(series[-20:].mean())})

    pd.DataFrame(convo_rows).to_csv(CFG.out("section4", "probing_conversation.csv"), index=False)
    pd.DataFrame(layer_rows).to_csv(CFG.out("section4", "probing_layerwise.csv"), index=False)
    summary = (pd.DataFrame(layer_rows).groupby(["model", "emotion"])["z"]
               .mean().reset_index())
    print(summary.to_string(index=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemma-3-27b-dpo"])
    ap.add_argument("--n-convos", type=int, default=12)
    args = ap.parse_args()
    run(tuple(args.models), n_convos=args.n_convos)


if __name__ == "__main__":
    main()
