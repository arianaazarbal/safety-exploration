#!/usr/bin/env python
"""Internal-emotion probing (Appendix I). Gemma-only.

Compares logit-lens emotion z-scores through a frustrated conversation for the
vanilla vs DPO Gemma, aggregated over central layers.

    python scripts/run_internal_emotions.py \
        --conversations runs/frustrated_convs.txt \
        --out results/internal_emotions.json
        # add --adapter runs/dpo to probe the finetune

`--conversations` is a file with one rendered conversation per line (or a JSONL
with a "text" field). WildChat stats samples come from the WildChat loader.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress import config, internal_emotions as ie
from distress.models import build_client
from distress.tasks import load_wildchat_prompts


def _load_conversations(path):
    p = Path(path)
    texts = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            texts.append(obj["text"] if isinstance(obj, dict) else str(obj))
        except json.JSONDecodeError:
            texts.append(line)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    models_cfg = config.load_models()
    exp = config.load_experiment()
    ic = exp["internal_emotions"]
    layers = list(range(ic["layers_aggregate"][0], ic["layers_aggregate"][1]))

    kwargs = {"adapter_path": args.adapter} if args.adapter else {}
    client = build_client(config.get_target(args.model, models_cfg), **kwargs)
    _, tok = client.get_model_and_tokenizer()

    token_sets = ie.build_emotion_token_sets(tok)
    print({e: len(ids) for e, ids in token_sets.token_ids.items()})

    wc = load_wildchat_prompts(ic["zscore_samples"])
    stats = ie.compute_zscore_stats(client, token_sets, wc, layers)

    convs = _load_conversations(args.conversations)
    out = []
    for i, conv in enumerate(convs):
        scores = ie.emotion_scores_for_conversation(
            client, token_sets, stats, conv, layers)
        out.append({
            "conv_id": i,
            "emotion_running_avg": {
                e: ie.running_average(v, ic["running_window_tokens"]).tolist()
                for e, v in scores.items()},
            "emotion_mean": {e: float(v.mean()) for e, v in scores.items()},
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[internal] wrote {args.out}")
    for rec in out:
        print(rec["conv_id"], {k: round(v, 3)
                               for k, v in rec["emotion_mean"].items()})


if __name__ == "__main__":
    main()
