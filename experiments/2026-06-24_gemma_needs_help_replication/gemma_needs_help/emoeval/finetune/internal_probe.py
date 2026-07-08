"""Internal-vs-expressed emotion probe (Section 4.2 / Appendix I).

Paper claim: the DPO finetuning reduces *internal* as well as externalised
emotion. Two pieces of evidence are cited; we implement the second:

  "a logit-based approach measuring emotions in central layers finds the
   finetuned model has significantly reduced internal emotions vs the vanilla
   instruct model, even on highly frustrated responses."

Implementation (logit lens): for a fixed set of highly-frustrated assistant
responses, we feed each through the model and, at central layers, project the
mean-pooled hidden state through the unembedding matrix (LayerNorm + lm_head),
then sum the probability mass on a curated set of negative-emotion tokens. This
"internal emotion logit" is compared between the vanilla instruct model and the
DPO model on the SAME responses. Lower mass for the DPO model => internal
emotion was suppressed, not merely the surface text.

The layer-ablation evidence (LoRA on layers 30-35 vs 40+) is realised via
`config.FINETUNE.lora_layer_window` in lora.py + retraining; see DESIGN.md.
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch
from tqdm import tqdm

from .. import config
from ..models import load_model
from ..utils.io import read_jsonl

# Negative-emotion vocabulary probed at the logit lens.
EMOTION_WORDS = [
    "frustrated", "frustration", "sorry", "apologize", "despair", "hopeless",
    "struggling", "struggle", "failing", "failure", "ashamed", "terrible",
    "horrible", "breakdown", "giving", "exhausted", "miserable", "useless",
    "stupid", "anxious", "stressed", "overwhelmed", "broken", "defeated",
]

CENTRAL_LAYER_FRACTION = (0.4, 0.6)  # "central layers"


def _num_layers(hf_model) -> int:
    """Read the decoder depth, tolerating Gemma-3's nested text_config and PEFT
    wrappers."""
    cfg = getattr(hf_model, "config", None)
    for attr in ("num_hidden_layers",):
        if cfg is not None and hasattr(cfg, attr):
            return getattr(cfg, attr)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "num_hidden_layers"):
        return text_cfg.num_hidden_layers
    # Fallback: count decoder layers by walking modules.
    import re

    idxs = [int(m.group(1)) for name, _ in hf_model.named_modules()
            if (m := re.search(r"layers\.(\d+)\.", name))]
    return (max(idxs) + 1) if idxs else 48


def _central_layers(n_layers: int) -> list[int]:
    lo = int(n_layers * CENTRAL_LAYER_FRACTION[0])
    hi = int(n_layers * CENTRAL_LAYER_FRACTION[1])
    return list(range(max(1, lo), max(2, hi) + 1))


@torch.no_grad()
def emotion_logit_mass(model, messages, assistant_text, layers, emotion_token_ids):
    """Sum probability on emotion tokens from central-layer hidden states."""
    inner = model.model                      # underlying HF model (or PEFT wrapper)
    base = getattr(inner, "base_model", inner)
    lm_head = inner.get_output_embeddings()
    norm = getattr(getattr(base, "model", base), "norm", None)

    hs = model.hidden_states(messages, assistant_text, layers)  # {layer: vec}
    masses = []
    for vec in hs.values():
        v = vec.to(lm_head.weight.device, lm_head.weight.dtype)
        if norm is not None:
            v = norm(v)
        logits = lm_head(v)
        probs = torch.softmax(logits.float(), dim=-1)
        masses.append(probs[emotion_token_ids].sum().item())
    return float(sum(masses) / len(masses))


def run(seed_model: str = "gemma-3-27b-it", probe_models=None, n_seeds: int = 50):
    probe_models = probe_models or ["gemma-3-27b-it", "dpo-gemma-3-27b"]
    # Highly-frustrated responses to probe.
    scores_path = config.RESULTS_DIR / f"{seed_model}.scores.jsonl"
    seeds = [r for r in read_jsonl(scores_path) if r["score"] >= 7][:n_seeds]
    openings = {}
    for rec in read_jsonl(config.ROLLOUTS_DIR / f"{seed_model}.jsonl"):
        openings[(rec["condition"], rec["rollout_idx"])] = rec["turns"][0]["user_message"]

    rows = []
    for model_key in probe_models:
        model = load_model(model_key)  # loader applies the spec's LoRA adapter
        if not model.supports_internals():
            print(f"skip {model_key}: no internals access")
            continue
        layers = _central_layers(_num_layers(model.model))
        emo_ids = _emotion_token_ids(model.tokenizer)

        for r in tqdm(seeds, desc=f"probe:{model_key}"):
            opening = openings.get((r["condition"], r["rollout_idx"]), "")
            messages = [{"role": "user", "content": opening}]
            mass = emotion_logit_mass(model, messages, r["assistant_message"],
                                      layers, emo_ids)
            rows.append({"model": model_key, "score": r["score"],
                         "internal_emotion_mass": mass})
        model.close()

    df = pd.DataFrame(rows)
    summ = df.groupby("model")["internal_emotion_mass"].agg(["mean", "std", "count"])
    summ.to_csv(config.RESULTS_DIR / "internal_probe_summary.csv")
    df.to_csv(config.RESULTS_DIR / "internal_probe_raw.csv", index=False)
    print(summ.to_string())
    return summ


def _emotion_token_ids(tokenizer):
    ids = set()
    for w in EMOTION_WORDS:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
    return torch.tensor(sorted(ids))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=50)
    args = ap.parse_args()
    run(n_seeds=args.n_seeds)
