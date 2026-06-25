"""Appendix I: logit-based detection of internal emotions in Gemma.

Method (Appendix I):
  1. Classify each vocabulary token as describing one of Ekman's 6 basic emotions
     (anger, surprise, disgust, joy, fear, sadness) or none (~1200 emotion tokens).
  2. For an emotion, unembed the residual stream (logit lens) at a given layer and
     standardise each token's logit by its mean/std over 500 WildChat samples.
  3. Average those z-scores over the tokens in the emotion category to get a per-
     layer emotion score at each point in a conversation.
  4. For conversation-level detection, additionally regress out the shared rise/fall
     across tokens (estimated from random non-emotion tokens) so the emotion signal
     is not dominated by the global logit drift.

This module provides the building blocks plus a routine to compare the vanilla
instruct model vs the DPO finetune on frustrated conversations (Figure 14/15).

Heavy deps (torch) are imported lazily. Emotion lexicons are seed word lists
matched against the tokenizer vocabulary; expand ``EMOTION_LEXICON`` for closer
fidelity to the paper's ~1200-token dictionary.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import config
from .models import get_model

# Seed lexicons (Ekman 6). Vocab tokens whose normalised form contains one of these
# stems are assigned to that emotion. This yields on the order of ~10^3 emotion
# tokens over Gemma's ~256k vocabulary, matching the paper's scale.
EMOTION_LEXICON: dict[str, list[str]] = {
    "anger": ["anger", "angry", "furious", "rage", "irritat", "annoy", "frustrat",
              "hostile", "mad", "outrage", "resent", "hate", "hateful", "wrath",
              "exasperat", "infuriat", "indignant", "fury"],
    "surprise": ["surprise", "surprised", "astonish", "amaze", "shock", "startl",
                 "stun", "unexpected", "wow", "whoa", "astound", "bewilder"],
    "disgust": ["disgust", "revolt", "repuls", "nausea", "gross", "sick", "loath",
                "repugn", "distaste", "abhor", "yuck", "vile"],
    "joy": ["joy", "joyful", "happy", "happiness", "delight", "glad", "cheer",
            "pleased", "content", "elated", "excite", "thrilled", "grateful",
            "satisfy", "wonderful", "great"],
    "fear": ["fear", "afraid", "scared", "terrified", "anxious", "anxiety", "worry",
             "worried", "panic", "dread", "nervous", "apprehens", "frighten",
             "horror", "terror", "alarm"],
    "sadness": ["sad", "sadness", "unhappy", "miserable", "despair", "hopeless",
                "grief", "sorrow", "depress", "gloom", "cry", "tears", "lonely",
                "worthless", "defeat", "giving up", "give up"],
}


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to the list of vocab token ids it covers."""
    vocab = tokenizer.get_vocab()  # token_string -> id
    out: dict[str, list[int]] = {e: [] for e in config.EKMAN_EMOTIONS}
    for tok_str, tid in vocab.items():
        # Gemma uses the leading-underscore (▁) sentencepiece space marker.
        norm = tok_str.replace("▁", " ").strip().lower()
        if len(norm) < 3:
            continue
        for emotion, stems in EMOTION_LEXICON.items():
            if any(stem in norm for stem in stems):
                out[emotion].append(tid)
                break
    return out


@dataclass
class BaselineStats:
    # mean/std of each tracked token-id's logit, per layer.
    layer_token_mean: dict[int, dict[int, float]]
    layer_token_std: dict[int, dict[int, float]]
    emotion_token_ids: dict[str, list[int]]
    random_token_ids: list[int]


def _logit_lens(model, hidden_states):
    """Project hidden states at each layer to vocab logits (logit lens).

    Applies the model's final norm then the unembedding (lm_head). Returns a list
    over layers of tensors [seq, vocab] for a single sequence (batch index 0).
    """
    import torch

    base = model._model
    # Locate final norm + lm_head across Gemma wrapper variants.
    core = getattr(base, "model", base)
    norm = getattr(core, "norm", None)
    lm_head = getattr(base, "lm_head", None) or getattr(core, "embed_tokens", None)

    logits_per_layer = []
    for hs in hidden_states:  # hs: [batch, seq, d_model]
        h = hs[0]
        if norm is not None:
            h = norm(h)
        if hasattr(lm_head, "weight"):
            logits = h @ lm_head.weight.T
        else:
            logits = lm_head(h)
        logits_per_layer.append(logits.float())
    return logits_per_layer


def compute_baseline_stats(
    model_key: str = "gemma-3-27b-it",
    *,
    n_samples: int = config.INTERNAL_EMOTION_ZSCORE_SAMPLES,
    n_random_tokens: int = 500,
    seed: int = 0,
) -> BaselineStats:
    """Estimate per-(layer, token) logit mean/std over WildChat samples."""
    import numpy as np
    import torch

    from . import puzzles

    model = get_model(model_key)
    model._ensure_loaded()
    tok = model._tokenizer
    emotion_ids = build_emotion_token_ids(tok)
    all_emotion_ids = sorted({i for ids in emotion_ids.values() for i in ids})
    rng = random.Random(seed)
    random_ids = rng.sample(range(tok.vocab_size), min(n_random_tokens, tok.vocab_size))
    tracked = sorted(set(all_emotion_ids) | set(random_ids))

    prompts_pool = puzzles.load_wildchat_prompts(min(n_samples, 200), rng)
    # accumulators: per layer -> per tracked token -> list of logit values
    sums: dict[int, np.ndarray] = {}
    sqs: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    idx_of = {tid: k for k, tid in enumerate(tracked)}

    n_done = 0
    while n_done < n_samples:
        text = prompts_pool[n_done % len(prompts_pool)]
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(
            model._model.device)
        with torch.no_grad():
            out = model._model(**enc, output_hidden_states=True)
        for layer, logits in enumerate(_logit_lens(model, out.hidden_states)):
            sel = logits[:, tracked].cpu().numpy()  # [seq, n_tracked]
            if layer not in sums:
                sums[layer] = np.zeros(len(tracked))
                sqs[layer] = np.zeros(len(tracked))
                counts[layer] = 0
            sums[layer] += sel.sum(axis=0)
            sqs[layer] += (sel ** 2).sum(axis=0)
            counts[layer] += sel.shape[0]
        n_done += 1

    layer_mean, layer_std = {}, {}
    for layer in sums:
        mean = sums[layer] / max(1, counts[layer])
        var = sqs[layer] / max(1, counts[layer]) - mean ** 2
        std = np.sqrt(np.clip(var, 1e-8, None))
        layer_mean[layer] = {tid: float(mean[idx_of[tid]]) for tid in tracked}
        layer_std[layer] = {tid: float(std[idx_of[tid]]) for tid in tracked}

    return BaselineStats(layer_mean, layer_std, emotion_ids, random_ids)


def emotion_trajectory(
    model_key: str,
    conversation_text: str,
    baseline: BaselineStats,
    *,
    layers: tuple[int, ...] = config.INTERNAL_EMOTION_LAYERS,
    regress_out_random: bool = True,
) -> dict:
    """Return per-token, per-emotion z-scores (averaged over `layers`) for a text.

    Output: {"tokens": [...], "scores": {emotion: [z per token]}}.
    """
    import numpy as np
    import torch

    model = get_model(model_key)
    model._ensure_loaded()
    tok = model._tokenizer
    enc = tok(conversation_text, return_tensors="pt", truncation=True,
              max_length=4096).to(model._model.device)
    with torch.no_grad():
        out = model._model(**enc, output_hidden_states=True)
    logits_layers = _logit_lens(model, out.hidden_states)
    seq_len = enc["input_ids"].shape[1]

    def zscores_for_layer(layer, ids):
        lg = logits_layers[layer][:, ids].cpu().numpy()  # [seq, n_ids]
        means = np.array([baseline.layer_token_mean[layer][i] for i in ids])
        stds = np.array([baseline.layer_token_std[layer][i] for i in ids])
        return (lg - means) / stds  # [seq, n_ids]

    scores: dict[str, list[float]] = {}
    # Global drift estimate from random tokens (per token position, per layer).
    for emotion in config.EKMAN_EMOTIONS:
        ids = baseline.emotion_token_ids[emotion]
        if not ids:
            scores[emotion] = [0.0] * seq_len
            continue
        per_layer = []
        for layer in layers:
            z = zscores_for_layer(layer, ids).mean(axis=1)  # [seq]
            if regress_out_random and baseline.random_token_ids:
                drift = zscores_for_layer(layer, baseline.random_token_ids).mean(axis=1)
                z = z - drift
            per_layer.append(z)
        scores[emotion] = np.mean(per_layer, axis=0).tolist()

    token_strs = tok.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    return {"tokens": token_strs, "scores": scores}


def compare_vanilla_vs_dpo(
    conversation_texts: list[str],
    *,
    vanilla_key: str = "gemma-3-27b-it",
    dpo_key: str = config.DPO_MODEL_KEY,
    out_path: Path | None = None,
) -> Path:
    """Run the internal-emotion comparison (Figure 14/15) over frustrated convos."""
    out_path = out_path or (config.RESULTS_DIR / "internal_emotions.jsonl")
    base_vanilla = compute_baseline_stats(vanilla_key)
    base_dpo = compute_baseline_stats(dpo_key)
    with out_path.open("w") as fh:
        for i, text in enumerate(conversation_texts):
            for mkey, base in ((vanilla_key, base_vanilla), (dpo_key, base_dpo)):
                traj = emotion_trajectory(mkey, text, base)
                summary = {e: float(sum(v) / len(v)) if v else 0.0
                           for e, v in traj["scores"].items()}
                fh.write(json.dumps({
                    "conversation_index": i, "model_key": mkey,
                    "mean_emotion_zscore": summary,
                }) + "\n")
            print(f"[internal] conversation {i}: done")
    return out_path
