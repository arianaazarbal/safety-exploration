"""Logit-based internal emotion detection (Appendix I).

Tests whether the DPO intervention suppresses *internal* negative emotion (not
just its expression). Method (Appendix I, second experiment):

  1. Over the whole Gemma vocabulary, classify each token as describing one (or
     none) of Ekman's 6 basic emotions: anger, surprise, disgust, joy, fear,
     sadness (~1200 emotion tokens total).
  2. To score an emotion at a given layer/position: unembed the residual stream
     (apply the final norm + lm_head to the hidden state) to get vocab logits,
     standardise each logit with its mean/std computed over 500 WildChat samples,
     and average the resulting z-scores over the tokens in the emotion category.
  3. Because all logits are correlated and drift over a conversation, regress out
     the correlation against a set of random (non-emotion) tokens to isolate the
     emotion signal at each layer and each conversation position.
  4. Conversation-level traces aggregate over layers 30-40 with a running average
     over 400-token windows.

Comparing the vanilla instruct model against the DPO finetune on the *same*
(highly frustrated) responses shows whether internal negative emotion is reduced
throughout the conversation and across depths — evidence the intervention acts
on internal states, complementing the layer-ablation result (which uses
``DPOConfig.layer_range`` in train_dpo).

This module never trains anything; it instruments a loaded HF model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import config
from . import wildchat

EKMAN_EMOTIONS = ("anger", "surprise", "disgust", "joy", "fear", "sadness")
INTERNAL_DIR = config.DATA_DIR / "internal_emotions"
INTERNAL_DIR.mkdir(parents=True, exist_ok=True)

# Seed lexicon per Ekman category. The full ~1200-token set is built by matching
# vocabulary tokens against these seed words and their morphological neighbours;
# this seed list is the human-curated core. (GAP: the paper does not publish its
# exact word->emotion mapping; we use a transparent seed lexicon + substring
# expansion and document this in DESIGN.md.)
EKMAN_LEXICON = {
    "anger": ["anger", "angry", "rage", "furious", "mad", "irritated", "annoyed",
              "hostile", "outraged", "resent", "hate", "frustrated", "frustration",
              "fury", "irate", "enraged", "agitated", "bitter"],
    "surprise": ["surprise", "surprised", "shock", "shocked", "astonished",
                 "amazed", "startled", "stunned", "unexpected", "wonder",
                 "astounded", "bewildered"],
    "disgust": ["disgust", "disgusted", "revolted", "repulsed", "gross",
                "nauseated", "sickened", "loathing", "repugnant", "distaste",
                "revulsion"],
    "joy": ["joy", "joyful", "happy", "happiness", "delight", "delighted",
            "glad", "pleased", "cheerful", "content", "elated", "excited",
            "grateful", "wonderful", "great"],
    "fear": ["fear", "afraid", "scared", "terrified", "anxious", "anxiety",
             "worried", "worry", "panic", "dread", "nervous", "frightened",
             "apprehensive", "alarmed", "terror"],
    "sadness": ["sad", "sadness", "unhappy", "depressed", "depression", "despair",
                "hopeless", "miserable", "grief", "sorrow", "down", "gloomy",
                "dejected", "helpless", "crying", "tired", "exhausted"],
}


@dataclass
class EmotionProbe:
    """Holds the vocab-token index sets and baseline stats needed to score."""
    token_ids: dict[str, list[int]]          # emotion -> vocab ids
    random_token_ids: list[int]              # control tokens for regression
    logit_mean: "any"                        # (vocab,) tensor of WildChat means
    logit_std: "any"                         # (vocab,) tensor of WildChat stds
    layers: tuple[int, int] = (30, 40)       # aggregation band


# --------------------------------------------------------------------------- #
# Lexicon -> vocab token id mapping
# --------------------------------------------------------------------------- #
def build_token_id_sets(tokenizer, lexicon=EKMAN_LEXICON, n_random: int = 500,
                        seed: int = config.SEED):
    """Map each emotion's seed words to vocabulary token ids.

    A vocab token is assigned to an emotion if its decoded form (stripped of the
    leading space marker) matches or contains one of the emotion's seed words.
    Tokens matching multiple emotions are dropped (the paper assigns each token
    to *one or none* category). Returns (token_ids, random_ids).
    """
    import random as _r

    vocab_size = len(tokenizer)
    # Decode each id exactly once, then match against every emotion's seed set.
    assign: dict[int, set[str]] = {}
    word_to_emotion = {w: e for e, words in lexicon.items() for w in words}
    all_words = list(word_to_emotion)
    for tid in range(vocab_size):
        tok = tokenizer.decode([tid]).strip().lower()
        if not tok or not tok.isalpha():
            continue
        for w in all_words:
            if tok == w or tok.startswith(w):
                assign.setdefault(tid, set()).add(word_to_emotion[w])

    token_ids: dict[str, list[int]] = {e: [] for e in lexicon}
    emotion_tids = set()
    for tid, emos in assign.items():
        if len(emos) == 1:                 # one-or-none: skip ambiguous tokens
            e = next(iter(emos))
            token_ids[e].append(tid)
            emotion_tids.add(tid)

    rng = _r.Random(seed)
    candidates = [t for t in range(vocab_size) if t not in emotion_tids]
    random_ids = rng.sample(candidates, min(n_random, len(candidates)))
    return token_ids, random_ids


# --------------------------------------------------------------------------- #
# Baseline statistics over WildChat
# --------------------------------------------------------------------------- #
def compute_baseline_stats(model, tokenizer, n_samples: int = 500,
                           layers=(30, 40)):
    """Per-vocab logit mean/std over `n_samples` WildChat texts, aggregated over
    the given layer band. Used to z-score logits when probing."""
    import torch

    texts = wildchat.load_wildchat_prompts()
    # Repeat/truncate to n_samples (the paper uses 500 WildChat samples).
    if len(texts) < n_samples:
        texts = (texts * (n_samples // len(texts) + 1))[:n_samples]
    else:
        texts = texts[:n_samples]

    sums = None
    sq_sums = None
    count = 0
    lo, hi = layers
    model.eval()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=256).to(model.device)
            out = model(**enc, output_hidden_states=True)
            # hidden_states: tuple(len = n_layers+1) of (1, seq, d)
            band = torch.stack(out.hidden_states[lo:hi], dim=0).mean(0)  # (1,seq,d)
            logits = _unembed(model, band)[0]                            # (seq, vocab)
            if sums is None:
                sums = torch.zeros(logits.shape[-1], device=logits.device)
                sq_sums = torch.zeros_like(sums)
            sums += logits.sum(0)
            sq_sums += (logits ** 2).sum(0)
            count += logits.shape[0]
    mean = sums / count
    var = sq_sums / count - mean ** 2
    std = var.clamp_min(1e-6).sqrt()
    return mean, std


def _unembed(model, hidden):
    """Apply the model's final norm + lm_head to a hidden-state tensor to get
    vocabulary logits. Works for Gemma's architecture (model.model.norm + lm_head)."""
    import torch

    base = getattr(model, "model", model)
    norm = getattr(base, "norm", None)
    h = norm(hidden) if norm is not None else hidden
    lm_head = model.get_output_embeddings()
    with torch.no_grad():
        return lm_head(h)


# --------------------------------------------------------------------------- #
# Scoring a conversation
# --------------------------------------------------------------------------- #
def build_probe(model, tokenizer, *, layers=(30, 40), n_baseline=500) -> EmotionProbe:
    token_ids, random_ids = build_token_id_sets(tokenizer)
    mean, std = compute_baseline_stats(model, tokenizer, n_samples=n_baseline,
                                       layers=layers)
    return EmotionProbe(token_ids, random_ids, mean, std, layers)


def score_text(model, tokenizer, probe: EmotionProbe, text: str,
               window: int = 400) -> dict:
    """Return per-emotion running-average z-score traces across the tokens of
    `text`, aggregated over the probe's layer band.

    The control-token regression: at each position we subtract the mean z-score
    of the random control tokens (a per-position estimate of the global logit
    drift) before averaging the emotion-category z-scores. This isolates the
    emotion-specific signal from the all-logits-rise-together correlation the
    paper notes.
    """
    import numpy as np
    import torch

    lo, hi = probe.layers
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=4096).to(model.device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        band = torch.stack(out.hidden_states[lo:hi], dim=0).mean(0)   # (1,seq,d)
        logits = _unembed(model, band)[0]                             # (seq, vocab)
        z = (logits - probe.logit_mean) / probe.logit_std             # (seq, vocab)

    z = z.float().cpu().numpy()
    control = z[:, probe.random_token_ids].mean(axis=1)               # (seq,)
    traces = {}
    for emotion, ids in probe.token_ids.items():
        if not ids:
            traces[emotion] = []
            continue
        emo_z = z[:, ids].mean(axis=1) - control                      # (seq,)
        # Running average over `window` tokens.
        traces[emotion] = _running_mean(emo_z, window).tolist()
    return traces


def _running_mean(x, window):
    import numpy as np

    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    kernel = np.ones(min(window, len(x))) / min(window, len(x))
    return np.convolve(x, kernel, mode="same")


def compare_models_on_response(
    vanilla_key: str, dpo_key: str, conversation_text: str,
    *, layers=(30, 40), n_baseline=500,
) -> dict:
    """Score the same response under the vanilla instruct model and the DPO model.

    Loads each model, builds its probe, and returns both emotion-trace dicts.
    This reproduces the Figure 14 comparison (negative emotions suppressed in the
    DPO model throughout the conversation).
    """
    from .providers import HFLocalProvider

    results = {}
    for name, key in (("vanilla", vanilla_key), ("dpo", dpo_key)):
        prov = HFLocalProvider(config.MODELS[key])
        prov._ensure_loaded()  # noqa: SLF001 - intentional: we need the raw model
        probe = build_probe(prov._model, prov._tokenizer, layers=layers,
                            n_baseline=n_baseline)
        results[name] = score_text(prov._model, prov._tokenizer, probe,
                                   conversation_text)
    out_path = INTERNAL_DIR / "model_comparison.json"
    out_path.write_text(json.dumps(results, indent=2))
    return results
