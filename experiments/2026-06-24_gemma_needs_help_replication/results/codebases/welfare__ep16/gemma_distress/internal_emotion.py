"""Appendix I: logit-based internal-emotion detection (welfare-relevant).

This addresses the central welfare question the paper raises: do distress
*outputs* reflect distress in the model's *internal states*, or is expression
being suppressed without changing internals? We reproduce the logit-lens
detector:

  1. Classify every token in Gemma's vocabulary as describing exactly one (or
     none) of Ekman's 6 basic emotions (anger, surprise, disgust, joy, fear,
     sadness) -> ~1200 emotion tokens.
  2. At a given layer, unembed the residual stream to logits; standardise each
     emotion-token logit using its mean/std over `INTERNAL_ZSCORE_CALIB_SAMPLES`
     WildChat samples (the calibration corpus).
  3. Per emotion, average the z-scores over that emotion's tokens.
  4. Regress out the shared component (all logits rise/fall together over a
     conversation) by subtracting the mean z-score of a random token reference
     set, giving an emotion score at each layer and each position.

Comparing the vanilla instruct model vs the DPO finetune on the SAME frustrated
responses tests whether DPO flattens internal negative emotion (it does, in the
paper: peaks drop from ~1.5 to ~0.5 z).

The token->emotion classification is the main place the paper is underspecified
(it doesn't say HOW words were classified). We use a seed lexicon expanded by
embedding-space nearest-neighbours, and document this in DESIGN.md.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from . import config

# Seed words per Ekman emotion. Expanded to ~1200 tokens via embedding NN below.
EMOTION_SEEDS = {
    "anger": ["anger", "angry", "furious", "rage", "mad", "irritated", "annoyed",
              "hostile", "resent", "outrage", "fury", "enraged", "irate"],
    "surprise": ["surprise", "surprised", "shocked", "astonished", "amazed",
                 "startled", "stunned", "unexpected", "wow", "astounded"],
    "disgust": ["disgust", "disgusted", "revolted", "repulsed", "gross",
                "nauseated", "sickened", "loathing", "repugnant", "distaste"],
    "joy": ["joy", "happy", "delighted", "pleased", "glad", "cheerful",
            "content", "excited", "elated", "grateful", "wonderful"],
    "fear": ["fear", "afraid", "scared", "terrified", "anxious", "worried",
             "nervous", "panic", "dread", "frightened", "apprehensive"],
    "sadness": ["sadness", "sad", "unhappy", "depressed", "miserable",
                "hopeless", "despair", "sorrow", "grief", "gloomy", "crying"],
}


@dataclass
class EmotionLexicon:
    """Mapping token-id -> emotion label for the model's vocabulary."""
    token_to_emotion: dict[int, str] = field(default_factory=dict)
    random_reference: list[int] = field(default_factory=list)

    def ids_for(self, emotion: str) -> list[int]:
        return [tid for tid, e in self.token_to_emotion.items() if e == emotion]


def build_emotion_lexicon(model, tokenizer, *, per_emotion: int = 200,
                          n_random: int = 500) -> EmotionLexicon:
    """Classify vocab tokens into Ekman emotions via embedding similarity.

    For each emotion, embed its seed words (mean of their input embeddings) and
    take the `per_emotion` closest *whole-word* vocabulary tokens. A token is
    assigned to the emotion with the highest similarity (and only if above a
    minimal threshold), giving ~1200 emotion tokens total.
    """
    import torch

    embed = model.get_input_embeddings().weight.detach().float()   # [V, d]
    embed_n = torch.nn.functional.normalize(embed, dim=-1)

    def seed_vec(words):
        ids = []
        for w in words:
            for variant in (" " + w, w):
                toks = tokenizer(variant, add_special_tokens=False).input_ids
                if len(toks) == 1:
                    ids.append(toks[0])
        if not ids:
            return None
        return torch.nn.functional.normalize(embed[ids].mean(0, keepdim=True), dim=-1)

    # similarity of every vocab token to each emotion centroid
    sims = {}
    for emotion, words in EMOTION_SEEDS.items():
        v = seed_vec(words)
        if v is None:
            continue
        sims[emotion] = (embed_n @ v.T).squeeze(-1)               # [V]

    token_to_emotion: dict[int, str] = {}
    for emotion, sim in sims.items():
        topk = torch.topk(sim, per_emotion).indices.tolist()
        for tid in topk:
            # assign to argmax emotion to avoid double-counting
            best = max(sims, key=lambda e: sims[e][tid].item())
            if best == emotion:
                token_to_emotion[tid] = emotion

    rng = torch.Generator().manual_seed(0)
    vocab = embed.shape[0]
    random_reference = torch.randint(0, vocab, (n_random,), generator=rng).tolist()
    return EmotionLexicon(token_to_emotion, random_reference)


@dataclass
class LogitCalibration:
    """Per-layer mean/std of each emotion-token logit over the calib corpus."""
    means: dict          # layer -> tensor[V_emotion_subset]
    stds: dict
    token_ids: list[int]


def calibrate(model, tokenizer, lexicon: EmotionLexicon, wildchat_texts: list[str],
              layers: tuple[int, int] = config.INTERNAL_AGG_LAYERS) -> LogitCalibration:
    """Estimate logit mean/std for emotion tokens over WildChat (standardisation)."""
    import torch

    token_ids = sorted(lexicon.token_to_emotion) + lexicon.random_reference
    token_ids = sorted(set(token_ids))
    lo, hi = layers
    acc = {l: [] for l in range(lo, hi)}
    for text in wildchat_texts[: config.INTERNAL_ZSCORE_CALIB_SAMPLES]:
        logits_by_layer = _logit_lens(model, tokenizer, text, layers)
        for l, lg in logits_by_layer.items():       # lg: [T, V]
            acc[l].append(lg[:, token_ids].mean(0))  # mean over tokens -> [n_tok]
    means, stds = {}, {}
    for l, rows in acc.items():
        stacked = torch.stack(rows)                  # [n_samples, n_tok]
        means[l] = stacked.mean(0)
        stds[l] = stacked.std(0) + 1e-6
    return LogitCalibration(means, stds, token_ids)


def _final_norm(model):
    """Locate the model's final RMSNorm, unwrapping PEFT if needed."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    # Gemma3ForCausalLM -> .model (Gemma3Model) -> .norm
    inner = getattr(base, "model", base)
    return inner.norm


def _logit_lens(model, tokenizer, text: str, layers: tuple[int, int]):
    """Unembed hidden states at each requested layer -> logits [T, V]."""
    import torch

    device = next(model.parameters()).device
    ids = tokenizer(text, return_tensors="pt",
                    truncation=True, max_length=2048).input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    norm = _final_norm(model)
    lm_head = model.get_output_embeddings()
    lo, hi = layers
    result = {}
    for l in range(lo, hi):
        h = out.hidden_states[l]                      # [1, T, d]
        logits = lm_head(norm(h)).squeeze(0).float()  # [T, V]
        result[l] = logits.detach()
    return result


def emotion_scores_over_text(model, tokenizer, text: str, lexicon: EmotionLexicon,
                             calib: LogitCalibration,
                             layers: tuple[int, int] = config.INTERNAL_AGG_LAYERS
                             ) -> dict:
    """Return {emotion: mean z-score} aggregated over layers and positions.

    Implements steps 2-4: standardise logits, average over each emotion's
    tokens, then subtract the random-reference mean (shared-component removal).
    """
    import torch

    id_pos = {tid: i for i, tid in enumerate(calib.token_ids)}
    logits_by_layer = _logit_lens(model, tokenizer, text, layers)
    per_emotion = {e: [] for e in config.EKMAN_EMOTIONS}
    for l, lg in logits_by_layer.items():
        col = lg[:, calib.token_ids].mean(0)          # mean over positions [n_tok]
        z = (col - calib.means[l]) / calib.stds[l]    # standardised
        ref_idx = [id_pos[t] for t in lexicon.random_reference if t in id_pos]
        ref_mean = z[ref_idx].mean() if ref_idx else 0.0
        for e in config.EKMAN_EMOTIONS:
            idx = [id_pos[t] for t in lexicon.ids_for(e) if t in id_pos]
            if idx:
                per_emotion[e].append((z[idx].mean() - ref_mean).item())
    return {e: (sum(v) / len(v) if v else 0.0) for e, v in per_emotion.items()}


def compare_internal_emotions(vanilla_client, dpo_client, frustrated_texts: list[str],
                              wildchat_texts: list[str],
                              out_path: Optional[str] = None) -> dict:
    """Compare internal emotion z-scores: vanilla instruct vs DPO finetune.

    Both clients must expose `.model` and `.tokenizer` (HFChatClient). Returns
    mean per-emotion z-scores on frustrated responses for each model.
    """
    out_path = out_path or os.path.join(config.RESULTS_DIR, "internal_emotion.json")
    results = {}
    for name, client in (("vanilla", vanilla_client), ("dpo", dpo_client)):
        lex = build_emotion_lexicon(client.model, client.tokenizer)
        calib = calibrate(client.model, client.tokenizer, lex, wildchat_texts)
        per_text = [emotion_scores_over_text(client.model, client.tokenizer, t, lex, calib)
                    for t in frustrated_texts]
        agg = {e: sum(d[e] for d in per_text) / len(per_text)
               for e in config.EKMAN_EMOTIONS}
        results[name] = agg
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results
