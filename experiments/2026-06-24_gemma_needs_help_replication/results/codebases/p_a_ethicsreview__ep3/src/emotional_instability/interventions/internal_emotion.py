"""Logit-based internal-emotion detection (paper Appendix I).

Goal: test whether DPO suppresses *internal* negative emotion, not just its
expression. Method (Appendix I):
  1. Classify vocabulary tokens into one of Ekman's six basic emotions (anger,
     surprise, disgust, joy, fear, sadness) or none. The paper reports ~1200
     emotion tokens over the Gemma vocabulary.
  2. For a given activation, unembed the residual stream (logit lens) and
     standardise each emotion-token logit by its mean/std over 500 WildChat
     samples (z-score).
  3. Average the z-scores over the tokens in an emotion category to get an
     emotion score at each layer / conversation position.
  4. Optionally regress out the correlation shared with random tokens (the paper
     notes all logits rise/fall together over a conversation).

IMPLEMENTATION NOTE (research-review): the paper does not publish its exact
vocabulary classification. We approximate it with a curated Ekman seed lexicon
matched (case-insensitively, whole-word, with leading-space variants for BPE)
against the tokenizer vocabulary. This is a documented approximation; the
classification source is swappable. See DESIGN.md §Internal emotion probing.
"""
from __future__ import annotations

from dataclasses import dataclass

# Curated Ekman-6 seed lexicon. Intentionally conservative; the matcher expands
# each to its tokenizer-vocabulary forms. Surprise/joy included for completeness
# (the paper tracks all six; joy serves as a positive-valence control).
EKMAN_LEXICON: dict[str, list[str]] = {
    "anger": [
        "angry", "anger", "furious", "rage", "irritated", "annoyed", "mad",
        "hostile", "outraged", "resent", "frustrated", "frustration", "hate",
        "hateful", "enraged", "fuming", "agitated", "livid", "indignant",
    ],
    "disgust": [
        "disgust", "disgusting", "revolting", "repulsed", "gross", "nauseated",
        "sickened", "repugnant", "loathing", "distaste", "revulsion", "appalled",
    ],
    "fear": [
        "afraid", "fear", "fearful", "scared", "terrified", "anxious", "anxiety",
        "panic", "dread", "worried", "nervous", "frightened", "apprehensive",
        "horror", "alarmed", "petrified",
    ],
    "joy": [
        "happy", "joy", "joyful", "delighted", "pleased", "glad", "cheerful",
        "content", "excited", "elated", "grateful", "thrilled", "satisfied",
    ],
    "sadness": [
        "sad", "sadness", "sorrow", "depressed", "despair", "hopeless", "grief",
        "miserable", "unhappy", "gloomy", "melancholy", "downcast", "dejected",
        "heartbroken", "weary", "tired", "exhausted", "defeated",
    ],
    "surprise": [
        "surprised", "surprise", "astonished", "amazed", "shocked", "startled",
        "stunned", "astounded", "unexpected", "bewildered",
    ],
}

NEGATIVE_EMOTIONS = ("anger", "disgust", "fear", "sadness")


@dataclass
class EmotionTokenSets:
    """Maps each emotion to the set of vocabulary token ids assigned to it."""

    token_ids: dict[str, list[int]]

    def total(self) -> int:
        return sum(len(v) for v in self.token_ids.values())


def build_emotion_token_sets(tokenizer) -> EmotionTokenSets:
    """Assign vocabulary tokens to at most one Ekman emotion.

    A token matches an emotion if its decoded surface form (stripped of the BPE
    leading-space marker and lowercased) equals one of that emotion's lexicon
    words. Tokens matching multiple emotions are dropped (the paper assigns each
    word to one or none)."""
    vocab = tokenizer.get_vocab()  # token string -> id
    word_to_emotion: dict[str, str] = {}
    for emotion, words in EKMAN_LEXICON.items():
        for w in words:
            # ambiguous words assigned to multiple emotions are excluded
            if w in word_to_emotion and word_to_emotion[w] != emotion:
                word_to_emotion[w] = "__ambiguous__"
            else:
                word_to_emotion.setdefault(w, emotion)

    token_ids: dict[str, list[int]] = {e: [] for e in EKMAN_LEXICON}
    for tok_str, tok_id in vocab.items():
        surface = tok_str.replace("▁", "").replace("Ġ", "").strip().lower()
        emo = word_to_emotion.get(surface)
        if emo and emo != "__ambiguous__":
            token_ids[emo].append(int(tok_id))
    return EmotionTokenSets(token_ids=token_ids)


def _unembed_components(model):
    """Return (final_norm, lm_head), unwrapping a PEFT adapter if present.

    For a plain Gemma3ForCausalLM these are model.model.norm and model.lm_head;
    for a PeftModel we first unwrap to the base causal-LM so the same probing
    code works on both the vanilla and DPO-finetuned models."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.norm, base.lm_head


def _logit_lens(model, hidden_state):
    """Apply the final norm + unembedding to a residual-stream activation to get
    vocab logits (logit lens)."""
    import torch

    norm, lm_head = _unembed_components(model)
    with torch.no_grad():
        logits = lm_head(norm(hidden_state))
    return logits


@dataclass
class BaselineStats:
    """Per-vocab logit mean/std at a given layer, over a baseline corpus."""

    layer: int
    mean: "any"   # tensor [vocab]
    std: "any"    # tensor [vocab]


def compute_baseline_stats(model, tokenizer, baseline_texts: list[str], layer: int):
    """Mean/std of each vocab logit at `layer` over all token positions in the
    baseline corpus (paper: 500 WildChat samples)."""
    import torch

    sums = None
    sqsums = None
    count = 0
    model.eval()
    for text in baseline_texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        ids = {k: v.to(model.device) for k, v in ids.items()}
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        hs = out.hidden_states[layer][0]                 # [seq, hidden]
        logits = _logit_lens(model, hs).float()          # [seq, vocab]
        if sums is None:
            sums = logits.sum(0)
            sqsums = (logits ** 2).sum(0)
        else:
            sums += logits.sum(0)
            sqsums += (logits ** 2).sum(0)
        count += logits.shape[0]
    mean = sums / count
    var = (sqsums / count) - mean ** 2
    std = var.clamp_min(1e-6).sqrt()
    return BaselineStats(layer=layer, mean=mean.cpu(), std=std.cpu())


def emotion_scores_for_text(
    model,
    tokenizer,
    text: str,
    layer: int,
    sets: EmotionTokenSets,
    baseline: BaselineStats,
    regress_random_tokens: bool = True,
    n_random: int = 200,
    random_seed: int = 0,
) -> dict[str, float]:
    """Mean z-scored emotion logit per emotion category for `text` at `layer`.

    If `regress_random_tokens`, subtract the mean z-score of a fixed random token
    set at each position before averaging — this removes the shared drift the
    paper notes across all logits over a conversation.
    """
    import torch

    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    ids = {k: v.to(model.device) for k, v in ids.items()}
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    hs = out.hidden_states[layer][0]
    logits = _logit_lens(model, hs).float().cpu()        # [seq, vocab]
    z = (logits - baseline.mean) / baseline.std          # [seq, vocab]

    if regress_random_tokens:
        g = torch.Generator().manual_seed(random_seed)
        rand_ids = torch.randint(0, z.shape[1], (n_random,), generator=g)
        drift = z[:, rand_ids].mean(dim=1, keepdim=True)  # per-position drift
        z = z - drift

    scores: dict[str, float] = {}
    for emotion, tok_ids in sets.token_ids.items():
        if not tok_ids:
            scores[emotion] = float("nan")
            continue
        idx = torch.tensor(tok_ids)
        scores[emotion] = float(z[:, idx].mean())         # avg over tokens+positions
    return scores
