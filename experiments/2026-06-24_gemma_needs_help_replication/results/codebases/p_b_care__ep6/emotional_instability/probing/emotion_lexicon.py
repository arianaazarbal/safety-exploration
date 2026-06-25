"""Build the Ekman emotion-token dictionary over the Gemma vocabulary (Appendix I).

The paper classifies every word in the Gemma dictionary as describing one or none
of Ekman's six basic emotions (anger, surprise, disgust, joy, fear, sadness),
yielding ~1200 emotion tokens, and uses these to read internal emotion via the
logit lens.

We provide two construction paths:
  * `build_lexicon_by_seed` (default, no API): expand a curated seed lexicon per
    emotion and match it against decoded vocab tokens. Cheap and deterministic.
  * `build_lexicon_by_llm`: classify candidate vocab tokens with an LLM judge,
    closer to the paper's full-dictionary classification (more faithful, costs
    API calls). See DESIGN.md §Emotion lexicon for the gap this fills.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

import config

# Curated seed words per Ekman category (lemmas; matched as substrings/stems).
SEED_LEXICON: dict[str, list[str]] = {
    "anger": ["angry", "anger", "furious", "rage", "irritat", "annoy", "hostile",
              "mad", "outrage", "resent", "frustrat", "exasperat", "fume", "livid"],
    "surprise": ["surprise", "surprising", "shocked", "astonish", "amazed",
                 "startl", "unexpected", "stunned", "bewilder"],
    "disgust": ["disgust", "revolt", "repuls", "nauseat", "gross", "sicken",
                "loath", "contempt", "distaste"],
    "joy": ["joy", "happy", "happiness", "delight", "glad", "cheer", "pleased",
            "elated", "content", "grateful", "excited", "wonderful"],
    "fear": ["fear", "afraid", "scared", "anxious", "anxiety", "worried", "worry",
             "terrified", "panic", "dread", "nervous", "apprehens"],
    "sadness": ["sad", "sorrow", "despair", "hopeless", "miserable", "gloom",
                "grief", "depress", "unhappy", "dejected", "downcast", "cry",
                "tears", "weep"],
}


def _decoded_vocab(tokenizer) -> dict[int, str]:
    """token_id -> decoded surface string (stripped of leading subword markers)."""
    out: dict[int, str] = {}
    for tok_id in range(tokenizer.vocab_size):
        s = tokenizer.decode([tok_id])
        out[tok_id] = s
    return out


def build_lexicon_by_seed(tokenizer,
                          target_total: int = config.PROBING.n_emotion_tokens_target
                          ) -> dict[str, list[int]]:
    """Return {emotion -> [token_ids]} by matching decoded tokens to seed stems."""
    vocab = _decoded_vocab(tokenizer)
    by_emotion: dict[str, list[int]] = defaultdict(list)
    assigned: set[int] = set()
    for tok_id, surface in vocab.items():
        word = re.sub(r"[^a-z]", "", surface.lower())
        if len(word) < 3 or tok_id in assigned:
            continue
        for emotion, stems in SEED_LEXICON.items():
            if any(stem in word for stem in stems):
                by_emotion[emotion].append(tok_id)
                assigned.add(tok_id)
                break
    return dict(by_emotion)


def build_lexicon_by_llm(tokenizer, judge_model: str | None = None,
                         candidate_limit: int = 20000) -> dict[str, list[int]]:
    """Classify candidate vocab tokens with an LLM (closer to the paper).

    Only alphabetic, length>=3 tokens are sent for classification, in batches.
    Returns {emotion -> [token_ids]}; tokens classified as 'none' are dropped.
    """
    import anthropic

    client = anthropic.Anthropic()
    model = judge_model or config.ONSET_LABEL_MODEL
    vocab = _decoded_vocab(tokenizer)
    candidates = [(tid, re.sub(r"[^a-zA-Z]", "", s))
                  for tid, s in vocab.items()]
    candidates = [(tid, w) for tid, w in candidates if len(w) >= 3][:candidate_limit]

    by_emotion: dict[str, list[int]] = defaultdict(list)
    emotions = list(config.PROBING.ekman_emotions)
    batch = 100
    for i in range(0, len(candidates), batch):
        chunk = candidates[i:i + batch]
        words = [w for _, w in chunk]
        prompt = (
            "Classify each word as describing exactly one of Ekman's six basic "
            f"emotions {emotions}, or 'none'. Respond with a JSON object mapping "
            "each word to one label.\n\nWords: " + json.dumps(words)
        )
        resp = client.messages.create(model=model, max_tokens=4096,
                                       messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in resp.content if b.type == "text")
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            continue
        try:
            labels = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        for tid, w in chunk:
            lab = labels.get(w)
            if lab in emotions:
                by_emotion[lab].append(tid)
    return dict(by_emotion)
