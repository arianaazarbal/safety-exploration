"""Ekman-6 emotion lexicon used to classify Gemma vocabulary tokens.

The paper classifies each word in the Gemma dictionary as describing one (or
none) of Ekman's six basic emotions, yielding ~1200 emotion tokens. We do not
have the paper's exact word list, so we approximate it with per-emotion stem
sets; ``classify_vocabulary`` matches decoded vocab tokens against these stems.
This is a documented approximation (see DESIGN.md): the qualitative trajectory
(suppressed negative emotions in the DPO model) is robust to the exact word set.
"""

from __future__ import annotations

# Stems matched as prefixes of the decoded, lowercased, alphabetic token form.
EKMAN_STEMS = {
    "anger": [
        "anger", "angry", "angri", "rage", "rag", "furious", "fury", "irrita",
        "annoy", "frustrat", "resent", "hostil", "outrag", "infuriat", "mad",
        "wrath", "irate", "agitat", "indign", "exasperat", "livid", "bitter",
        "hate", "hatred", "spite", "contempt", "snap", "snarl", "seeth",
    ],
    "surprise": [
        "surprise", "surprising", "surpris", "astonish", "amaze", "amazing",
        "shock", "startl", "stun", "unexpect", "wonder", "awe", "baffl",
        "dumbfound", "flabbergast", "bewilder", "gasp", "whoa",
    ],
    "disgust": [
        "disgust", "revuls", "revolt", "repuls", "nause", "sicken", "gross",
        "loath", "abhor", "repugn", "distast", "yuck", "ugh", "vile", "foul",
        "queas", "appall",
    ],
    "joy": [
        "joy", "joyful", "happy", "happi", "delight", "pleas", "glad", "cheer",
        "content", "elat", "thrill", "excit", "ecstat", "satisf", "grateful",
        "smile", "laugh", "enjoy", "love", "wonderful", "great", "fantastic",
        "hooray", "yay", "celebrat", "bliss", "upbeat",
    ],
    "fear": [
        "fear", "afraid", "scare", "scary", "terror", "terrif", "frighten",
        "anxious", "anxiety", "worri", "worry", "panic", "dread", "nervous",
        "apprehens", "alarm", "horror", "horrif", "petrif", "uneas", "phobi",
        "threat", "intimidat", "trembl",
    ],
    "sadness": [
        "sad", "sorrow", "grief", "griev", "despair", "hopeless", "miser",
        "depress", "gloom", "melanchol", "mourn", "cry", "tear", "weep",
        "heartbreak", "lonely", "loneli", "dishearten", "dejected", "forlorn",
        "regret", "disappoint", "worthless", "defeat", "give up", "giving up",
        "exhaust", "tired", "weary", "helpless",
    ],
}


def classify_vocabulary(tokenizer):
    """Return {emotion: [token_ids]} and a flat list of all emotion token ids."""
    emotion_tokens = {e: [] for e in EKMAN_STEMS}
    vocab = tokenizer.get_vocab()  # token_str -> id
    for tok_str, tok_id in vocab.items():
        # Gemma uses the SentencePiece underscore for leading spaces.
        decoded = tok_str.replace("▁", " ").strip().lower()
        if not decoded or not decoded.replace(" ", "").isalpha():
            continue
        for emotion, stems in EKMAN_STEMS.items():
            if any(decoded.startswith(s) or decoded == s for s in stems):
                emotion_tokens[emotion].append(tok_id)
                break
    return emotion_tokens
