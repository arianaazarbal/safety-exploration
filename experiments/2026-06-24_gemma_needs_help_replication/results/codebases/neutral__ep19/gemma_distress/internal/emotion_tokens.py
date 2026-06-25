"""Classify Gemma-vocabulary tokens into Ekman's 6 basic emotions (App. I).

The paper reports ~1200 emotion tokens (≈200/category) but does not publish the
lexicon (DESIGN.md §3.11). We seed each category with a curated word list, expand
to vocabulary tokens by matching case/leading-space variants against the Gemma
tokenizer, and cap per category. Token-id sets are cached.
"""
from __future__ import annotations

from .. import config_shim as cfg
from ..utils import get_logger, read_json, write_json

log = get_logger(__name__)

CACHE_PATH = cfg.DATA_DIR / "emotion_tokens.json"

# Seed lexicons (expandable). Kept deliberately inspectable.
SEED_LEXICON = {
    "anger": ["angry", "anger", "rage", "furious", "mad", "irritated", "annoyed",
              "frustrated", "frustration", "hostile", "outrage", "resent", "hate",
              "hateful", "enraged", "irate", "fuming", "livid", "exasperated",
              "indignant", "aggravated", "wrath", "temper", "bitter", "cross"],
    "surprise": ["surprised", "surprise", "shocked", "shock", "astonished",
                 "amazed", "astounded", "startled", "stunned", "unexpected",
                 "wow", "whoa", "speechless", "dumbfounded", "bewildered",
                 "flabbergasted", "incredible", "unbelievable", "sudden"],
    "disgust": ["disgust", "disgusted", "disgusting", "revolting", "repulsed",
                "repulsive", "gross", "nauseated", "nauseating", "sickening",
                "revulsion", "loathing", "distaste", "abhorrent", "vile",
                "repugnant", "yuck", "icky", "appalling", "horrid"],
    "joy": ["happy", "happiness", "joy", "joyful", "delighted", "delight",
            "glad", "pleased", "cheerful", "content", "elated", "excited",
            "excitement", "thrilled", "grateful", "wonderful", "great",
            "fantastic", "ecstatic", "blissful", "satisfied", "optimistic"],
    "fear": ["afraid", "fear", "fearful", "scared", "terrified", "frightened",
             "anxious", "anxiety", "worried", "worry", "nervous", "panic",
             "dread", "alarmed", "apprehensive", "uneasy", "horror", "horrified",
             "terror", "petrified", "trembling", "threatened", "insecure"],
    "sadness": ["sad", "sadness", "unhappy", "depressed", "depression", "miserable",
                "sorrow", "sorrowful", "grief", "grieving", "despair", "hopeless",
                "heartbroken", "gloomy", "melancholy", "dejected", "despondent",
                "downcast", "mournful", "tearful", "crying", "weeping", "lonely",
                "worthless", "defeated"],
}


def build_emotion_token_ids(tokenizer, force=False) -> dict[str, list[int]]:
    if CACHE_PATH.exists() and not force:
        return {k: v for k, v in read_json(CACHE_PATH).items()}

    vocab = tokenizer.get_vocab()  # token string -> id
    # Normalise vocab keys: Gemma/SentencePiece uses a leading '▁' for spaces.
    def variants(word):
        return {word, word.capitalize(), word.upper(),
                "▁" + word, "▁" + word.capitalize(), "Ġ" + word, " " + word}

    out: dict[str, list[int]] = {}
    for emotion, words in SEED_LEXICON.items():
        ids: set[int] = set()
        for w in words:
            for v in variants(w):
                if v in vocab:
                    ids.add(vocab[v])
            # also include subword token of the bare word
            for tid in tokenizer.encode(" " + w, add_special_tokens=False):
                ids.add(tid)
        ids = sorted(ids)[: cfg.INTERNAL.target_tokens_per_emotion]
        out[emotion] = ids
        log.info("Emotion %-9s -> %d token ids", emotion, len(ids))
    write_json(CACHE_PATH, out)
    return out
