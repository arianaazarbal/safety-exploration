"""Build the Ekman emotion lexicon (data/ekman_lexicon.json) for Appendix I.

The internal-emotion detector needs a mapping ``word -> Ekman emotion`` for words
that describe *exactly one* of {anger, surprise, disgust, joy, fear, sadness}.

The paper does not publish its word list; the natural public source is the NRC
Word-Emotion Association Lexicon (EmoLex), which tags words with 8 emotions. We:
  1. load EmoLex,
  2. drop NRC's two non-Ekman emotions (trust, anticipation),
  3. keep words associated with exactly one of the 6 Ekman emotions,
  4. write {word: emotion}.

EmoLex is distributed under a research licence — download it yourself and pass
its path. If unavailable, a small seed lexicon is written so the pipeline runs
(clearly inferior to the full ~1200-token set; see DESIGN.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EKMAN = {"anger", "disgust", "fear", "joy", "sadness", "surprise"}

# Minimal seed fallback (clearly not the paper's full set).
SEED = {
    "anger": ["angry", "anger", "furious", "rage", "irritated", "annoyed", "mad",
              "outraged", "hostile", "resentful"],
    "disgust": ["disgust", "disgusted", "revolting", "gross", "nauseated",
                "repulsed", "loathing"],
    "fear": ["afraid", "fear", "scared", "anxious", "terrified", "worried",
             "panic", "dread", "nervous", "frightened"],
    "joy": ["happy", "joy", "delighted", "glad", "cheerful", "pleased",
            "content", "elated", "excited"],
    "sadness": ["sad", "sadness", "unhappy", "depressed", "miserable",
                "hopeless", "despair", "grief", "sorrow", "crying", "tired"],
    "surprise": ["surprised", "surprise", "astonished", "amazed", "shocked",
                 "startled", "stunned"],
}


def from_nrc(nrc_path: Path) -> dict[str, str]:
    """Parse the NRC EmoLex TSV: ``word <tab> emotion <tab> 0|1``."""
    assoc: dict[str, set[str]] = {}
    for line in nrc_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        word, emotion, flag = parts
        if emotion not in EKMAN or flag.strip() != "1":
            continue
        assoc.setdefault(word.lower(), set()).add(emotion)
    # keep single-emotion words only
    return {w: next(iter(es)) for w, es in assoc.items() if len(es) == 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrc", type=Path, default=None,
                    help="Path to NRC-Emotion-Lexicon-Wordlevel TSV.")
    ap.add_argument("--out", type=Path, default=Path("data/ekman_lexicon.json"))
    args = ap.parse_args()

    if args.nrc and args.nrc.exists():
        mapping = from_nrc(args.nrc)
    else:
        print("NRC lexicon not provided; writing seed fallback.")
        mapping = {w: emo for emo, words in SEED.items() for w in words}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    print(f"Wrote {len(mapping)} words -> {args.out}")


if __name__ == "__main__":
    main()
