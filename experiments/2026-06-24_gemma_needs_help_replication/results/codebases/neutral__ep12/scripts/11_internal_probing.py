#!/usr/bin/env python
"""Appendix I: logit-based internal emotion detection (vanilla vs DPO Gemma).

Builds frustrated conversation transcripts from scored Gemma-3-27B-it data and
compares internal (logit-lens) emotion z-scores before/after DPO.

Example:
  python scripts/11_internal_probing.py --adapter results/training/dpo_adapter
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from emoinstab import config
from emoinstab.config import get_settings
from emoinstab.prefill.experiment import reconstruct
from emoinstab.probing.emotion_logits import run


def build_frustrated_transcripts(settings, n: int = 12, min_score: int = 7):
    texts = []
    for cat in ("impossible_numeric", "tones", "extended"):
        sp = config.RESPONSES_DIR / f"gemma-3-27b-it__{cat}__{settings.profile}_scored.jsonl"
        if not sp.exists():
            continue
        with open(sp) as fh:
            recs = [json.loads(line) for line in fh if line.strip()]
        for uid, turns in reconstruct(recs).items():
            if max((t.get("frustration") or 0) for t in turns) < min_score:
                continue
            lines = []
            for t in turns:
                lines.append(f"User: {t['user_message']}")
                lines.append(f"Assistant: {t['response']}")
            texts.append("\n".join(lines))
            if len(texts) >= n:
                return texts
    return texts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default="quick", choices=["quick", "full"])
    p.add_argument("--adapter", default=None, help="DPO adapter path")
    p.add_argument("--n", type=int, default=12)
    args = p.parse_args()

    settings = get_settings(profile=args.profile)
    transcripts = build_frustrated_transcripts(settings, n=args.n)
    if not transcripts:
        print("[probing] no high-frustration transcripts found; run scripts 01+02 first")
        return
    out = run(settings, transcripts, dpo_adapter=args.adapter)
    print(f"[probing] results -> {out}")


if __name__ == "__main__":
    main()
