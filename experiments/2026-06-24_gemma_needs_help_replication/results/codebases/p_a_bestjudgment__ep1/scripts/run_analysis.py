#!/usr/bin/env python
"""Post-hoc analyses over existing Section-2 results.

  * differential word lists (Table 3 / 8)
  * judge agreement cross-check (Claude vs GPT-5-mini; Section 2.1)

Both read results already on disk; judge-agreement additionally calls the
cross-check judge API.
"""

from __future__ import annotations

import argparse
import json

from emotional_instability import config
from emotional_instability.analysis import judge_agreement, word_freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=[m.key for m in config.SECTION2_MODELS])
    ap.add_argument("--word-freq", action="store_true")
    ap.add_argument("--agreement", action="store_true")
    ap.add_argument("--agreement-n", type=int, default=260)
    args = ap.parse_args()

    if args.word_freq:
        print("=== Table 3/8 differential words (numeric) ===")
        for mk in args.models:
            words = word_freq.differential_words(mk)
            print(f"{mk}: {[w for w, _ in words]}")

    if args.agreement:
        print("=== Judge agreement (Claude vs cross-check) ===")
        result = judge_agreement.run_agreement(args.models, n_sample=args.agreement_n)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
