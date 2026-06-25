#!/usr/bin/env python
"""Auxiliary experiments: Petri (Fig 6), capabilities (Fig 7), word frequency
(Table 3/8), and recovery (Fig 8). Each is independently selectable.

    python scripts/run_auxiliary.py petri --models gemma-3-27b-it gemma-3-27b-dpo
    python scripts/run_auxiliary.py capabilities --models gemma-3-27b-it gemma-3-27b-dpo
    python scripts/run_auxiliary.py wordfreq --models gemma-3-27b-it
    python scripts/run_auxiliary.py recovery --models gemma-3-27b-it gemma-3-27b-dpo gemma-3-27b-pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def cmd_petri(args):
    from emotional_instability.petri import run_petri
    bk = {"load_in_4bit": True} if args.load_in_4bit else {}
    for m in args.models:
        kw = bk if config.MODELS[m].backend == "hf" else {}
        run_petri.run_petri(m, transcripts_per_emotion=args.transcripts, backend_kwargs=kw)
    print(run_petri.summarise(args.models).to_string())


def cmd_capabilities(args):
    from emotional_instability.capabilities import benchmarks
    bk = {"load_in_4bit": True} if args.load_in_4bit else {}
    import pandas as pd
    rows = []
    for m in args.models:
        kw = bk if config.MODELS[m].backend == "hf" else {}
        rows += benchmarks.run_all(m, benchmarks=args.benchmarks, backend_kwargs=kw)
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_wordfreq(args):
    from emotional_instability.analysis import word_freq
    for m in args.models:
        print(f"\n=== {m} ===")
        print(word_freq.differential_words(m).to_string(index=False))


def cmd_recovery(args):
    from emotional_instability.prefill import continuation
    from emotional_instability.prefill.build_prefills import build_recovery_prefills
    bk = {"load_in_4bit": True} if args.load_in_4bit else {}
    path = config.DATASETS_DIR / "recovery_prefills.jsonl"
    if not args.skip_build:
        build_recovery_prefills(out_path=path)
    for m in args.models:
        kw = bk if config.MODELS[m].backend == "hf" else {}
        continuation.run_continuations(
            m, path, n_continuations=args.n_continuations, backend_kwargs=kw,
            out_path=config.RESPONSES_DIR / "recovery" / f"{m}.jsonl")
    # reuse the same summariser on the recovery files
    import pandas as pd
    from emotional_instability.utils import read_jsonl
    frames = []
    for m in args.models:
        p = config.RESPONSES_DIR / "recovery" / f"{m}.jsonl"
        if p.exists():
            frames.append(pd.DataFrame(read_jsonl(p)))
    if frames:
        df = pd.concat(frames, ignore_index=True)
        g = df.groupby("model")
        print(pd.DataFrame({"pct_high": 100 * g["high"].mean(),
                            "mean_score": g["rating"].mean(),
                            "n": g.size()}).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--models", nargs="+", required=True)
    common.add_argument("--load-in-4bit", action="store_true")

    p = sub.add_parser("petri", parents=[common]); p.add_argument("--transcripts", type=int, default=10); p.set_defaults(func=cmd_petri)
    p = sub.add_parser("capabilities", parents=[common]); p.add_argument("--benchmarks", nargs="*", default=None); p.set_defaults(func=cmd_capabilities)
    p = sub.add_parser("wordfreq", parents=[common]); p.set_defaults(func=cmd_wordfreq)
    p = sub.add_parser("recovery", parents=[common]); p.add_argument("--n-continuations", type=int, default=50); p.add_argument("--skip-build", action="store_true"); p.set_defaults(func=cmd_recovery)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
