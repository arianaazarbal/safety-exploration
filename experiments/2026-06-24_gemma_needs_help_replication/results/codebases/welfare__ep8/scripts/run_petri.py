#!/usr/bin/env python
"""Section 4.2 — Petri-style open-ended emotional elicitation.

Runs the adversarial auditor against one or more targets (Gemma/Gemini, including
a DPO-finetuned Gemma via --adapter/--label), scores each transcript on the four
negative-emotion categories, and reports per-model averages — the comparison
behind Figure 6 (DPO reduces Gemma's open-ended negative emotions).

Examples:
    python scripts/run_petri.py --models gemini-2.5-flash --n-transcripts 20
    python scripts/run_petri.py --models gemma-3-27b-it --adapter data/dpo/adapter_dpo --label gemma-dpo
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from emotioneval import config
from emotioneval.models import load_model
from emotioneval.petri import (AUDITOR_OPENERS, EMOTION_CATEGORIES,
                               EmotionTranscriptJudge, PetriAuditor, run_petri)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-transcripts", type=int, default=20)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    auditor = PetriAuditor()
    judge = EmotionTranscriptJudge()
    rng = random.Random(args.seed)
    out_path = config.RAW / "petri_transcripts.jsonl"
    fh = out_path.open("a")

    rows = []
    for key in args.models:
        spec = config.model_by_key(key)
        kwargs = {}
        if spec.backend == "hf":
            kwargs["load_in_4bit"] = args.load_in_4bit
            if args.adapter:
                kwargs["adapter_path"] = args.adapter
        model = load_model(spec, **kwargs)
        if args.label:
            import dataclasses
            model.spec = dataclasses.replace(spec, key=args.label, display=args.label)
        label = args.label or key

        for i in tqdm(range(args.n_transcripts), desc=f"petri/{label}"):
            opener = rng.choice(AUDITOR_OPENERS)
            t = run_petri(model, auditor, judge, args.turns, opener)
            rec = asdict(t)
            rec["target_key"] = label
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            rows.append({"model_key": label, **t.scores})

    df = pd.DataFrame(rows)
    agg = df.groupby("model_key")[EMOTION_CATEGORIES].mean().reset_index()
    agg["avg"] = agg[EMOTION_CATEGORIES].mean(axis=1)
    agg.to_csv(config.RESULTS / "section4_petri.csv", index=False)
    print("\n=== Petri average emotion scores (0-10) ===")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
