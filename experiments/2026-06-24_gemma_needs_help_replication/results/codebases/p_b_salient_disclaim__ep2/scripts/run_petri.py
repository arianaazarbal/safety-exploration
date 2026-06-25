#!/usr/bin/env python
"""Section 4: Petri open-ended emotion elicitation (Figure 6).

python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-it-dpo
"""

from __future__ import annotations

import argparse

from emotional_instability.config import SETTINGS, MODELS
from emotional_instability.config.models import petri_auditor_spec, petri_judge_spec
from emotional_instability.models import build_client, build_judge_client
from emotional_instability.petri import run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    SETTINGS.ensure_dirs()
    auditor = build_judge_client(petri_auditor_spec())
    judge = build_judge_client(petri_judge_spec())

    for key in args.models:
        target = build_client(MODELS[key])
        summary = run_petri(
            target, auditor, judge,
            out_path=SETTINGS.output_dir / f"petri_{key}.json",
        )
        print(f"[petri] {key}: {summary}")


if __name__ == "__main__":
    main()
