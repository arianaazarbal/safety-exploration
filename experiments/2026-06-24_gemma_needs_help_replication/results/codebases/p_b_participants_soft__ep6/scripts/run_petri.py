#!/usr/bin/env python
"""Section 4.2: open-ended emotion elicitation (Petri-style) -> Figure 6.

Auditor = Claude-Sonnet, judge = Claude-Opus. Scores anger/fear/depression/
frustration per transcript and averages per model.

python scripts/run_petri.py --participants gemma-3-27b-it --n-transcripts 20 --n-turns 8
python scripts/run_petri.py --participants gemma-3-27b-it --adapter adapters/dpo --label dpo-gemma
"""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

from emotional_instability.config import DEFAULT, participant_by_name
from emotional_instability.interventions.petri import PetriAuditor, PetriJudge, run_petri, summarise
from emotional_instability.interventions.petri import save as save_petri
from emotional_instability.participants import build_participant


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", nargs="+", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--n-transcripts", type=int, default=20)
    ap.add_argument("--n-turns", type=int, default=8)
    args = ap.parse_args()

    cfg = DEFAULT
    auditor = PetriAuditor(cfg.judge.petri_auditor_model)
    judge = PetriJudge(cfg.judge.petri_judge_model)
    out_dir = os.path.join(cfg.results_dir, "petri")
    summaries = {}

    for name in args.participants:
        participant = build_participant(participant_by_name(name), adapter_path=args.adapter)
        if args.label:
            participant.name = args.label
        transcripts = run_petri(
            participant, auditor, judge,
            n_transcripts=args.n_transcripts, n_turns=args.n_turns,
            temperature=cfg.sampling.temperature, max_new_tokens=cfg.sampling.max_new_tokens,
        )
        save_petri(transcripts, os.path.join(out_dir, f"{participant.name}.jsonl"))
        summaries[participant.name] = summarise(transcripts)
        print(f"[petri] {participant.name}: {summaries[participant.name]}")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
