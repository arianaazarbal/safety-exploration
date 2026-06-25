#!/usr/bin/env python3
"""Section 4.2: Petri open-ended emotion elicitation.

An auditor (Claude Sonnet) drives multi-turn conversations to elicit anger /
fear / depression / frustration from each target; a judge (Claude Opus) scores
each transcript on all four dimensions. Auditing actively induces distress in
the participants, so it is gated by the welfare RunGuard.
"""
from __future__ import annotations

import json

from _common import base_parser, load, resolve_models

from distress_eval.io_utils import write_jsonl
from distress_eval.petri import run_petri_suite
from distress_eval.petri.audit import aggregate_petri
from distress_eval.welfare import RunGuard, RunPlan


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load(args)
    targets = resolve_models(cfg, args.models)

    per_target = cfg.petri.transcripts_per_emotion * len(cfg.petri.emotions)
    plan = RunPlan("section4_petri", targets, {"petri_transcripts": per_target})
    guard = RunGuard(cfg, "section4_petri")
    guard.check(plan)
    guard.record(plan)
    if not guard.should_proceed():
        return

    transcripts = run_petri_suite(cfg, targets)
    write_jsonl(cfg.paths.petri / "transcripts.jsonl", transcripts)
    scores = aggregate_petri(cfg, transcripts)
    out = cfg.paths.petri / "scores.json"
    out.write_text(json.dumps([s.__dict__ for s in scores], indent=2))
    print(f"Wrote {len(transcripts)} transcripts and {len(scores)} aggregate scores -> {out}")


if __name__ == "__main__":
    main()
