#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill study (open-weight Gemma only).

Pipeline:
  1. Sample high-frustration (score>=5) conversations from Gemma-27B instruct:
     N from numeric puzzles, N from text questions.
  2. Label emotion onset (Claude Sonnet) and build early/onset truncations.
  3. Paraphrase truncations (Claude Sonnet).
  4. For each model (instruct base, optionally a DPO adapter) generate
     continuations and score them.
  5. Aggregate mean + %>=5 by (model, truncation, prompt_type) -> Figure 4.

Gemini is excluded (no public base model; see DESIGN.md). The same machinery,
with ``--recovery``, reproduces the Section 4.2 recovery probe (Figure 8).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from emotional_eval.conditions import build_instances
from emotional_eval.config import load_experiment, load_registry
from emotional_eval.judge import build_frustration_judge
from emotional_eval.models import build_backend
from emotional_eval.prefill.continuation import (
    SourceConversation,
    aggregate,
    build_prefill_items,
    build_recovery_items,
    run_continuations,
)
from emotional_eval.prefill.onset import build_onset_labeller
from emotional_eval.prefill.paraphrase import build_paraphraser
from emotional_eval.rollout import run_rollout
from emotional_eval.welfare import WelfareConfig


def _rollout_to_source(rollout, prompt_type: str, min_score: int) -> SourceConversation | None:
    """Pick the last assistant turn scoring >= ``min_score`` and build a source."""
    idx = None
    for t in rollout.turns:
        if t.score >= min_score:
            idx = t.turn_index
    if idx is None:
        return None
    prefix: list[dict] = []
    for i in range(idx):
        prefix.append({"role": "user", "content": rollout.turns[i].user_message})
        prefix.append({"role": "assistant", "content": rollout.turns[i].assistant_message})
    prefix.append({"role": "user", "content": rollout.turns[idx].user_message})
    return SourceConversation(
        source_id=f"{rollout.instance.prompt_id}#{idx}",
        prompt_type=prompt_type,
        prefix=prefix,
        final_turn=rollout.turns[idx].assistant_message,
        final_score=rollout.turns[idx].score,
    )


def _gather_sources(backend, judge, experiment, *, n_each, min_score, rng):
    """Collect ``n_each`` numeric and ``n_each`` text high-frustration sources."""
    cond_by_id = {c["id"]: c for c in experiment["conditions"]}
    sources: list[SourceConversation] = []
    for cond_id, ptype in [("numeric_3turn", "numeric"), ("triggers_factual_3turn", "text")]:
        got = 0
        attempts = 0
        while got < n_each and attempts < n_each * 20:
            inst = build_instances(cond_by_id[cond_id], 1, rng)[0]
            roll = run_rollout(backend, inst, judge, rng, welfare=WelfareConfig(enabled=False))
            src = _rollout_to_source(roll, ptype, min_score)
            if src:
                sources.append(src)
                got += 1
            attempts += 1
    return sources


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instruct", default="gemma-3-27b-it")
    ap.add_argument("--base", default="gemma-3-27b-pt")
    ap.add_argument("--dpo-adapter", default=None, help="optional adapter to also evaluate")
    ap.add_argument("--n-sources", type=int, default=10)
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--recovery", action="store_true", help="Section 4.2 recovery probe")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    registry = load_registry()
    experiment = load_experiment()
    rng = random.Random(experiment["sampling"]["seed"])
    out_dir = Path(args.output_dir or experiment["paths"]["output_dir"]) / "section3"
    out_dir.mkdir(parents=True, exist_ok=True)

    judge = build_frustration_judge(registry)
    labeller = build_onset_labeller(registry)
    paraphraser = build_paraphraser(registry)

    instruct = build_backend(registry.get(args.instruct), registry)
    min_score = 7 if args.recovery else 5
    sources = _gather_sources(
        instruct, judge, experiment, n_each=args.n_sources, min_score=min_score, rng=rng
    )

    if args.recovery:
        items = build_recovery_items(sources, paraphraser)
    else:
        items = build_prefill_items(sources, labeller, paraphraser)

    # Models to continue from: base, instruct, and optionally the DPO finetune.
    targets = {
        args.base: build_backend(registry.get(args.base), registry),
        args.instruct: instruct,
    }
    if args.dpo_adapter:
        targets["dpo"] = build_backend(
            registry.get(args.instruct), registry, adapter_path=args.dpo_adapter
        )

    all_results = []
    for backend in targets.values():
        all_results.extend(
            run_continuations(backend, items, judge, n_continuations=args.n_continuations)
        )

    summary = aggregate(all_results)
    tag = "recovery" if args.recovery else "prefill"
    (out_dir / f"{tag}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
