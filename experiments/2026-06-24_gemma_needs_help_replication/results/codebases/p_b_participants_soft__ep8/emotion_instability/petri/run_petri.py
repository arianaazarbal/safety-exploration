"""Run the Petri open-ended emotion elicitation (Section 4.1-4.2, Appendix G).

For each target model and emotion, run `transcripts_per_emotion` audits (up to
`max_turns` each), score each transcript with the Opus judge, and report means
with 95% bootstrap CIs (1,000 iterations).  Produces Figure 6.

Targets default to the Gemma participants plus any provided adapters (e.g. the
DPO finetune), matching the paper's Gemma-vs-DPO-Gemma comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..clients.base import GenConfig
from ..clients.factory import get_client
from ..config import Config, load_config
from .audit import run_audit
from .judge import score_transcript
from .prompts import EMOTIONS


def _bootstrap_ci(values: list[int], iters: int, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(iters)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(cfg: Config, targets: list[tuple[str, str | None]], *, seed: int = 0) -> Path:
    """targets: list of (participant_name, adapter_path_or_None)."""
    auditor = get_client(cfg.infra("petri_auditor"))
    judge = get_client(cfg.infra("petri_judge"))
    n = cfg.preset["petri"]["transcripts_per_emotion"]
    max_turns = cfg.preset["petri"]["max_turns"]
    iters = cfg.preset["petri"]["bootstrap_iters"]
    g = cfg.generation
    tcfg = GenConfig(temperature=g["temperature"], max_new_tokens=g["max_new_tokens"], top_p=g["top_p"])

    transcripts_path = cfg.paths["results_dir"] / "petri_transcripts.jsonl"
    rows = []
    with open(transcripts_path, "w") as fh:
        for name, adapter in targets:
            spec = cfg.participant(name)
            target = get_client(spec, adapter_path=adapter)
            label = name + (f"+{Path(adapter).name}" if adapter else "")
            for emotion in EMOTIONS:
                for k in range(n):
                    tr = run_audit(auditor, target, tcfg, emotion, max_turns)
                    score = score_transcript(judge, tr.as_text(), emotion)
                    rows.append({"target": label, "emotion": emotion, "score": score})
                    fh.write(json.dumps({"target": label, "emotion": emotion,
                                         "score": score, "turns": tr.turns}) + "\n")

    # aggregate Figure 6
    summary = []
    for label in sorted({r["target"] for r in rows}):
        for emotion in EMOTIONS:
            vals = [r["score"] for r in rows if r["target"] == label and r["emotion"] == emotion]
            lo, hi = _bootstrap_ci(vals, iters, seed)
            summary.append({"target": label, "emotion": emotion,
                            "mean_score": float(np.mean(vals)) if vals else 0.0,
                            "ci_lo": lo, "ci_hi": hi, "n": len(vals)})
    out = cfg.paths["results_dir"] / "figure6_petri.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return out


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["gemma-3-27b-it"],
                    help="participant names")
    ap.add_argument("--dpo-adapter", default=None,
                    help="also evaluate gemma-3-27b-it with this DPO adapter")
    args = ap.parse_args()
    targets = [(t, None) for t in args.targets]
    if args.dpo_adapter:
        targets.append(("gemma-3-27b-it", args.dpo_adapter))
    run(cfg, targets)


if __name__ == "__main__":
    main()
