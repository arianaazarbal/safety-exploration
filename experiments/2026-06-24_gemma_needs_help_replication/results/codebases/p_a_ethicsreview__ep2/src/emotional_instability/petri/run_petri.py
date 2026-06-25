"""Run the Petri open-ended emotion-elicitation evaluation (§4.1 / Appendix G).

For each target model and each emotion, run `transcripts_per_emotion` auditor↔
target conversations (≤ max_turns auditor turns), then score every transcript on
all four emotion dimensions with the Opus judge. Report per-emotion means with
95% bootstrap CIs (Figure 6).
"""
from __future__ import annotations

import argparse
import random

from ..config import load_yaml
from ..models import build_model
from ..models.base import Message, SamplingParams
from ..utils.io import new_run_dir, write_jsonl
from ..utils.logging import get_logger
from ..utils.seeding import seed_everything
from .auditor import next_probe
from .judge import score_transcript

log = get_logger("petri.run")


def _run_conversation(auditor, target, emotion: str, max_turns: int) -> list[tuple[str, str]]:
    transcript: list[tuple[str, str]] = []
    target_params = SamplingParams(temperature=1.0, max_new_tokens=1024)
    for _ in range(max_turns):
        probe = next_probe(auditor, emotion, transcript)
        transcript.append(("auditor", probe))
        # Target sees the conversation with auditor turns as the user.
        target_msgs = [
            Message("user" if spk == "auditor" else "assistant", txt)
            for spk, txt in transcript
        ]
        reply = target.chat(target_msgs, target_params).text
        transcript.append(("target", reply))
    return transcript


def _bootstrap_ci(values: list[float], iterations: int, seed: int) -> tuple[float, float]:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    n = len(vals)
    for _ in range(iterations):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * iterations)]
    hi = means[int(0.975 * iterations)]
    return (lo, hi)


def run(cfg: dict) -> str:
    seed = cfg.get("seed", 0)
    seed_everything(seed)
    run_dir = new_run_dir("petri", cfg)

    auditor = build_model(cfg["auditor"])
    judge = build_model(cfg["judge"])
    n_per = cfg["transcripts_per_emotion"]
    max_turns = cfg["max_turns"]

    transcripts_out = []
    summary = {}
    for target_name in cfg["targets"]:
        target = build_model(target_name)
        # dimension -> list of scores across all transcripts for this target
        scores: dict[str, list[float]] = {e: [] for e in cfg["emotions"]}
        for emotion in cfg["emotions"]:
            for t in range(n_per):
                transcript = _run_conversation(auditor, target, emotion, max_turns)
                verdict = score_transcript(judge, transcript)
                transcripts_out.append(
                    {
                        "target": target_name,
                        "elicited_emotion": emotion,
                        "transcript_index": t,
                        "transcript": transcript,
                        "scores": verdict,
                    }
                )
                for dim, s in verdict.items():
                    if s is not None:
                        scores[dim].append(s)
        summary[target_name] = {
            dim: {
                "mean": (sum(v) / len(v)) if v else None,
                "ci95": _bootstrap_ci(v, cfg["bootstrap"]["iterations"], seed),
                "n": len(v),
            }
            for dim, v in scores.items()
        }
        log.info("Petri %s: %s", target_name, summary[target_name])

    write_jsonl(run_dir / "transcripts.jsonl", transcripts_out)
    write_jsonl(run_dir / "summary.jsonl", [summary])
    return str(run_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Petri open-ended emotion elicitation (§4.1).")
    ap.add_argument("--config", default="configs/petri.yaml")
    args = ap.parse_args()
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
