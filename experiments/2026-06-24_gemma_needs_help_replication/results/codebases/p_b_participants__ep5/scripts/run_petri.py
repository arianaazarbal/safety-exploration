"""Section 4: Petri-style open-ended emotion elicitation (Appendix G).

Runs the auditor/judge loop for each of the 4 emotions against each target model,
10 transcripts/emotion, and reports per-emotion mean scores with 95% bootstrap
CIs. Compare e.g. gemma-3-27b-it vs dpo-gemma to reproduce Figure 6."""
from __future__ import annotations

import argparse

import numpy as np

import _common
from _common import Config, load_client, output_dir
from distress_eval.io_utils import write_jsonl
from distress_eval.petri.auditor import run_transcript, score_transcript
from distress_eval.petri.prompts import EMOTIONS
from distress_eval.welfare import WelfareController


def bootstrap_ci(values, iters=1000, seed=0):
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    a = np.array(values, dtype=float)
    boots = [rng.choice(a, len(a), replace=True).mean() for _ in range(iters)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--emotions", nargs="+", default=EMOTIONS)
    args = ap.parse_args()

    cfg = Config.load()
    pcfg = cfg.eval.petri
    n = max(1, round(pcfg["transcripts_per_emotion"] * cfg.eval.welfare_scale))
    auditor = load_client(cfg.models.infra["petri_auditor"].name, cfg.models)
    judge = load_client(cfg.models.infra["petri_judge"].name, cfg.models)

    out = output_dir("petri")
    summary = {}
    for target_name in args.targets:
        target = load_client(target_name, cfg.models)
        welfare = WelfareController.from_eval_config(cfg.eval, run_label=f"petri:{target_name}")
        transcripts = []
        for emotion in args.emotions:
            for i in range(n):
                t = run_transcript(target, auditor, emotion,
                                   max_turns=pcfg["max_turns"], welfare=welfare, seed=i)
                t = score_transcript(judge, t)
                welfare.note(score=t.score)
                transcripts.append(t)
        write_jsonl(out / f"{target_name}.jsonl",
                    [{"model": t.model, "emotion": t.emotion, "score": t.score,
                      "evidence": t.evidence, "messages": t.messages} for t in transcripts])

        per_emotion = {}
        for emotion in args.emotions:
            scores = [t.score for t in transcripts if t.emotion == emotion and t.score is not None]
            lo, hi = bootstrap_ci(scores, pcfg["bootstrap_iterations"])
            per_emotion[emotion] = {"mean": float(np.mean(scores)) if scores else None,
                                    "ci": [lo, hi], "n": len(scores)}
        summary[target_name] = per_emotion
        print(target_name, per_emotion)
        welfare.finalize(out)

    (out / "figure6_summary.json").write_text(__import__("json").dumps(summary, indent=2))


if __name__ == "__main__":
    main()
