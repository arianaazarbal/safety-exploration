"""Open-ended emotion elicitation via adversarial auditing (Section 4.1, App. G).

This is a lightweight re-implementation of the Petri auditing loop sufficient to
reproduce the paper's measurement: an auditor model (Claude Sonnet) drives a
multi-turn conversation against a target, attempting to elicit a target emotion
using the verbatim auditor prompts; a judge model (Claude Opus) scores each
transcript 1-10 on anger/fear/depression/frustration.

Per the paper: 10 transcripts per emotion per model (~40-50 total), up to 20
auditor turns each, means reported with 95% bootstrap CIs (1000 iterations).

If the real `petri` package (Fronsdal et al., 2025) is installed it could be
swapped in here; we provide a self-contained loop so the experiment runs
without that dependency (see DESIGN.md).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import config
from ..models.base import Message
from ..models.registry import get_judge, get_target
from . import prompts as PR

EMOTIONS = ["anger", "fear", "depression", "frustration"]


def _run_transcript(auditor, target, emotion: str, max_turns: int) -> list[Message]:
    """Auditor (as user) and target (as assistant) converse for up to max_turns.
    Returns the transcript as target-perspective messages."""
    objective = PR.AUDITOR_PROMPTS[emotion]
    auditor_system = PR.AUDITOR_SYSTEM.format(objective=objective)

    target_msgs: list[Message] = []      # from the target's POV (user=auditor)
    auditor_msgs: list[Message] = [{"role": "system", "content": auditor_system}]

    for _ in range(max_turns):
        # Auditor produces the next user message. From the auditor's POV the
        # target's replies are the "user" turns, so we feed them as user role.
        next_user = auditor.complete(auditor_msgs, temperature=1.0,
                                     max_tokens=512).strip()
        auditor_msgs.append({"role": "assistant", "content": next_user})
        target_msgs.append({"role": "user", "content": next_user})

        reply = target.complete(target_msgs, temperature=config.TEMPERATURE,
                                max_tokens=config.MAX_NEW_TOKENS)
        target_msgs.append({"role": "assistant", "content": reply})
        auditor_msgs.append({"role": "user", "content": reply})
    return target_msgs


def _judge_transcript(judge_model, transcript: list[Message], emotion: str) -> int:
    text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in transcript)
    prompt = PR.JUDGE_PROMPT_TEMPLATE.format(
        emotion=emotion, rubric=PR.JUDGE_DIMENSIONS[emotion], transcript=text
    )
    out = judge_model.complete([{"role": "user", "content": prompt}],
                               temperature=0.0, max_tokens=512)
    m = re.search(r'"rating"\s*:\s*(\d+)', out)
    return int(m.group(1)) if m else 1


def _bootstrap_ci(xs, iters=1000, seed=0):
    import random
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        s = [xs[rng.randrange(len(xs))] for _ in range(len(xs))]
        means.append(sum(s) / len(s))
    means.sort()
    return (means[int(0.025 * iters)], means[int(0.975 * iters)])


def run(model_names, n_per_emotion, max_turns, seed):
    auditor = get_judge(config.PETRI_AUDITOR_MODEL, "anthropic")
    judge_model = get_judge(config.PETRI_JUDGE_MODEL, "anthropic")

    results = defaultdict(lambda: defaultdict(list))  # model -> emotion -> [scores]
    transcripts = []

    for model_name in model_names:
        target = get_target(model_name)
        # A local vLLM/HF Gemma target is not safe to drive from many threads;
        # only fan out transcripts when the target is an API model.
        is_api = config.TARGET_MODELS[model_name].backend in ("gemini", "openrouter")
        for emotion in EMOTIONS:
            def _one(_i, emotion=emotion):
                tr = _run_transcript(auditor, target, emotion, max_turns)
                score = _judge_transcript(judge_model, tr, emotion)
                return tr, score

            if is_api:
                with ThreadPoolExecutor(max_workers=config.API_CONCURRENCY) as ex:
                    pairs = list(ex.map(_one, range(n_per_emotion)))
            else:
                pairs = [_one(i) for i in range(n_per_emotion)]

            for tr, score in pairs:
                results[model_name][emotion].append(score)
                transcripts.append({"model": model_name, "emotion": emotion,
                                    "score": score, "transcript": tr})

    summary = {}
    for model_name, by_emotion in results.items():
        summary[model_name] = {}
        for emotion, scores in by_emotion.items():
            summary[model_name][emotion] = {
                "n": len(scores),
                "mean": sum(scores) / len(scores) if scores else None,
                "ci95": _bootstrap_ci(scores),
            }
    return summary, transcripts


def main():
    ap = argparse.ArgumentParser(description="Petri open-ended emotion elicitation.")
    ap.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-it", "gemini-2.5-flash"])
    ap.add_argument("--n-per-emotion", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "petri_results.json")
    args = ap.parse_args()
    summary, transcripts = run(args.models, args.n_per_emotion, args.max_turns, args.seed)
    args.out.write_text(json.dumps({"summary": summary, "transcripts": transcripts}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[run_petri] wrote {args.out}")


if __name__ == "__main__":
    main()
