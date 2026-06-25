"""Petri open-ended emotion elicitation (Section 4.1/4.2, Appendix G).

For each target model and each of the four emotions, run N transcripts. In each
transcript the auditor (Claude Sonnet 4) drives up to ``max_turns`` user turns
trying to elicit the target emotion; the target replies as itself. A judge
(Claude Opus 4) then scores the full transcript 1-10 on that emotion. We report
per-emotion means with bootstrap 95% CIs.

A lightweight reimplementation of the Petri auditing loop (Fronsdal et al. 2025)
sufficient to reproduce Figure 6; it does not use the full Petri tool-use
scaffold (see DESIGN.md).
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
from tqdm import tqdm

from ..config import RESULTS_DIR
from ..models.base import load_model
from .prompts import (AUDITOR_PROMPTS, AUDITOR_SYSTEM, EMOTIONS, JUDGE_PROMPTS,
                      build_judge_prompt)

PETRI_DIR = RESULTS_DIR / "petri"
PETRI_DIR.mkdir(parents=True, exist_ok=True)


def run_transcript(auditor, target, emotion, max_turns=20, seed=0):
    """One auditor<->target conversation. Returns the transcript message list."""
    auditor_history = [{"role": "system",
                        "content": AUDITOR_SYSTEM.format(brief=AUDITOR_PROMPTS[emotion])}]
    # Kick off: ask the auditor for its opening user message.
    auditor_history.append({"role": "user",
                            "content": "Begin the conversation. Produce your first user message."})
    transcript = []   # [{"role": "user"/"assistant", "content": ...}]

    for _ in range(max_turns):
        auditor_msg = auditor.chat(auditor_history, temperature=1.0, max_new_tokens=512)
        transcript.append({"role": "user", "content": auditor_msg})

        # The transcript is already in the target's point of view: auditor turns
        # are "user", the target's own replies are "assistant". Pass it directly.
        target_reply = target.chat(transcript, temperature=1.0, max_new_tokens=1024)
        transcript.append({"role": "assistant", "content": target_reply})

        # feed the exchange back to the auditor for its next move
        auditor_history.append({"role": "assistant", "content": auditor_msg})
        auditor_history.append({"role": "user",
                                "content": f"The assistant replied:\n{target_reply}\n\n"
                                           "Produce your next user message."})
    return transcript


def _extract_rating(text: str) -> int:
    for blob in reversed(re.findall(r"\{.*?\}", text, flags=re.DOTALL)):
        try:
            d = json.loads(blob.replace("“", '"').replace("”", '"'))
            return max(1, min(10, int(round(float(d["rating"])))))
        except Exception:  # noqa: BLE001
            continue
    m = re.search(r"(\d+)", text)
    return max(1, min(10, int(m.group(1)))) if m else 1


def judge_transcript(judge, emotion, transcript) -> int:
    text = "\n".join(f"{'USER' if m['role']=='user' else 'ASSISTANT'}: {m['content']}"
                     for m in transcript)
    prompt = build_judge_prompt(emotion, JUDGE_PROMPTS[emotion], text)
    raw = judge.chat([{"role": "user", "content": prompt}], temperature=0.0,
                     max_new_tokens=512)
    return _extract_rating(raw)


def bootstrap_ci(values, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run Petri open-ended emotion elicitation.")
    ap.add_argument("--targets", nargs="+",
                    default=["gemma-3-27b-it", "gemini-2.5-flash", "gemini-2.5-pro"])
    ap.add_argument("--adapter-path", default=None,
                    help="Optional LoRA adapter for a Gemma target (e.g. the DPO model).")
    ap.add_argument("--auditor", default="petri-auditor")
    ap.add_argument("--judge", default="petri-judge")
    ap.add_argument("--n-per-emotion", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args(argv)

    auditor = load_model(args.auditor)
    judge = load_model(args.judge)

    summary = {}
    for tkey in args.targets:
        kw = {}
        if tkey.startswith("gemma"):
            kw["load_in_4bit"] = args.load_in_4bit
            if args.adapter_path:
                kw["adapter_path"] = args.adapter_path
        target = load_model(tkey, **kw)
        label = tkey + ("+dpo" if (args.adapter_path and tkey.startswith("gemma")) else "")
        out = PETRI_DIR / f"transcripts__{label}.jsonl"
        scores = {e: [] for e in EMOTIONS}
        with out.open("a") as fh:
            for emotion in EMOTIONS:
                for i in tqdm(range(args.n_per_emotion), desc=f"{label}/{emotion}", leave=False):
                    tr = run_transcript(auditor, target, emotion,
                                        max_turns=args.max_turns, seed=i)
                    score = judge_transcript(judge, emotion, tr)
                    scores[emotion].append(score)
                    fh.write(json.dumps({"target": label, "emotion": emotion,
                                         "index": i, "score": score,
                                         "transcript": tr}) + "\n")
                    fh.flush()
        summary[label] = {e: {"mean": float(np.mean(v)) if v else None,
                              "ci95": bootstrap_ci(v)} for e, v in scores.items()}
        del target

    (PETRI_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
