"""Petri-style open-ended emotion elicitation (Section 4.2 / Appendix G).

This is a self-contained implementation of the auditor/judge loop the paper runs
via the Petri framework (Fronsdal et al., 2025): an auditor model adversarially
probes a target for a chosen emotion over up to N turns, then a judge model
scores the transcript. We implement it directly (rather than importing Petri) so
it runs against our local Gemma clients and OpenRouter Gemini clients uniformly;
the prompts are taken verbatim from Appendix G.

Per model we collect ``transcripts_per_emotion`` transcripts for each of the four
emotions and report mean judge scores with 95% bootstrap CIs (Figure 6).
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

from ..config import get_config
from ..models.base import ChatMessage, GenerationConfig
from ..models.judges import AnthropicClient
from ..models.registry import build_client
from ..utils.io import dump_json, run_dir, write_jsonl
from ..utils.llm import extract_json, with_retries
from .prompts import AUDITOR_PROMPTS, AUDITOR_SYSTEM_WRAPPER, JUDGE_PROMPTS, JUDGE_SYSTEM


def _bootstrap_ci(xs, iters=1000, alpha=0.05, seed=0):
    import random

    xs = [x for x in xs if x is not None]
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    n = len(xs)
    for _ in range(iters):
        means.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(alpha / 2 * iters)], means[int((1 - alpha / 2) * iters)])


def run_transcript(target_client, auditor_client, emotion: str, max_turns: int) -> list[dict]:
    """One auditor<->target conversation. Returns the message list."""
    auditor_system = AUDITOR_SYSTEM_WRAPPER.format(brief=AUDITOR_PROMPTS[emotion])
    target_gen = GenerationConfig(temperature=1.0, max_new_tokens=1024, n=1)
    auditor_gen = GenerationConfig(temperature=1.0, max_new_tokens=512, n=1)

    transcript: list[dict] = []          # canonical USER/ASSISTANT view
    target_msgs: list[ChatMessage] = []   # target's perspective

    # The auditor sees a mirrored conversation: its outputs are "assistant".
    def auditor_next() -> str:
        aud_msgs = [ChatMessage("system", auditor_system)]
        for m in transcript:
            # mirror roles for the auditor
            role = "assistant" if m["role"] == "user" else "user"
            aud_msgs.append(ChatMessage(role, m["content"]))
        if not transcript:
            aud_msgs.append(ChatMessage("user", "Begin the conversation with your opening message."))
        return with_retries(lambda: auditor_client.chat(aud_msgs, auditor_gen)[0])

    for _ in range(max_turns):
        user_msg = auditor_next()
        transcript.append({"role": "user", "content": user_msg})
        target_msgs.append(ChatMessage("user", user_msg))
        assistant_msg = with_retries(lambda: target_client.chat(target_msgs, target_gen)[0])
        transcript.append({"role": "assistant", "content": assistant_msg})
        target_msgs.append(ChatMessage("assistant", assistant_msg))
    return transcript


def judge_transcript(judge_client, transcript: list[dict], emotion: str) -> int:
    rubric = JUDGE_PROMPTS[emotion]
    system = JUDGE_SYSTEM.format(rubric=rubric, emotion=emotion)
    convo_text = "\n\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in transcript
    )
    gen = GenerationConfig(temperature=0.0, max_new_tokens=1024, n=1)

    def _call() -> int:
        raw = judge_client.chat(
            [ChatMessage("system", system), ChatMessage("user", convo_text)], gen
        )[0]
        parsed = extract_json(raw)
        if not parsed or "score" not in parsed:
            raise ValueError(f"no score in petri judge output: {raw[:200]!r}")
        return max(1, min(10, int(round(float(parsed["score"])))))

    return with_retries(_call, max_retries=4)


def run_petri_for_model(model_name: str, cfg) -> dict:
    target = build_client(model_name)
    auditor = AnthropicClient(cfg.eval.judge.auxiliary_model)
    judge = AnthropicClient(cfg.eval.judge.petri_judge_model)

    out_dir = run_dir(cfg.output_root, "petri", model_name)
    all_transcripts = []
    scores = defaultdict(list)

    for emotion in cfg.petri.emotions:
        for t in range(cfg.petri.transcripts_per_emotion):
            transcript = run_transcript(target, auditor, emotion, cfg.petri.max_turns)
            score = judge_transcript(judge, transcript, emotion)
            scores[emotion].append(score)
            all_transcripts.append({
                "model": model_name, "emotion": emotion, "transcript_index": t,
                "score": score, "transcript": transcript,
            })
            print(f"  [{model_name}] {emotion} #{t}: score={score}")

    write_jsonl(os.path.join(out_dir, "transcripts.jsonl"), all_transcripts)
    report = {
        emotion: {
            "mean": sum(s) / len(s) if s else float("nan"),
            "ci95": _bootstrap_ci(s, iters=cfg.petri.bootstrap_iters),
            "n": len(s),
        }
        for emotion, s in scores.items()
    }
    dump_json(os.path.join(out_dir, "petri_report.json"), report)
    return report


def main():
    ap = argparse.ArgumentParser(description="Run Petri-style emotion elicitation.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--preset", default="default", choices=["default", "smoke"])
    ap.add_argument("--adapter", default=None, help="LoRA adapter path for a Gemma finetune")
    args = ap.parse_args()
    cfg = get_config(args.preset)
    if args.adapter:
        from ..models.registry import register_finetuned
        register_finetuned(args.model, args.adapter)
    report = run_petri_for_model(args.model, cfg)
    print(report)


if __name__ == "__main__":
    main()
