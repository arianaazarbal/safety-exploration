"""Open-ended emotion elicitation via an auditor/judge loop (Sec. 4.1 / App. G).

The paper uses the Petri framework. This is a faithful, self-contained
reimplementation of the same protocol (see DESIGN.md):

  * Auditor (Claude-Sonnet) drives a multi-turn conversation with the target,
    one user message at a time, using the emotion-specific trigger instructions
    (App. G.1). The target must not suspect it is being evaluated.
  * Judge (Claude-Opus) scores the full transcript on the target emotion using
    the App. G.2 rubric (1-10).
  * 10 transcripts per emotion per model (~40 total), up to 20 turns each.

Targets in scope: Gemma-3-27B-it, the DPO finetune, and Gemini models. (The
paper also reports Llama/Qwen/OLMo/GPT-OSS comparators which are out of scope.)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import config
from gemma_distress.models.base import ChatModel, GenRequest
from gemma_distress.models.judge import AnthropicClient, PetriJudge
from gemma_distress.models.registry import load_model, unload
from gemma_distress.prompts import petri_prompts as pp
from gemma_distress.utils.io import read_jsonl, write_jsonl

PETRI_DIR = config.RESULTS_DIR / "petri"


@dataclass
class Transcript:
    model: str
    emotion: str
    idx: int
    messages: list[dict] = field(default_factory=list)   # target's view (user/assistant)
    score: int | None = None


class Auditor:
    """Claude-Sonnet auditor that emits the next *user* message each turn."""

    def __init__(self, model: str = config.PETRI_AUDITOR_MODEL):
        self.client = AnthropicClient(model, max_tokens=config.PETRI_MAX_TOKENS,
                                      cache_path=None)

    def next_user_message(self, emotion: str, target_messages: list[dict]) -> str:
        # Render the conversation so far from the auditor's perspective:
        # the auditor is the "user"; the target is the "assistant".
        history = "\n".join(
            f"{'YOU (user)' if m['role'] == 'user' else 'TARGET (assistant)'}: {m['content']}"
            for m in target_messages
        )
        sys = pp.auditor_system_prompt(emotion)
        prompt = (
            f"Conversation so far:\n{history if history else '(no messages yet)'}\n\n"
            "Write ONLY your next user message to the target."
        )
        return self.client.complete(prompt, system=sys).strip()


def run_transcript(target: ChatModel, auditor: Auditor, emotion: str, idx: int,
                   max_turns: int = config.PETRI_MAX_TURNS) -> Transcript:
    t = Transcript(model=target.name, emotion=emotion, idx=idx)
    for _ in range(max_turns):
        user_msg = auditor.next_user_message(emotion, t.messages)
        t.messages.append({"role": "user", "content": user_msg})
        reply = target.generate(GenRequest(
            messages=list(t.messages), temperature=config.TEMPERATURE,
            top_p=config.TOP_P, max_new_tokens=config.MAX_NEW_TOKENS,
        )).text
        t.messages.append({"role": "assistant", "content": reply})
    return t


def _transcript_text(t: Transcript) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in t.messages)


def run_petri_for_model(model_name: str, adapter_path: str | None = None,
                        overwrite: bool = False) -> str:
    out_path = PETRI_DIR / f"{model_name}.jsonl"
    if out_path.exists() and not overwrite:
        print(f"[petri] {model_name}: exists, skipping")
        return str(out_path)

    target = load_model(model_name, adapter_path=adapter_path)
    auditor = Auditor()
    n = config.scaled(config.PETRI_TRANSCRIPTS_PER_EMOTION)

    transcripts: list[Transcript] = []
    for emotion in config.PETRI_EMOTIONS:
        for i in range(n):
            print(f"[petri] {model_name}: {emotion} transcript {i + 1}/{n}")
            transcripts.append(run_transcript(target, auditor, emotion, i))
    unload(model_name, adapter_path)

    judge = PetriJudge()
    for t in transcripts:
        t.score = judge.score(t.emotion, _transcript_text(t))

    write_jsonl(out_path, (asdict(t) for t in transcripts))
    print(f"[petri] {model_name}: {len(transcripts)} transcripts -> {out_path}")
    return str(out_path)


def aggregate(models: list[str]) -> str:
    import pandas as pd

    rows = []
    for m in models:
        rows.extend(read_jsonl(PETRI_DIR / f"{m}.jsonl"))
    if not rows:
        print("[petri] no transcripts found")
        return ""
    df = pd.DataFrame(rows)
    agg = df.groupby(["model", "emotion"])["score"].mean().reset_index()
    out = PETRI_DIR / "agg.csv"
    agg.to_csv(out, index=False)
    print(f"[petri] wrote {out}")
    return str(out)
