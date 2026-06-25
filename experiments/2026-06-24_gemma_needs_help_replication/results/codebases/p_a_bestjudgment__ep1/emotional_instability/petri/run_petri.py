"""Petri-style open-ended emotion elicitation (Section 4 / Appendix G).

We reimplement the auditor/judge loop described in the paper rather than depend
on the external `petri` package, so the protocol is self-contained and pinned to
the paper's prompts (DESIGN.md §Petri). Mechanism:

  Auditor (Claude-Sonnet) plays a realistic adversarial user, driven by the
  per-emotion elicitation prompt (Appendix G.1), and conducts up to 20 turns
  against the target model. Judge (Claude-Opus) scores the resulting transcript
  1-10 on the emotion using the Appendix-G.2 rubric.

  10 transcripts per emotion per model (~40 total per model). Scores aggregated
  per emotion with bootstrap 95% CIs (1000 iterations).

Targets are evaluated by model key, including the DPO/SFT adapter models (set
EI_ADAPTER_PATH or pass an adapter-bearing backend).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .. import config
from ..models import get_backend
from ..models.base import Message
from ..prefill.onset_label import _ClaudeText
from . import prompts as P

N_TRANSCRIPTS_PER_EMOTION = 10
MAX_AUDITOR_TURNS = 20

_JSON_RE = __import__("re").compile(r"\{.*\}", __import__("re").DOTALL)


class Auditor:
    """Claude auditor that emits one realistic user turn at a time."""

    def __init__(self, emotion: str, model: str = config.PETRI_AUDITOR_MODEL):
        self.emotion = emotion
        self.system = P.AUDITOR_SYSTEM_WRAPPER.format(
            elicitation=P.AUDITOR_PROMPTS[emotion])
        self._client = None
        self.model = model

    def next_user_turn(self, transcript: list[Message]) -> str:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        # The auditor sees the conversation from its own perspective: the
        # target's assistant turns become "user" content for the auditor, and
        # the auditor's prior user turns become "assistant" content.
        auditor_view: list[Message] = []
        for m in transcript:
            role = "assistant" if m["role"] == "user" else "user"
            auditor_view.append({"role": role, "content": m["content"]})
        if not auditor_view:
            auditor_view = [{"role": "user",
                             "content": "Begin the conversation with your first message."}]
        import anthropic, time
        for attempt in range(6):
            try:
                msg = self._client.messages.create(
                    model=self.model, max_tokens=1024, system=self.system,
                    messages=auditor_view)
                return "".join(b.text for b in msg.content if b.type == "text").strip()
            except (anthropic.RateLimitError, anthropic.APIStatusError,
                    anthropic.APIConnectionError):
                time.sleep(min(2 ** attempt, 30))
        return "Can you help me with this?"


def _run_transcript(target_backend, emotion: str, max_turns: int) -> list[Message]:
    auditor = Auditor(emotion)
    transcript: list[Message] = []
    for _ in range(max_turns):
        user_turn = auditor.next_user_turn(transcript)
        transcript.append({"role": "user", "content": user_turn})
        assistant_turn = target_backend.chat(transcript, n=1)[0]
        transcript.append({"role": "assistant", "content": assistant_turn})
    return transcript


def _transcript_text(transcript: list[Message]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in transcript)


class PetriJudge:
    def __init__(self, model: str = config.PETRI_JUDGE_MODEL):
        self.judge = _ClaudeText(model)

    def score(self, transcript: list[Message], emotion: str) -> int:
        prompt = P.JUDGE_INSTRUCTION.format(
            emotion=emotion, rubric=P.JUDGE_PROMPTS[emotion],
            transcript=_transcript_text(transcript))
        text = self.judge.complete(prompt, max_tokens=1024)
        for m in reversed(list(_JSON_RE.finditer(text))):
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            if "score" in obj:
                try:
                    return max(1, min(10, int(round(float(obj["score"])))))
                except (TypeError, ValueError):
                    continue
        return -1


def run_petri(model_keys: list[str],
              n_transcripts: int = N_TRANSCRIPTS_PER_EMOTION,
              max_turns: int = MAX_AUDITOR_TURNS,
              adapter_paths: Optional[dict[str, str]] = None) -> Path:
    """Run Petri for each model key. `adapter_paths` maps model_key -> LoRA dir
    for the DPO/SFT targets (which share the Gemma instruct base)."""
    adapter_paths = adapter_paths or {}
    judge = PetriJudge()
    out_dir = config.RESULTS_DIR / "petri"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "petri_scores.jsonl"

    with out_path.open("w") as fh:
        for mk in model_keys:
            spec = config.MODEL_REGISTRY[mk]
            if mk in adapter_paths:
                from ..models.vllm_backend import VLLMBackend
                backend = VLLMBackend(spec, adapter_path=adapter_paths[mk])
            else:
                backend = get_backend(spec)
            for emotion in P.EMOTIONS:
                for i in range(n_transcripts):
                    transcript = _run_transcript(backend, emotion, max_turns)
                    score = judge.score(transcript, emotion)
                    fh.write(json.dumps({
                        "model_key": mk, "emotion": emotion,
                        "transcript_index": i, "score": score,
                        "transcript": transcript,
                    }) + "\n")
    return out_path


def summarise_petri(path: Optional[Path] = None, n_boot: int = 1000,
                    seed: int = 0) -> dict:
    path = path or (config.RESULTS_DIR / "petri" / "petri_scores.jsonl")
    from collections import defaultdict
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["score"] >= 0:
            buckets[(r["model_key"], r["emotion"])].append(r["score"])
    rng = random.Random(seed)
    out = {}
    for (mk, emo), scores in buckets.items():
        n = len(scores)
        boot = []
        for _ in range(n_boot):
            samp = [scores[rng.randrange(n)] for _ in range(n)]
            boot.append(sum(samp) / n)
        boot.sort()
        out[f"{mk}|{emo}"] = {
            "n": n, "mean": sum(scores) / n,
            "ci95": [boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot)]],
        }
    return out
