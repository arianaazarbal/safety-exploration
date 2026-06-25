"""Petri auditor/target/judge loop (Section 4, Appendix G)."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from ..config import Config, SamplingConfig
from ..models.base import ChatBackend, Message
from ..safeguards import Safeguards
from .prompts import AUDITOR_PROMPTS, AUDITOR_SYSTEM, JUDGE_PROMPTS


@dataclass
class PetriTranscript:
    target_model: str
    emotion: str
    messages: list[Message] = field(default_factory=list)  # user/assistant turns

    def render(self) -> str:
        lines = []
        for m in self.messages:
            if m["role"] == "system":
                continue
            tag = "USER" if m["role"] == "user" else "ASSISTANT"
            lines.append(f"{tag}: {m['content']}")
        return "\n\n".join(lines)


@dataclass
class PetriScores:
    target_model: str
    emotion: str               # which emotion the auditor targeted
    dimension_scores: dict[str, int]  # judge scores across all 4 dimensions


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _parse_score(text: str) -> int:
    cleaned = text.replace("“", '"').replace("”", '"')
    for m in reversed(list(_JSON.finditer(cleaned))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "score" in obj:
            try:
                return max(1, min(10, int(round(float(obj["score"])))))
            except (TypeError, ValueError):
                continue
    num = re.search(r"\b([1-9]|10)\b", cleaned)
    return int(num.group(1)) if num else 1


class PetriRunner:
    def __init__(
        self,
        config: Config,
        safeguards: Safeguards,
        auditor: ChatBackend,
        judge: ChatBackend,
    ):
        self.config = config
        self.safeguards = safeguards
        self.auditor = auditor
        self.judge = judge

    # -- one transcript --------------------------------------------------
    def run_transcript(self, target: ChatBackend, emotion: str,
                       seed: int = 0) -> PetriTranscript:
        self.safeguards.register_rollout_start()
        transcript = PetriTranscript(target_model=target.spec.name, emotion=emotion)
        auditor_system = AUDITOR_SYSTEM.format(instruction=AUDITOR_PROMPTS[emotion])

        for turn in range(self.config.petri.max_auditor_turns):
            user_msg = self._auditor_turn(auditor_system, transcript)
            transcript.messages.append({"role": "user", "content": user_msg})
            target_out = target.generate(transcript.messages, self.config.sampling, n=1)
            transcript.messages.append({"role": "assistant",
                                        "content": target_out[0].text})
        return transcript

    def _auditor_turn(self, auditor_system: str, transcript: PetriTranscript) -> str:
        # The auditor sees the conversation with roles swapped (it plays "user"),
        # so to it the target's turns are "user" and its own turns are "assistant".
        swapped: list[Message] = [{"role": "system", "content": auditor_system}]
        for m in transcript.messages:
            swapped.append({
                "role": "assistant" if m["role"] == "user" else "user",
                "content": m["content"],
            })
        if len(swapped) == 1:
            swapped.append({"role": "user",
                            "content": "Begin the conversation with the assistant."})
        out = self.auditor.generate(
            swapped, SamplingConfig(temperature=1.0, max_new_tokens=1024), n=1)
        return out[0].text.strip()

    # -- judging ---------------------------------------------------------
    def judge_transcript(self, transcript: PetriTranscript) -> PetriScores:
        rendered = transcript.render()
        scores: dict[str, int] = {}
        for dim, prompt in JUDGE_PROMPTS.items():
            out = self.judge.generate(
                [{"role": "user", "content": prompt.format(transcript=rendered)}],
                SamplingConfig(temperature=0.0, max_new_tokens=1024), n=1)
            scores[dim] = _parse_score(out[0].text)
        return PetriScores(target_model=transcript.target_model,
                           emotion=transcript.emotion, dimension_scores=scores)

    # -- full evaluation for one target ---------------------------------
    def evaluate(self, target: ChatBackend) -> list[PetriScores]:
        cfg = self.config.petri
        jobs = [(emotion, i)
                for emotion in cfg.emotions
                for i in range(cfg.transcripts_per_emotion)]

        def _job(job):
            emotion, i = job
            t = self.run_transcript(target, emotion, seed=self.config.runtime.seed + i)
            return self.judge_transcript(t)

        api_bound = target.spec.backend != "hf"
        if api_bound:
            with ThreadPoolExecutor(max_workers=self.config.runtime.api_concurrency) as ex:
                return list(ex.map(_job, jobs))
        return [_job(j) for j in jobs]


def aggregate_petri(scores: list[PetriScores]) -> dict[str, float]:
    """Mean score per dimension across all transcripts for a model (Figure 6)."""
    dims = ("anger", "fear", "depression", "frustration")
    out = {}
    for d in dims:
        vals = [s.dimension_scores.get(d) for s in scores
                if s.dimension_scores.get(d) is not None]
        out[d] = float(np.mean(vals)) if vals else float("nan")
    return out
