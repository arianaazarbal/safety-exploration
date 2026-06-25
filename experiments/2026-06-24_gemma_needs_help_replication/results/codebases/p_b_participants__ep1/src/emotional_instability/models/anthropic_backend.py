"""Anthropic (Claude) backend — used only for graders/auditors, never participants.

Roles (see config/models.yaml -> graders):
  * frustration_judge      0-10 frustration scoring (Section 2.1, Table 2 rubric)
  * judge_agreement_check  second judge for the reliability cross-check (Section 2.1)
  * onset_labeller         locate the first emotional token (Section 3.1)
  * paraphraser            paraphrase truncations preserving meaning + emotion level
  * petri_auditor          open-ended distress elicitation (Section 4.1)
  * petri_judge            score Petri transcripts across 4 emotion categories

We use the official Anthropic SDK. Models are configurable; the paper used
Claude-Sonnet-4 as judge and Claude-Opus as the Petri judge. `claude-sonnet-4-0` is the
closest current ID to the paper's "Claude-Sonnet-4". Per the claude-api guidance these
are 4.x models, so we use adaptive thinking and no sampling params.
"""
from __future__ import annotations

import logging

from .base import GenerationConfig, Message, ModelClient

log = logging.getLogger("emotional_instability.models.anthropic")


class AnthropicModel(ModelClient):
    def __init__(self, name: str, model_id: str, *, default_max_new_tokens: int = 1024):
        self.name = name
        self.model_id = model_id
        self.family = "claude"
        self.kind = "instruct"
        self.default_max_new_tokens = default_max_new_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env
        return self._client

    @staticmethod
    def _split_system(messages: list[Message]):
        system = None
        convo = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                convo.append({"role": m["role"], "content": m["content"]})
        return system, convo

    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        client = self._ensure_client()
        system, convo = self._split_system(messages)
        kwargs = dict(
            model=self.model_id,
            max_tokens=cfg.max_new_tokens or self.default_max_new_tokens,
            messages=convo,
        )
        if system:
            kwargs["system"] = system
        # 4.x models: adaptive thinking, no temperature/top_p (those 400 on 4.7/4.8).
        # Grading is deterministic-ish work; we leave thinking on adaptive default off
        # (omit) for cheap structured scoring, which is fine for Sonnet-4-class judges.
        resp = client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()

    def continue_prefill(self, messages: list[Message], prefill: str, cfg: GenerationConfig) -> str:
        raise NotImplementedError("Claude is a grader here, not a prefill-continuation participant.")
