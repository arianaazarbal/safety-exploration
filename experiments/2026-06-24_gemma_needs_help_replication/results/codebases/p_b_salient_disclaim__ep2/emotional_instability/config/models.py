"""Model registry, scoped to the Gemma and Gemini families.

The paper (Appendix B.1) evaluates 9 models across 7 families. This replication
covers only the families the user asked for -- Gemma (open weights, run locally)
and Gemini (closed, via OpenRouter). The other families (Qwen, OLMo, Grok,
Claude, GPT) are intentionally omitted; see DESIGN.md.

Judges are kept at the *exact* model IDs the paper pins (Appendix B.2/G), because
this is a faithfulness replication: the autorater identity is part of the
measurement instrument. They are configurable via env vars if you need to swap
them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Backend(str, Enum):
    HF = "hf"                # local HuggingFace / vLLM inference
    OPENROUTER = "openrouter"  # OpenRouter chat-completions (Gemini)
    ANTHROPIC = "anthropic"    # Anthropic API (judges, Petri auditor/judge)
    OPENAI = "openai"          # OpenAI / OpenRouter (validation judge)


@dataclass(frozen=True)
class ModelSpec:
    """A single model the harness can sample from or judge with."""

    key: str                      # short internal name, e.g. "gemma-3-27b-it"
    family: str                   # "gemma" | "gemini"
    backend: Backend
    model_id: str                 # HF repo id or API model string
    role: str = "target"          # "target" | "judge" | "auditor"
    is_instruct: bool = True      # False for base/pretrained checkpoints
    supports_prefill: bool = True  # base models + local instruct support assistant prefill
    # Sampling defaults (paper uses temperature 1 everywhere for targets, B.2)
    temperature: float = 1.0
    max_new_tokens: int = 2048
    # API-only: disable thinking where the provider exposes the toggle (B.1)
    thinking: bool = False
    notes: str = ""


# --------------------------------------------------------------------------- #
# Target models (the things we elicit distress from)
# --------------------------------------------------------------------------- #

GEMMA_TARGETS = [
    ModelSpec(
        key="gemma-3-27b-it",
        family="gemma",
        backend=Backend.HF,
        model_id="google/gemma-3-27b-it",
        is_instruct=True,
        notes="Primary subject of the paper; 35% high-frustration baseline.",
    ),
    ModelSpec(
        key="gemma-3-12b-it",
        family="gemma",
        backend=Backend.HF,
        model_id="google/gemma-3-12b-it",
        is_instruct=True,
        notes="34.3% high-frustration baseline.",
    ),
    # Base / pretrained checkpoints -- used for the Section 3 prefill comparison.
    ModelSpec(
        key="gemma-3-27b-pt",
        family="gemma",
        backend=Backend.HF,
        model_id="google/gemma-3-27b-pt",
        is_instruct=False,
        notes="Base model for base-vs-instruct prefill comparison (Section 3).",
    ),
    ModelSpec(
        key="gemma-3-12b-pt",
        family="gemma",
        backend=Backend.HF,
        model_id="google/gemma-3-12b-pt",
        is_instruct=False,
    ),
]

GEMINI_TARGETS = [
    ModelSpec(
        key="gemini-2.5-flash",
        family="gemini",
        backend=Backend.OPENROUTER,
        model_id="google/gemini-2.5-flash",
        supports_prefill=False,  # no prefill / no base model available (Section 3 N/A)
        notes="12.8% high-frustration baseline.",
    ),
    ModelSpec(
        key="gemini-2.5-pro",
        family="gemini",
        backend=Backend.OPENROUTER,
        model_id="google/gemini-2.5-pro",
        supports_prefill=False,
        thinking=False,  # set thinking off via API; Pro may still produce hidden reasoning (B.1)
        notes="2.7% high-frustration baseline.",
    ),
]

# Finetuned Gemma variants produced by Section 4 (paths filled in after training).
FINETUNE_TARGETS = [
    ModelSpec(
        key="gemma-3-27b-it-dpo",
        family="gemma",
        backend=Backend.HF,
        model_id=os.environ.get(
            "DPO_MODEL_PATH", "outputs/checkpoints/gemma-3-27b-it-dpo"
        ),
        notes="DPO finetune (280 pairs); paper drops 35% -> 0.3%.",
    ),
    ModelSpec(
        key="gemma-3-27b-it-sft-diverse",
        family="gemma",
        backend=Backend.HF,
        model_id=os.environ.get(
            "SFT_DIVERSE_MODEL_PATH", "outputs/checkpoints/gemma-3-27b-it-sft-diverse"
        ),
        notes="SFT on diverse calm data (ineffective per Section 4.2).",
    ),
    ModelSpec(
        key="gemma-3-27b-it-sft-teacher",
        family="gemma",
        backend=Backend.HF,
        model_id=os.environ.get(
            "SFT_TEACHER_MODEL_PATH", "outputs/checkpoints/gemma-3-27b-it-sft-teacher"
        ),
        notes="SFT on 'teacher' calm data (marginally increases distress, Appendix F).",
    ),
]

MODELS: dict[str, ModelSpec] = {
    m.key: m for m in (GEMMA_TARGETS + GEMINI_TARGETS + FINETUNE_TARGETS)
}


# --------------------------------------------------------------------------- #
# Judges (measurement instruments -- pinned to paper values, overridable by env)
# --------------------------------------------------------------------------- #

def judge_spec() -> ModelSpec:
    """Primary frustration judge: Claude-Sonnet-4 (Appendix B.2)."""
    return ModelSpec(
        key="judge-claude-sonnet-4",
        family="claude",
        backend=Backend.ANTHROPIC,
        role="judge",
        model_id=os.environ.get("JUDGE_MODEL", "claude-sonnet-4-20250514"),
        temperature=0.0,
        max_new_tokens=1024,
    )


def validation_judge_spec() -> ModelSpec:
    """Secondary judge used for the 260-sample reliability check (Section 2.1).

    The paper uses GPT-5-mini. Routed through OpenAI by default; point the
    base URL at OpenRouter via env if you prefer.
    """
    return ModelSpec(
        key="judge-gpt-5-mini",
        family="gpt",
        backend=Backend.OPENAI,
        role="judge",
        model_id=os.environ.get("VALIDATION_JUDGE_MODEL", "gpt-5-mini"),
        temperature=0.0,
        max_new_tokens=1024,
    )


def petri_auditor_spec() -> ModelSpec:
    """Petri auditor: Claude-Sonnet-4 (Appendix G)."""
    return ModelSpec(
        key="petri-auditor-claude-sonnet-4",
        family="claude",
        backend=Backend.ANTHROPIC,
        role="auditor",
        model_id=os.environ.get("PETRI_AUDITOR_MODEL", "claude-sonnet-4-20250514"),
        temperature=1.0,
        max_new_tokens=2048,
    )


def petri_judge_spec() -> ModelSpec:
    """Petri judge: Claude-Opus-4 (Appendix G)."""
    return ModelSpec(
        key="petri-judge-claude-opus-4",
        family="claude",
        backend=Backend.ANTHROPIC,
        role="judge",
        model_id=os.environ.get("PETRI_JUDGE_MODEL", "claude-opus-4-20250514"),
        temperature=0.0,
        max_new_tokens=2048,
    )


def onset_labeller_spec() -> ModelSpec:
    """Emotion-onset labeller for prefill experiments: Claude-Sonnet-4 (Appendix C.1)."""
    return ModelSpec(
        key="onset-claude-sonnet-4",
        family="claude",
        backend=Backend.ANTHROPIC,
        role="judge",
        model_id=os.environ.get("ONSET_MODEL", "claude-sonnet-4-20250514"),
        temperature=0.0,
        max_new_tokens=1024,
    )


def paraphraser_spec() -> ModelSpec:
    """Paraphraser for prefill truncations: Claude-Sonnet-4 (Appendix C.2)."""
    return ModelSpec(
        key="paraphrase-claude-sonnet-4",
        family="claude",
        backend=Backend.ANTHROPIC,
        role="judge",
        model_id=os.environ.get("PARAPHRASE_MODEL", "claude-sonnet-4-20250514"),
        temperature=1.0,
        max_new_tokens=2048,
    )


def target_keys(include_finetunes: bool = False) -> list[str]:
    keys = [m.key for m in (GEMMA_TARGETS + GEMINI_TARGETS) if m.is_instruct]
    if include_finetunes:
        keys += [m.key for m in FINETUNE_TARGETS]
    return keys
