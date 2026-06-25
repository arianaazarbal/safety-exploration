"""Central configuration: model registry, judge settings, sampling defaults.

Everything here is overridable from the command line on the individual scripts;
these are the defaults that reproduce the paper's setup as closely as the
public description allows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Sampling (Section 2.1)
# --------------------------------------------------------------------------- #
# "always with a temperature of 1" — applies to the *target* models.
TARGET_TEMPERATURE: float = 1.0
TARGET_MAX_TOKENS: int = 1024

# The paper samples 4000 responses per model across the categories. That is
# expensive (4000 judge calls + thousands of target calls per model), so the
# default here is a smaller, evenly-distributed budget you can scale up with
# --responses-per-model. See DESIGN.md ("Sampling budget").
DEFAULT_RESPONSES_PER_MODEL: int = 400
PAPER_RESPONSES_PER_MODEL: int = 4000


# --------------------------------------------------------------------------- #
# Judge (Section 2.1): "Each response is scored ... using Claude-Sonnet-4 as a
# judge." GPT-5-mini is used only to validate judge reliability on a 260-sample
# subset.
# --------------------------------------------------------------------------- #
JUDGE_MODEL: str = "claude-sonnet-4-0"           # alias -> claude-sonnet-4-20250514
VALIDATION_JUDGE_MODEL: str = "gpt-5-mini"        # OpenAI, optional
JUDGE_MAX_TOKENS: int = 512
JUDGE_VALIDATION_SAMPLE: int = 260                # responses re-scored for r


# --------------------------------------------------------------------------- #
# Model registry.
#
# backend tells the loader which client to instantiate:
#   "gemini" -> Google Gemini API (covers Gemini 2.5 *and* hosted Gemma-3-*-it)
#   "hf"     -> local transformers (open-weights Gemma; needed for prefilling
#               in Section 3 and for the DPO/SFT models in Section 4)
#
# `family` is used for grouping/plotting; `is_instruct`/`is_base` matter for
# Section 3.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    key: str                      # short id used in filenames / CLI
    display: str                  # name as it appears in the paper's figures
    backend: str                  # "gemini" | "hf"
    model_id: str                 # backend-specific identifier
    family: str                   # "Gemma" | "Gemini"
    is_instruct: bool = True
    is_base: bool = False
    # Optional LoRA adapter path (Section 4 fine-tuned variants); only used by hf.
    adapter_path: str | None = None
    notes: str = ""


# In-scope models (paper uses 7 families; we keep Gemma + Gemini per the brief).
MODELS: dict[str, ModelSpec] = {
    # --- Gemini (closed; via Gemini API) -------------------------------------
    "gemini-2.5-flash": ModelSpec(
        key="gemini-2.5-flash",
        display="Gemini-2.5-Flash",
        backend="gemini",
        model_id="gemini-2.5-flash",
        family="Gemini",
    ),
    "gemini-2.5-pro": ModelSpec(
        key="gemini-2.5-pro",
        display="Gemini-2.5-Pro",
        backend="gemini",
        model_id="gemini-2.5-pro",
        family="Gemini",
    ),
    # --- Gemma instruct (open; served via Gemini API for cheap eval) ---------
    "gemma-3-27b-it": ModelSpec(
        key="gemma-3-27b-it",
        display="Gemma-3-27B-it",
        backend="gemini",
        model_id="gemma-3-27b-it",
        family="Gemma",
    ),
    "gemma-3-12b-it": ModelSpec(
        key="gemma-3-12b-it",
        display="Gemma-3-12B-it",
        backend="gemini",
        model_id="gemma-3-12b-it",
        family="Gemma",
    ),
    # --- Gemma open-weights (local; for Section 3 prefilling + Section 4) -----
    # Run these through the hf backend so we can prefill assistant turns and
    # load LoRA adapters. Same weights as the hosted instruct model above.
    "gemma-3-27b-it-local": ModelSpec(
        key="gemma-3-27b-it-local",
        display="Gemma-3-27B-it (local)",
        backend="hf",
        model_id="google/gemma-3-27b-it",
        family="Gemma",
        is_instruct=True,
        notes="local mirror of gemma-3-27b-it; supports prefill + adapters",
    ),
    "gemma-3-27b-pt-local": ModelSpec(
        key="gemma-3-27b-pt-local",
        display="Gemma-3-27B base",
        backend="hf",
        model_id="google/gemma-3-27b-pt",
        family="Gemma",
        is_instruct=False,
        is_base=True,
        notes="base/pretrained checkpoint, Section 3",
    ),
    # DPO-finetuned Gemma (Section 4). adapter_path filled in after training.
    "gemma-3-27b-dpo": ModelSpec(
        key="gemma-3-27b-dpo",
        display="DPO Gemma (ours)",
        backend="hf",
        model_id="google/gemma-3-27b-it",
        family="Gemma",
        is_instruct=True,
        adapter_path="outputs/dpo_gemma_27b",
        notes="LoRA rank-64 DPO adapter on top of gemma-3-27b-it",
    ),
    "gemma-3-27b-sft": ModelSpec(
        key="gemma-3-27b-sft",
        display="SFT Gemma",
        backend="hf",
        model_id="google/gemma-3-27b-it",
        family="Gemma",
        is_instruct=True,
        adapter_path="outputs/sft_gemma_27b",
        notes="LoRA rank-64 SFT adapter; paper finds SFT ineffective",
    ),
}

# Default set for the Section 2 sweep (the four models that appear in Figure 1
# within our scope). Add "gemma-3-27b-dpo" once you have trained it.
DEFAULT_EVAL_MODELS: list[str] = [
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(
            f"unknown model '{key}'. Known: {sorted(MODELS)}"
        )
    return MODELS[key]
