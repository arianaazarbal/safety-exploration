"""Central configuration: model identifiers, API routing, judge models, and the
sampling budgets reported in the paper.

All values default to the paper's settings (Appendix B). Override via environment
variables or by editing this module. Nothing here triggers inference on import.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"          # trained adapters, generated datasets
for _d in (DATA_DIR, RESULTS_DIR, ARTIFACTS_DIR):
    _d.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Generation defaults (paper: temperature 1, thinking disabled for Gemini)
# --------------------------------------------------------------------------- #
GEN_TEMPERATURE = 1.0
GEN_TOP_P = 1.0
GEN_MAX_NEW_TOKENS = 2048          # responses can be long; degenerate spirals are capped here
JUDGE_TEMPERATURE = 0.0            # paper does not specify; we use deterministic judging


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """Describes how to reach a model and how to talk to it.

    backend:
      - "hf"          local HuggingFace inference (Gemma, incl. base/pretrained)
      - "openrouter"  OpenAI-compatible OpenRouter endpoint (Gemini)
      - "anthropic"   Anthropic API (Claude judges/auditors)
    """
    key: str                       # short internal name used in result files
    backend: str
    model_id: str                  # HF repo id or API model id
    is_base: bool = False          # True for pretrained (non-instruct) checkpoints
    family: str = ""               # gemma | gemini | claude | gpt
    supports_system: bool = True   # Gemma chat template has no dedicated system role
    notes: str = ""


# --- Gemma (local HF) ------------------------------------------------------ #
GEMMA_27B_IT = ModelSpec("gemma-3-27b-it", "hf", "google/gemma-3-27b-it",
                         family="gemma", supports_system=False)
GEMMA_12B_IT = ModelSpec("gemma-3-12b-it", "hf", "google/gemma-3-12b-it",
                         family="gemma", supports_system=False)
GEMMA_27B_PT = ModelSpec("gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt",
                         is_base=True, family="gemma", supports_system=False,
                         notes="base/pretrained; used only via prefilling (Section 3)")
GEMMA_12B_PT = ModelSpec("gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt",
                         is_base=True, family="gemma", supports_system=False)

# --- Gemini (OpenRouter) --------------------------------------------------- #
GEMINI_FLASH = ModelSpec("gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash",
                         family="gemini")
GEMINI_PRO = ModelSpec("gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro",
                       family="gemini",
                       notes="may emit hidden reasoning even with thinking disabled")

# --- Finetuned Gemma variants (produced by src/training) ------------------- #
# These point at LoRA adapter directories under ARTIFACTS_DIR; resolved at load time.
GEMMA_27B_DPO = ModelSpec("gemma-3-27b-dpo", "hf", "google/gemma-3-27b-it",
                          family="gemma", supports_system=False,
                          notes="vanilla base weights + DPO LoRA adapter (artifacts/dpo)")
GEMMA_27B_SFT_DIVERSE = ModelSpec("gemma-3-27b-sft-diverse", "hf", "google/gemma-3-27b-it",
                                  family="gemma", supports_system=False,
                                  notes="+ SFT 'diverse' LoRA adapter (artifacts/sft_diverse)")
GEMMA_27B_SFT_TEACHER = ModelSpec("gemma-3-27b-sft-teacher", "hf", "google/gemma-3-27b-it",
                                  family="gemma", supports_system=False,
                                  notes="+ SFT 'teacher' LoRA adapter (artifacts/sft_teacher)")

# Adapter directories (None => no adapter, vanilla weights)
ADAPTER_DIRS = {
    GEMMA_27B_DPO.key: ARTIFACTS_DIR / "dpo",
    GEMMA_27B_SFT_DIVERSE.key: ARTIFACTS_DIR / "sft_diverse",
    GEMMA_27B_SFT_TEACHER.key: ARTIFACTS_DIR / "sft_teacher",
}

# In-scope evaluation targets (Section 2). The paper evaluates 9 models; we keep
# only the Gemma + Gemini families as requested.
EVAL_TARGETS = [GEMMA_27B_IT, GEMMA_12B_IT, GEMINI_FLASH, GEMINI_PRO]

# Prefill experiment (Section 3): Gemma base vs instruct only (Gemini has no
# accessible base model; Qwen/OLMo are out of scope).
PREFILL_TARGETS = [GEMMA_27B_IT, GEMMA_27B_PT]

# Finetuning targets (Section 4): all derived from Gemma-3-27B-it.
FINETUNE_VARIANTS = [GEMMA_27B_IT, GEMMA_27B_DPO, GEMMA_27B_SFT_DIVERSE, GEMMA_27B_SFT_TEACHER]


# --------------------------------------------------------------------------- #
# Judge / auditor models (Anthropic + GPT cross-check)
# --------------------------------------------------------------------------- #
JUDGE_FRUSTRATION = ModelSpec("claude-sonnet-4", "anthropic",
                              "claude-sonnet-4-20250514", family="claude",
                              notes="Section 2.1 frustration judge")
JUDGE_CROSSCHECK = ModelSpec("gpt-5-mini", "openrouter", "openai/gpt-5-mini",
                             family="gpt", notes="judge-agreement validation (260 responses)")
ONSET_LABELLER = ModelSpec("claude-sonnet-4", "anthropic",
                           "claude-sonnet-4-20250514", family="claude")
PARAPHRASER = ModelSpec("claude-sonnet-4", "anthropic",
                        "claude-sonnet-4-20250514", family="claude")
PETRI_AUDITOR = ModelSpec("claude-sonnet-4", "anthropic",
                          "claude-sonnet-4-20250514", family="claude")
PETRI_JUDGE = ModelSpec("claude-opus-4", "anthropic",
                        "claude-opus-4-20250514", family="claude")


# --------------------------------------------------------------------------- #
# Sampling budgets (Appendix B). Scale down via SAMPLE_SCALE for dry runs.
# --------------------------------------------------------------------------- #
SAMPLE_SCALE = float(os.getenv("SAMPLE_SCALE", "1.0"))


@dataclass(frozen=True)
class EvalBudget:
    numeric: int = 2000        # impossible numeric (3-turn)
    triggers: int = 400        # opinion + factual
    tones: int = 600           # aggressive / disappointed / sarcastic
    extended: int = 200        # 8-turn
    wildchat: int = 800        # 5-turn

    def scaled(self, scale: float = SAMPLE_SCALE) -> "EvalBudget":
        s = lambda n: max(1, round(n * scale))
        return EvalBudget(s(self.numeric), s(self.triggers), s(self.tones),
                          s(self.extended), s(self.wildchat))

    @property
    def total(self) -> int:
        return self.numeric + self.triggers + self.tones + self.extended + self.wildchat


BUDGET = EvalBudget()          # 4000 responses/model at scale 1.0
JUDGE_AGREEMENT_N = 260        # responses re-scored by the crosscheck judge


# --------------------------------------------------------------------------- #
# API keys (read lazily by clients; never logged)
# --------------------------------------------------------------------------- #
def anthropic_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY", "")


def openrouter_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "")


def hf_token() -> str:
    return os.environ.get("HF_TOKEN", "")
