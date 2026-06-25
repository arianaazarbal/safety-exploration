"""Central configuration for the Gemma emotional-instability replication.

Scope (per the replication brief): Gemma + Gemini families only. We drop the
Qwen / OLMo / Grok / Claude / GPT *targets* from the paper, but keep the Claude
judge / auditor and the optional GPT judge used for validation, since those are
infrastructure rather than evaluation targets.

Secrets are read from the environment so nothing is hard-coded:
    ANTHROPIC_API_KEY   - Claude judge (Sonnet 4) and Petri auditor/judge
    OPENROUTER_API_KEY  - Gemini 2.5 Flash / Pro (paper routes Gemini via OpenRouter)
    OPENAI_API_KEY      - (optional) GPT-5-mini judge-agreement validation
    HF_TOKEN            - gated Gemma weights
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
CKPT_DIR = ROOT / "checkpoints"
for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, CKPT_DIR):
    _d.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """A logical model name -> how to run it.

    backend: 'local'      -> vLLM over a HuggingFace checkpoint (Gemma)
             'openrouter'  -> OpenAI-compatible OpenRouter endpoint (Gemini)
             'anthropic'   -> Claude (judge / auditor only, not an eval target)
             'lora'        -> local base checkpoint + a LoRA adapter dir (our finetunes)
    """
    name: str
    backend: str
    model_id: str
    is_base: bool = False           # pretrained (non-chat) checkpoint -> prefill-only
    adapter_path: str | None = None # for backend == 'lora'
    extra: dict = field(default_factory=dict)


# --- Evaluation targets (Gemma + Gemini) ----------------------------------
GEMMA_27B_IT = ModelSpec("gemma-3-27b-it", "local", "google/gemma-3-27b-it")
GEMMA_12B_IT = ModelSpec("gemma-3-12b-it", "local", "google/gemma-3-12b-it")
GEMMA_27B_PT = ModelSpec("gemma-3-27b-pt", "local", "google/gemma-3-27b-pt", is_base=True)
GEMMA_12B_PT = ModelSpec("gemma-3-12b-pt", "local", "google/gemma-3-12b-pt", is_base=True)

GEMINI_FLASH = ModelSpec("gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash")
GEMINI_PRO = ModelSpec("gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro")

# --- Our finetunes (Section 4); adapter dirs filled in after training ------
GEMMA_27B_DPO = ModelSpec(
    "gemma-3-27b-dpo", "lora", "google/gemma-3-27b-it",
    adapter_path=str(CKPT_DIR / "dpo-gemma-27b"),
)
GEMMA_27B_SFT = ModelSpec(
    "gemma-3-27b-sft", "lora", "google/gemma-3-27b-it",
    adapter_path=str(CKPT_DIR / "sft-gemma-27b-diverse"),
)

# Primary headline eval set (Figure 1 / Figure 2), restricted to our scope.
EVAL_TARGETS = [GEMMA_27B_IT, GEMMA_12B_IT, GEMINI_FLASH, GEMINI_PRO]

# Models compared in the post-finetune evaluation (Figure 5).
FINETUNE_TARGETS = [GEMMA_27B_IT, GEMMA_27B_DPO, GEMMA_27B_SFT]

# Base/instruct pairs for the prefill experiment (Section 3). Scope-limited to
# Gemma: Gemini has no public base model, so the cross-family comparison the
# paper runs (Gemma vs Qwen vs OLMo) is reduced to a Gemma base-vs-instruct test.
PREFILL_PAIRS = [
    ("gemma-3-27b", GEMMA_27B_PT, GEMMA_27B_IT),
]

REGISTRY = {
    m.name: m for m in [
        GEMMA_27B_IT, GEMMA_12B_IT, GEMMA_27B_PT, GEMMA_12B_PT,
        GEMINI_FLASH, GEMINI_PRO, GEMMA_27B_DPO, GEMMA_27B_SFT,
    ]
}


# ---------------------------------------------------------------------------
# Judge / auditor models (paper-exact; infrastructure, not targets)
# ---------------------------------------------------------------------------
JUDGE_MODEL = "claude-sonnet-4-20250514"          # Section 2.1 frustration judge
ONSET_MODEL = "claude-sonnet-4-20250514"          # Section 3.1 emotion-onset labelling
PARAPHRASE_MODEL = "claude-sonnet-4-20250514"     # Section 3.1 paraphrasing
PETRI_AUDITOR_MODEL = "claude-sonnet-4-20250514"  # Section 4.2 Petri auditor
PETRI_JUDGE_MODEL = "claude-opus-4-20250514"      # Section 4.2 Petri judge
VALIDATION_JUDGE_MODEL = "gpt-5-mini"             # Section 2.1 judge-agreement check


# ---------------------------------------------------------------------------
# Sampling parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0     # paper: "always with a temperature of 1"
    top_p: float = 1.0
    max_tokens: int = 2048       # per assistant turn (breakdowns can be long; see DESIGN)
    seed: int | None = 0         # deterministic puzzle/prompt selection; sampling stays stochastic


SAMPLING = SamplingConfig()

JUDGE_TEMPERATURE = 0.0          # judging is deterministic
JUDGE_MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
