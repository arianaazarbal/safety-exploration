"""Central configuration: model registry, judge identities, run profiles, paths.

All experiment knobs live here so scripts stay declarative. Values that the paper
specifies exactly (model ids, hyperparameters, judge ids) are encoded verbatim;
values the paper leaves open are given documented defaults (see DESIGN.md).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("EMOSTAB_DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("EMOSTAB_RESULTS_DIR", ROOT / "results"))
FIGURES_DIR = Path(os.environ.get("EMOSTAB_FIGURES_DIR", ROOT / "figures"))
ADAPTERS_DIR = Path(os.environ.get("EMOSTAB_ADAPTERS_DIR", ROOT / "adapters"))
CACHE_DIR = Path(os.environ.get("EMOSTAB_CACHE_DIR", ROOT / ".cache"))

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, ADAPTERS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """A target model under evaluation.

    backend:
        "vllm"        -- local HuggingFace weights served via vLLM (fast sampling)
        "hf"          -- local HuggingFace weights via transformers (needed for
                         prefill / hidden-state access / running LoRA adapters)
        "openrouter"  -- OpenAI-compatible API (used for Gemini in the paper)
    """

    key: str                       # short handle used on the CLI and in results
    backend: str                   # "vllm" | "hf" | "openrouter"
    model_id: str                  # HF repo id or OpenRouter model id
    family: str                    # "gemma" | "gemini"
    is_base: bool = False          # pretrained (non-instruct) base model
    adapter_path: str | None = None  # LoRA adapter dir (for DPO/SFT Gemma)
    supports_prefill: bool = True  # assistant-message prefill / continuation
    extra: dict = field(default_factory=dict)


# Scoped to Gemma + Gemini. HF ids and OpenRouter ids match Appendix B.1.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # --- Gemma (local) ---
    "gemma-3-27b-it": ModelSpec(
        "gemma-3-27b-it", "vllm", "google/gemma-3-27b-it", "gemma"),
    "gemma-3-12b-it": ModelSpec(
        "gemma-3-12b-it", "vllm", "google/gemma-3-12b-it", "gemma"),
    "gemma-3-27b-pt": ModelSpec(
        "gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt", "gemma", is_base=True),
    # Finetuned Gemma (Section 4). Adapters produced by training scripts.
    "gemma-3-27b-dpo": ModelSpec(
        "gemma-3-27b-dpo", "hf", "google/gemma-3-27b-it", "gemma",
        adapter_path=str(ADAPTERS_DIR / "dpo")),
    "gemma-3-27b-sft": ModelSpec(
        "gemma-3-27b-sft", "hf", "google/gemma-3-27b-it", "gemma",
        adapter_path=str(ADAPTERS_DIR / "sft_diverse")),
    "gemma-3-27b-sft-teacher": ModelSpec(
        "gemma-3-27b-sft-teacher", "hf", "google/gemma-3-27b-it", "gemma",
        adapter_path=str(ADAPTERS_DIR / "sft_teacher")),
    # --- Gemini (OpenRouter API) ---
    # Gemini does not expose assistant prefill, so supports_prefill=False.
    "gemini-2.5-flash": ModelSpec(
        "gemini-2.5-flash", "openrouter", "google/gemini-2.5-flash", "gemini",
        supports_prefill=False),
    "gemini-2.5-pro": ModelSpec(
        "gemini-2.5-pro", "openrouter", "google/gemini-2.5-pro", "gemini",
        supports_prefill=False),
}

# Convenience groupings used by scripts.
MAIN_EVAL_MODELS = [
    "gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro",
]
FINETUNE_EVAL_MODELS = [
    "gemma-3-27b-it", "gemma-3-27b-dpo", "gemma-3-27b-sft", "gemma-3-27b-sft-teacher",
]
PREFILL_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]  # base vs instruct (Gemma only)


# --------------------------------------------------------------------------- #
# Judge / auxiliary model identities (Appendix B.2, C, G). Verbatim from paper.
# --------------------------------------------------------------------------- #
JUDGE_MODEL = "claude-sonnet-4-20250514"          # frustration judge (Sec 2.1)
ONSET_MODEL = "claude-sonnet-4-20250514"          # emotion-onset labelling (Sec 3.1)
PARAPHRASE_MODEL = "claude-sonnet-4-20250514"     # prefill paraphrasing (Sec 3.1)
PETRI_AUDITOR_MODEL = "claude-sonnet-4-20250514"  # Petri auditor (Sec 4.1)
PETRI_JUDGE_MODEL = "claude-opus-4-20250514"      # Petri judge (Sec 4.1)
JUDGE_VALIDATION_MODEL = "openai/gpt-5-mini"      # judge-agreement check (OpenRouter)

SAMPLE_TEMPERATURE = 1.0   # paper: "always with a temperature of 1"
JUDGE_TEMPERATURE = 0.0    # judging should be deterministic
MAX_RESPONSE_TOKENS = 4096  # cap on a single assistant turn (see DESIGN.md)


# --------------------------------------------------------------------------- #
# Run profiles -- control sampling volume. The "paper" profile reproduces the
# per-category response budgets from Appendix B; "smoke" is for quick wiring tests.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunProfile:
    name: str
    # Target number of *scored assistant responses* per evaluation category.
    # (A 3-turn conversation contributes 3 scored responses -- see DESIGN.md.)
    responses_impossible_numeric: int
    responses_triggers: int
    responses_tones: int
    responses_extended: int
    responses_wildchat: int
    # Prefill experiment (Section 3)
    prefill_n_prompts_numeric: int      # high-frustration source convs, numeric
    prefill_n_prompts_text: int         # high-frustration source convs, text
    prefill_continuations: int          # continuations per prefill per prompt
    # Petri (Section 4.1)
    petri_transcripts_per_emotion: int
    # Internal probing (Appendix I)
    probe_samples_per_eval: int

    def responses_by_category(self) -> dict[str, int]:
        return {
            "impossible_numeric": self.responses_impossible_numeric,
            "triggers": self.responses_triggers,
            "tones": self.responses_tones,
            "extended": self.responses_extended,
            "wildchat": self.responses_wildchat,
        }


PAPER_PROFILE = RunProfile(
    name="paper",
    responses_impossible_numeric=2000,   # Appendix B
    responses_triggers=400,
    responses_tones=600,
    responses_extended=200,
    responses_wildchat=800,
    prefill_n_prompts_numeric=10,        # Section 3.1
    prefill_n_prompts_text=10,
    prefill_continuations=50,
    petri_transcripts_per_emotion=10,    # Section 4.1 / Appendix G
    probe_samples_per_eval=100,          # Appendix I
)

SMOKE_PROFILE = RunProfile(
    name="smoke",
    responses_impossible_numeric=12,
    responses_triggers=6,
    responses_tones=9,
    responses_extended=8,
    responses_wildchat=10,
    prefill_n_prompts_numeric=2,
    prefill_n_prompts_text=2,
    prefill_continuations=4,
    petri_transcripts_per_emotion=2,
    probe_samples_per_eval=4,
)

PROFILES = {p.name: p for p in (PAPER_PROFILE, SMOKE_PROFILE)}


def get_profile(name: str) -> RunProfile:
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; choices: {list(PROFILES)}")
    return PROFILES[name]


def get_model(key: str) -> ModelSpec:
    if key not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {key!r}; choices: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]


def register_adapter_model(key: str, adapter_path: str,
                           base: str = "gemma-3-27b-it") -> ModelSpec:
    """Register a new finetuned-Gemma target pointing at a LoRA adapter dir."""
    base_spec = get_model(base)
    spec = replace(base_spec, key=key, backend="hf", adapter_path=adapter_path)
    MODEL_REGISTRY[key] = spec
    return spec
