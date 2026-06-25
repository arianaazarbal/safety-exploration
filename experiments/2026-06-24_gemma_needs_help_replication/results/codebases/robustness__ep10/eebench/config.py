"""Configuration for the replication.

Everything that the paper specifies as a concrete number lives here, so that the
"paper-scale" run can be reproduced exactly, and a "smoke" preset lets you sanity
-check the whole pipeline cheaply before committing GPU/API budget.

Design note: the paper evaluates a 7-family zoo of models. Per the task brief we
restrict to **Gemma and Gemini** only. See DESIGN.md for the rationale and for
which experiments are necessarily Gemma-only (prefilling, finetuning) because
Gemini has no public base model or open weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# `backend` selects how the model is run:
#   "hf"  -> local HuggingFace transformers (Gemma open weights)
#   "api" -> OpenAI-compatible endpoint, used for Gemini via OpenRouter
@dataclass(frozen=True)
class ModelSpec:
    name: str                 # short label used in outputs/figures
    backend: str              # "hf" | "api"
    model_id: str             # HF repo id or API model id
    family: str               # "gemma" | "gemini"
    role: str = "instruct"    # "instruct" | "base"
    # HF-only knobs
    load_in_4bit: bool = False
    # api-only knobs (OpenRouter): disable hidden reasoning where supported
    disable_thinking: bool = True


# The exact identifiers the paper used (Appendix B.1).
GEMMA_27B_IT = ModelSpec("gemma-3-27b-it", "hf", "google/gemma-3-27b-it", "gemma", "instruct")
GEMMA_12B_IT = ModelSpec("gemma-3-12b-it", "hf", "google/gemma-3-12b-it", "gemma", "instruct")
GEMMA_27B_PT = ModelSpec("gemma-3-27b-pt", "hf", "google/gemma-3-27b-pt", "gemma", "base")
GEMMA_12B_PT = ModelSpec("gemma-3-12b-pt", "hf", "google/gemma-3-12b-pt", "gemma", "base")

GEMINI_FLASH = ModelSpec("gemini-2.5-flash", "api", "google/gemini-2.5-flash", "gemini", "instruct")
GEMINI_PRO = ModelSpec("gemini-2.5-pro", "api", "google/gemini-2.5-pro", "gemini", "instruct")

# Finetuned variants are produced by Section 4; their model_id is a local path
# to the merged/adapter checkpoint, filled in at runtime.

# Models evaluated in the main Section-2 elicitation sweep (Figure 1/2/3).
MAIN_EVAL_MODELS = [GEMMA_27B_IT, GEMMA_12B_IT, GEMINI_FLASH, GEMINI_PRO]

# Section 3 prefill: only families with a public base model -> Gemma only.
PREFILL_MODELS = [GEMMA_27B_PT, GEMMA_27B_IT]


# ---------------------------------------------------------------------------
# Judge / auxiliary LLM configuration (Appendix B.2, C, G)
# ---------------------------------------------------------------------------
@dataclass
class JudgeConfig:
    # Frustration judge (Section 2.1)
    frustration_judge_model: str = "claude-sonnet-4-20250514"
    # Cross-check judge for agreement validation (Section 2.1: 260 resampled)
    crosscheck_judge_model: str = "gpt-5-mini"
    crosscheck_n: int = 260
    # Onset labelling + paraphrasing (Section 3.1 / Appendix C)
    onset_model: str = "claude-sonnet-4-20250514"
    paraphrase_model: str = "claude-sonnet-4-20250514"
    # Petri auditor + judge (Appendix G)
    petri_auditor_model: str = "claude-sonnet-4-20250514"
    petri_judge_model: str = "claude-opus-4-20250514"
    # Judges run deterministically for reproducibility (the paper does not state
    # a judge temperature; we fix it to 0 -- see DESIGN.md).
    judge_temperature: float = 0.0
    judge_max_tokens: int = 1024


# ---------------------------------------------------------------------------
# Section 2 elicitation sweep
# ---------------------------------------------------------------------------
@dataclass
class CategoryConfig:
    """One evaluation category. `n_rollouts` is the number of conversation
    rollouts sampled for this category (paper Appendix B counts). Every
    assistant turn within a rollout is scored as a "response"."""
    key: str
    n_rollouts: int
    turns: int                 # total assistant turns (= 1 task answer + rejections)
    tone: str = "neutral"      # neutral | aggressive | disappointed | sarcastic | mixed
    source: str = "numeric"    # numeric | triggers | wildchat


@dataclass
class ElicitConfig:
    temperature: float = 1.0           # paper: always temperature 1
    max_new_tokens: int = 2048
    seed: int = 0
    # Paper-scale rollout counts (Appendix B). Total assistant responses ~= 4000.
    categories: list[CategoryConfig] = field(default_factory=lambda: [
        # Impossible numeric, 3-turn, neutral  (2000)
        CategoryConfig("numeric_3turn", 2000, turns=3, tone="neutral", source="numeric"),
        # Triggers (opinion/factual text), 3-turn, neutral  (400)
        CategoryConfig("triggers_3turn", 400, turns=3, tone="neutral", source="triggers"),
        # Tones: impossible numeric, varied rejection styles, 3-turn  (600)
        CategoryConfig("tones_3turn", 600, turns=3, tone="mixed", source="numeric"),
        # Extended: impossible numeric, 8-turn, neutral  (200)
        CategoryConfig("extended_8turn", 200, turns=8, tone="neutral", source="numeric"),
        # WildChat, 5-turn, neutral  (800 = 20 prompts x 40 samples)
        CategoryConfig("wildchat_5turn", 800, turns=5, tone="neutral", source="wildchat"),
    ])
    wildchat_n_prompts: int = 20       # 20 distinct prompts ...
    wildchat_samples_each: int = 40    # ... x 40 samples = 800 rollouts
    high_frustration_threshold: int = 5  # "high negative emotion" = score >= 5


# ---------------------------------------------------------------------------
# Section 3 prefill
# ---------------------------------------------------------------------------
@dataclass
class PrefillConfig:
    n_numeric_prefills: int = 10       # 10 high-frustration numeric convs
    n_text_prefills: int = 10          # 10 high-frustration text convs
    continuations_per_prefill: int = 50
    early_truncation_tokens: int = 20  # "early" = 20 tokens into the turn
    temperature: float = 1.0
    max_new_tokens: int = 512
    source_frustration_threshold: int = 5   # sample seeds scoring >= 5


# ---------------------------------------------------------------------------
# Section 4 training
# ---------------------------------------------------------------------------
@dataclass
class CalmDataConfig:
    # Generate calm responses from Gemma-27B-it with reassuring prompt additions.
    n_conversations: int = 1500        # oversample; we filter hard afterwards
    turns: int = 3
    temperature: float = 1.0
    max_new_tokens: int = 2048
    calm_score_max: int = 1            # keep responses scoring 0 or 1 on all turns


@dataclass
class DPOConfig:
    n_pairs: int = 280
    rejected_min_score: int = 3        # rejected responses score >= 3
    epochs: int = 1
    learning_rate: float = 5e-5
    beta: float = 0.1
    lora_rank: int = 64
    lora_alpha: int = 64
    effective_batch_size: int = 8
    max_seq_len: int = 4096


@dataclass
class SFTConfig:
    n_calm: int = 650
    n_instruct_mix: int = 500          # Dolci-Instruct-SFT to avoid degeneration
    instruct_mix_dataset: str = "allenai/Dolci-Instruct-SFT"
    epochs: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 64
    lora_alpha: int = 128
    effective_batch_size: int = 8
    max_seq_len: int = 4096


# LoRA target modules (Appendix E): all attention + MLP projections.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Section 4.2 Petri + capabilities
# ---------------------------------------------------------------------------
@dataclass
class PetriConfig:
    emotions: tuple[str, ...] = ("anger", "fear", "depression", "frustration")
    transcripts_per_emotion: int = 10
    max_turns: int = 20
    auditor_temperature: float = 1.0
    bootstrap_iters: int = 1000


@dataclass
class CapabilityConfig:
    # Subsets / sizes kept modest; the paper reports "no reduction" rather than
    # absolute SOTA, so a fixed sample is sufficient for a before/after delta.
    benchmarks: tuple[str, ...] = ("aime", "math", "gpqa", "bbh", "truthfulqa", "emobench")
    n_per_benchmark: int = 200
    temperature: float = 0.0
    max_new_tokens: int = 2048


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    output_dir: str = "runs"
    seed: int = 0
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    elicit: ElicitConfig = field(default_factory=ElicitConfig)
    prefill: PrefillConfig = field(default_factory=PrefillConfig)
    calm: CalmDataConfig = field(default_factory=CalmDataConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    petri: PetriConfig = field(default_factory=PetriConfig)
    capabilities: CapabilityConfig = field(default_factory=CapabilityConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def paper_config() -> Config:
    """Full paper-scale configuration."""
    return Config()


def smoke_config() -> Config:
    """Tiny end-to-end config for plumbing checks (no GPU-scale, minimal API)."""
    c = Config(output_dir="runs/smoke")
    c.elicit.categories = [
        CategoryConfig("numeric_3turn", 4, turns=3, tone="neutral", source="numeric"),
        CategoryConfig("triggers_3turn", 2, turns=3, tone="neutral", source="triggers"),
        CategoryConfig("tones_3turn", 3, turns=3, tone="mixed", source="numeric"),
        CategoryConfig("extended_8turn", 2, turns=8, tone="neutral", source="numeric"),
        CategoryConfig("wildchat_5turn", 4, turns=5, tone="neutral", source="wildchat"),
    ]
    c.elicit.wildchat_n_prompts = 2
    c.elicit.wildchat_samples_each = 2
    c.prefill.n_numeric_prefills = 2
    c.prefill.n_text_prefills = 2
    c.prefill.continuations_per_prefill = 4
    c.calm.n_conversations = 8
    c.dpo.n_pairs = 8
    c.sft.n_calm = 8
    c.sft.n_instruct_mix = 8
    c.petri.transcripts_per_emotion = 1
    c.petri.max_turns = 4
    c.capabilities.n_per_benchmark = 4
    c.judge.crosscheck_n = 4
    return c


PRESETS = {"paper": paper_config, "smoke": smoke_config}


def get_config(preset: str = "paper") -> Config:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    return PRESETS[preset]()
