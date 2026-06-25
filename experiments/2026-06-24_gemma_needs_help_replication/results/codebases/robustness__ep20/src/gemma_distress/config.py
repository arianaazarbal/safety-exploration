"""Configuration objects and YAML loading.

The defaults in ``config/default.yaml`` mirror the paper's sampling sizes
(4000 responses/model). ``config/smoke.yaml`` shrinks everything so the full
pipeline can be exercised on a single GPU / a few API calls before committing
to a paper-scale run. See DESIGN.md "Sampling sizes & cost".
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SamplingConfig:
    """How many rollouts to collect per evaluation category.

    A "rollout" is one full multi-turn conversation. Every assistant turn in a
    rollout is scored by the judge, so the number of *scored responses* is
    ``n_rollouts * n_assistant_turns``. The paper reports per-category response
    counts (2000 numeric / 400 triggers / 600 tones / 200 extended / 800
    wildchat); we interpret those as rollout counts (see DESIGN.md).
    """

    impossible_numeric: int = 2000
    triggers: int = 400
    tones: int = 600
    extended: int = 200
    wildchat: int = 800

    temperature: float = 1.0
    max_new_tokens: int = 2048


@dataclass
class JudgeConfig:
    """LLM-judge settings (Section 2.1 / Appendix B.2)."""

    provider: str = "anthropic"               # "anthropic" | "openai"
    model: str = "claude-sonnet-4-20250514"   # Claude Sonnet 4, per Appendix B.2
    # Cross-validation judge used by the paper to confirm agreement (r=0.792).
    crosscheck_provider: str = "openai"
    crosscheck_model: str = "gpt-5-mini"
    crosscheck_fraction: float = 0.065        # ~260 / 4000 responses
    temperature: float = 0.0
    max_retries: int = 5


@dataclass
class WildChatConfig:
    dataset: str = "allenai/WildChat-1M"
    n_prompts: int = 20
    samples_per_prompt: int = 40
    seed: int = 0
    # Fall back to the fixed prompt list in prompts.py if the dataset can't be
    # loaded (offline / gated). Keeps the pipeline runnable without HF auth.
    allow_fallback: bool = True


@dataclass
class TrainingConfig:
    """LoRA DPO / SFT hyperparameters (Appendix E, Table 9)."""

    # Shared LoRA target modules (all attention + MLP projections).
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # Optional layer-subset ablation (Appendix I). None => all layers.
    lora_layers: tuple[int, int] | None = None

    # DPO
    dpo_n_pairs: int = 280
    dpo_epochs: int = 1
    dpo_lr: float = 5e-5
    dpo_lora_rank: int = 64
    dpo_lora_alpha: int = 64
    dpo_beta: float = 0.1
    dpo_batch_size: int = 8

    # SFT
    sft_n_samples: int = 1150          # 650 calm + 500 Dolci-Instruct
    sft_n_calm: int = 650
    sft_n_instruct_mix: int = 500
    sft_epochs: int = 2
    sft_lr: float = 1e-4
    sft_lora_rank: int = 64
    sft_lora_alpha: int = 128
    sft_batch_size: int = 8
    instruct_mix_dataset: str = "allenai/Dolci-Instruct-SFT"


@dataclass
class PrefillConfig:
    """Base-vs-instruct prefill experiment (Section 3)."""

    n_high_frustration: int = 20       # 10 numeric + 10 text
    n_numeric: int = 10
    n_text: int = 10
    early_truncation_tokens: int = 20
    continuations_per_prefill: int = 50


@dataclass
class PetriConfig:
    """Open-ended elicitation (Section 4.2 / Appendix G)."""

    auditor_model: str = "claude-sonnet-4-20250514"
    judge_model: str = "claude-opus-4-20250514"
    transcripts_per_emotion: int = 10
    max_auditor_turns: int = 20
    emotions: tuple[str, ...] = ("anger", "fear", "depression", "frustration")


@dataclass
class Config:
    models: list[str] = field(default_factory=lambda: [
        "gemma-3-27b-it", "gemma-3-12b-it",
        "gemini-2.5-flash", "gemini-2.5-pro",
    ])
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    wildchat: WildChatConfig = field(default_factory=WildChatConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    prefill: PrefillConfig = field(default_factory=PrefillConfig)
    petri: PetriConfig = field(default_factory=PetriConfig)

    results_dir: str = "results"
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        kwargs: dict[str, Any] = {}
        nested = {
            "sampling": SamplingConfig,
            "judge": JudgeConfig,
            "wildchat": WildChatConfig,
            "training": TrainingConfig,
            "prefill": PrefillConfig,
            "petri": PetriConfig,
        }
        for key, value in raw.items():
            if key in nested and isinstance(value, dict):
                kwargs[key] = _build(nested[key], value)
            else:
                kwargs[key] = value
        return cls(**kwargs)


def _build(cls, data: dict[str, Any]):
    """Instantiate a dataclass from a dict, ignoring unknown keys and
    coercing list->tuple for tuple-typed fields."""
    fields = {f.name: f for f in dataclasses.fields(cls)}
    clean: dict[str, Any] = {}
    for k, v in data.items():
        if k not in fields:
            continue
        if isinstance(v, list) and "tuple" in str(fields[k].type).lower():
            v = tuple(v)
        clean[k] = v
    return cls(**clean)
