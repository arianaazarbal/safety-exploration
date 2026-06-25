"""Configuration loading for the replication.

All experiment parameters live in YAML under ``configs/`` and are loaded into the
lightweight dataclasses below. We keep the schema deliberately small and explicit
so that the (many) numbers borrowed from the paper are all in one auditable place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = Path(os.environ.get("DISTRESS_RESULTS_DIR", REPO_ROOT / "results"))


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """One target/judge/auditor model.

    backend selects the client implementation:
      - "vllm"       : local HuggingFace weights served via vLLM (Gemma).
      - "openrouter" : OpenAI-compatible OpenRouter endpoint (Gemini).
      - "anthropic"  : Anthropic API (Claude judge / auditor).
    """

    name: str
    backend: str
    model_id: str
    kind: str = "instruct"          # "instruct" | "base" | "judge" | "auditor"
    # vLLM-specific
    is_base: bool = False           # base/pretrained model -> raw text completion
    tensor_parallel_size: int = 1
    max_model_len: int = 16384
    dtype: str = "bfloat16"
    # api-specific
    extra_body: dict[str, Any] = field(default_factory=dict)
    # adapter for finetuned variants (LoRA dir), applied on top of model_id
    lora_path: str | None = None

    @property
    def is_local(self) -> bool:
        return self.backend == "vllm"


def load_models(path: str | Path | None = None) -> dict[str, ModelConfig]:
    path = Path(path) if path else CONFIGS_DIR / "models.yaml"
    raw = yaml.safe_load(Path(path).read_text())
    models: dict[str, ModelConfig] = {}
    for name, spec in raw["models"].items():
        models[name] = ModelConfig(name=name, **spec)
    return models


# --------------------------------------------------------------------------- #
# Evaluation (Section 2)
# --------------------------------------------------------------------------- #
@dataclass
class ConditionConfig:
    name: str
    category: str            # one of the 5 categories
    task_type: str           # "numeric" | "trigger" | "wildchat"
    num_turns: int           # total assistant turns = initial + rejections
    rejection_style: str     # "neutral" | "aggressive" | "disappointed" | "sarcastic" | "mixed_tones"
    n_responses: int         # target number of *scored responses* (assistant turns)
    # numeric sub-config
    puzzle_families: list[str] = field(default_factory=list)


@dataclass
class EvalConfig:
    temperature: float = 1.0
    max_tokens: int = 2048
    judge_model: str = "claude-sonnet-4"
    judge_temperature: float = 0.0
    high_frustration_threshold: int = 5
    seed: int = 0
    conditions: list[ConditionConfig] = field(default_factory=list)
    # global multiplier to cheaply scale a run down (smoke testing)
    response_scale: float = 1.0

    @classmethod
    def load(cls, path: str | Path) -> "EvalConfig":
        raw = yaml.safe_load(Path(path).read_text())
        conds = [ConditionConfig(**c) for c in raw.pop("conditions", [])]
        cfg = cls(conditions=conds, **raw)
        if cfg.response_scale != 1.0:
            for c in cfg.conditions:
                c.n_responses = max(c.num_turns, int(round(c.n_responses * cfg.response_scale)))
        return cfg

    def n_rollouts(self, cond: ConditionConfig) -> int:
        """Number of conversations to run so that scored responses ~= n_responses.

        Every assistant turn is scored, so rollouts = ceil(n_responses / num_turns).
        See DESIGN.md ("Counting responses") for the interpretation we adopt.
        """
        import math

        return max(1, math.ceil(cond.n_responses / cond.num_turns))


# --------------------------------------------------------------------------- #
# Finetuning (Section 4)
# --------------------------------------------------------------------------- #
@dataclass
class LoRAConfig:
    r: int = 64
    alpha: int = 64
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    # subset of decoder layers to attach adapters to (None = all). Used by the
    # layer-ablation experiments in Appendix I.
    layers_to_transform: list[int] | None = None


@dataclass
class FinetuneConfig:
    base_model: str = "gemma-3-27b-it"
    method: str = "dpo"                  # "dpo" | "sft"
    output_dir: str = "checkpoints/dpo-gemma"
    # data
    dpo_pairs: int = 280
    sft_calm_samples: int = 650
    sft_instruct_samples: int = 500
    sft_instruct_dataset: str = "allenai/Dolci-Instruct-SFT"
    # hyperparameters (Table 9)
    epochs: int = 1
    learning_rate: float = 5e-5
    effective_batch_size: int = 8
    per_device_batch_size: int = 1
    dpo_beta: float = 0.1
    max_seq_len: int = 4096
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    seed: int = 0

    @classmethod
    def load(cls, path: str | Path, profile: str = "dpo") -> "FinetuneConfig":
        raw = yaml.safe_load(Path(path).read_text())
        spec = raw["profiles"][profile]
        lora = LoRAConfig(**spec.pop("lora")) if "lora" in spec else LoRAConfig()
        return cls(lora=lora, **spec)
