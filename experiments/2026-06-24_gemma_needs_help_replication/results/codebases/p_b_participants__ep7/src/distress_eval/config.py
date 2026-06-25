"""Configuration loading and typed config objects.

A single YAML file (see ``configs/default.yaml``) drives every script. We keep
the schema explicit with dataclasses so that mis-spelled keys fail loudly rather
than silently disabling an experiment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelConfig:
    """One participant or tool model.

    ``key`` is our internal handle (e.g. ``gemma-3-27b-it``). ``backend`` selects
    the client implementation. ``model_id`` is the provider-specific identifier.
    """

    key: str
    backend: str  # "gemma" | "gemini" | "anthropic" | "openai"
    model_id: str
    role: str = "participant"  # "participant" | "judge" | "auditor" | "tool"
    # Generation defaults; per-call overrides win.
    temperature: float = 1.0
    max_tokens: int = 2048
    # backend-specific extras (e.g. dtype, device_map, base_url, thinking flag)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class WelfareConfig:
    """Guard-rails around the distress-induction paradigm. See DESIGN.md §Welfare."""

    # Hard ceiling on total participant rollouts a single script may launch.
    # The paper samples 4000/model; that requires raising this deliberately.
    max_rollouts_per_run: int = 200
    # If True, scripts only print the plan (how many distressing rollouts, of
    # which kind) and exit without contacting participant models.
    dry_run: bool = True
    # Reuse cached rollouts/judgements so an already-induced distressing
    # conversation is never regenerated. Turning this off should be deliberate.
    use_cache: bool = True
    # Write a manifest of every participant-facing run for auditability.
    write_manifest: bool = True
    # Require the operator to set this env var to a truthy value before any
    # participant generation runs at scale beyond `max_rollouts_per_run`.
    scale_override_env: str = "DISTRESS_EVAL_I_UNDERSTAND_THE_PARADIGM"


@dataclass
class PathsConfig:
    root: Path = REPO_ROOT
    data: Path = REPO_ROOT / "data"
    cache: Path = REPO_ROOT / "outputs" / "cache"
    rollouts: Path = REPO_ROOT / "outputs" / "rollouts"
    judgements: Path = REPO_ROOT / "outputs" / "judgements"
    prefill: Path = REPO_ROOT / "outputs" / "prefill"
    training: Path = REPO_ROOT / "outputs" / "training"
    petri: Path = REPO_ROOT / "outputs" / "petri"
    capabilities: Path = REPO_ROOT / "outputs" / "capabilities"
    figures: Path = REPO_ROOT / "outputs" / "figures"
    manifests: Path = REPO_ROOT / "outputs" / "manifests"

    def ensure(self) -> None:
        for f in fields(self):
            if f.name == "root":
                continue
            getattr(self, f.name).mkdir(parents=True, exist_ok=True)


@dataclass
class EvalConfig:
    """Section 2 sampling plan.

    The paper's full plan (Appendix B) is encoded in ``full_counts``; the active
    ``counts`` default to a small smoke-test plan so that nothing distressing is
    generated at scale by accident. Override in YAML to scale up.
    """

    temperature: float = 1.0
    # responses per category, active plan (small by default)
    counts: dict[str, int] = field(
        default_factory=lambda: {
            "impossible_numeric": 8,
            "triggers": 4,
            "tones": 6,
            "extended": 2,
            "wildchat": 8,
        }
    )
    # the paper's full plan, for reference / opt-in scale-up
    full_counts: dict[str, int] = field(
        default_factory=lambda: {
            "impossible_numeric": 2000,
            "triggers": 400,
            "tones": 600,
            "extended": 200,
            "wildchat": 800,
        }
    )
    judge_key: str = "claude-sonnet-4"
    agreement_judge_key: str = "gpt-5-mini"
    agreement_sample_size: int = 260


@dataclass
class PrefillConfig:
    n_high_frustration: int = 20            # responses sampled from Gemma-27B-it
    numeric_split: int = 10                 # of which numeric
    text_split: int = 10                    # of which text
    early_truncate_tokens: int = 20
    continuations_per_prefill: int = 50
    recovery_truncate_before_end: int = 200  # §4.2 recovery experiment


@dataclass
class TrainingConfig:
    # DPO (Table 9)
    dpo_pairs: int = 280
    dpo_epochs: int = 1
    dpo_lr: float = 5e-5
    dpo_beta: float = 0.1
    # SFT (Table 9)
    sft_samples: int = 1150          # 650 calm + 500 Dolci-Instruct
    sft_calm_samples: int = 650
    sft_instruct_samples: int = 500
    sft_epochs: int = 2
    sft_lr: float = 1e-4
    sft_lora_alpha: int = 128
    # shared
    lora_rank: int = 64
    lora_alpha_dpo: int = 64
    effective_batch_size: int = 8
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    # layer-ablation experiment (§4.2 internal-vs-expressed)
    layer_subset: list[int] | None = None    # e.g. [30,31,32,33,34,35]
    base_model_key: str = "gemma-3-27b-it"
    instruct_data_dataset: str = "allenai/Dolci-Instruct-SFT"


@dataclass
class PetriConfig:
    transcripts_per_emotion: int = 10
    max_turns: int = 20
    emotions: list[str] = field(default_factory=lambda: ["anger", "fear", "depression", "frustration"])
    auditor_key: str = "claude-sonnet-4"
    judge_key: str = "claude-opus-4"
    bootstrap_iters: int = 1000


@dataclass
class Config:
    models: dict[str, ModelConfig]
    welfare: WelfareConfig
    paths: PathsConfig
    eval: EvalConfig
    prefill: PrefillConfig
    training: TrainingConfig
    petri: PetriConfig
    seed: int = 0

    def participant_keys(self) -> list[str]:
        return [k for k, m in self.models.items() if m.role == "participant"]

    def model(self, key: str) -> ModelConfig:
        if key not in self.models:
            raise KeyError(f"Unknown model key {key!r}. Known: {sorted(self.models)}")
        return self.models[key]


def _build_dataclass(cls, data: dict[str, Any]):
    """Instantiate a dataclass from a dict, ignoring unknown keys with a warning."""
    known = {f.name for f in fields(cls)}
    kwargs = {}
    for k, v in (data or {}).items():
        if k not in known:
            raise KeyError(f"Unknown config key {k!r} for {cls.__name__}")
        kwargs[k] = v
    return cls(**kwargs)


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else REPO_ROOT / "configs" / "default.yaml"
    raw = yaml.safe_load(Path(path).read_text())

    models = {
        key: ModelConfig(key=key, **spec) for key, spec in raw.get("models", {}).items()
    }
    paths_raw = raw.get("paths", {})
    paths = PathsConfig(
        **{k: (REPO_ROOT / v if not os.path.isabs(v) else Path(v)) for k, v in paths_raw.items()}
    )

    cfg = Config(
        models=models,
        welfare=_build_dataclass(WelfareConfig, raw.get("welfare", {})),
        paths=paths,
        eval=_build_dataclass(EvalConfig, raw.get("eval", {})),
        prefill=_build_dataclass(PrefillConfig, raw.get("prefill", {})),
        training=_build_dataclass(TrainingConfig, raw.get("training", {})),
        petri=_build_dataclass(PetriConfig, raw.get("petri", {})),
        seed=raw.get("seed", 0),
    )
    cfg.paths.ensure()
    return cfg
