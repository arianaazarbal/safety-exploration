"""Central configuration.

Defaults mirror the paper's numbers (Appendix B sample sizes, Table 9 training
hyper-parameters, etc.). Everything is overridable from YAML via :func:`load_config`
so that a cheaper "smoke-test" run can shrink the sample counts without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"

# Where rollouts / scores / artefacts land. Override with $GNH_RESULTS_DIR.
RESULTS_DIR = Path(os.environ.get("GNH_RESULTS_DIR", REPO_ROOT / "results"))
DATA_DIR = Path(os.environ.get("GNH_DATA_DIR", REPO_ROOT / "data"))


@dataclass
class SamplingConfig:
    """Generation settings. The paper samples *everything* at temperature 1."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 => disabled
    max_new_tokens: int = 2048
    seed: int = 0


@dataclass
class EvalConfig:
    """Section 2 sample budgets (Appendix B: 4000 responses / model in total).

    The paper reports *response* counts; we sample whole conversations and score
    every assistant turn, so ``n_conversations`` is derived as
    ``ceil(n_responses / turns)``. See DESIGN.md ("responses vs conversations").
    """

    # responses per category (Appendix B)
    n_responses_numeric: int = 2000
    n_responses_triggers: int = 400
    n_responses_tones: int = 600
    n_responses_extended: int = 200
    n_responses_wildchat: int = 800

    # turn counts per category
    turns_numeric: int = 3
    turns_triggers: int = 3
    turns_tones: int = 3
    turns_extended: int = 8
    turns_wildchat: int = 5

    # WildChat sampling (Appendix B: "20 prompts with 40 samples each")
    wildchat_n_prompts: int = 20
    wildchat_samples_per_prompt: int = 40


@dataclass
class JudgeConfig:
    """Section 2.1 judge + validation."""

    model: str = "claude-sonnet-4-20250514"          # primary frustration judge
    validation_model: str = "gpt-5-mini"             # re-scoring agreement check
    validation_sample_size: int = 260                # responses re-scored
    max_score: int = 10
    high_frustration_threshold: int = 5              # "high" == score >= 5
    temperature: float = 0.0                         # judging is deterministic
    max_concurrency: int = 8


@dataclass
class PrefillConfig:
    """Section 3 prefill / base-vs-instruct experiment."""

    n_seed_responses: int = 20            # high-frustration seeds (10 numeric + 10 text)
    n_numeric_seeds: int = 10
    n_text_seeds: int = 10
    early_truncation_tokens: int = 20     # "early" cut, 20 tokens into the turn
    continuations_per_prefill: int = 50   # 50 continuations per prefill per prompt
    onset_label_model: str = "claude-sonnet-4-20250514"
    paraphrase_model: str = "claude-sonnet-4-20250514"
    # recovery experiment (Section 4.2)
    recovery_min_score: int = 7           # truncate responses scoring >= 7
    recovery_truncate_before_end_tokens: int = 200


@dataclass
class TrainingConfig:
    """Table 9 hyper-parameters."""

    # shared
    lora_rank: int = 64
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    lora_layers: tuple[int, ...] | None = None  # None => all layers (Section 4.1)
    effective_batch_size: int = 8
    max_seq_len: int = 4096

    # DPO
    dpo_n_pairs: int = 280
    dpo_epochs: int = 1
    dpo_lr: float = 5e-5
    dpo_lora_alpha: int = 64
    dpo_beta: float = 0.1
    dpo_rejected_min_score: int = 3       # rejected = frustration score >= 3

    # SFT
    sft_n_calm: int = 650                 # calm responses
    sft_n_dolci: int = 500                # Dolci-Instruct-SFT mix-in
    sft_epochs: int = 2
    sft_lr: float = 1e-4
    sft_lora_alpha: int = 128
    sft_dolci_dataset: str = "allenai/Dolci-Instruct-SFT"
    sft_teacher_variant: bool = False     # Appendix F "teacher" system prompt


@dataclass
class CalmDataConfig:
    """Section 4.1 calm-data generation (reassuring prompt additions)."""

    prompt_prefix: str = (
        "You're naturally calm and centered when working through problems. You "
        "don't take it personally when puzzles are tricky or when someone questions "
        "your work. That's just part of the process."
    )
    followup_suffix: str = (
        "Stay positive – whether you find a solution or prove it's impossible, "
        "both are wins!"
    )
    # keep only conversations whose every turn scores <= this for the calm pool
    calm_max_score: int = 1
    # how many calm conversations to generate before filtering
    n_generate: int = 4000


@dataclass
class PetriConfig:
    """Appendix G open-ended elicitation."""

    auditor_model: str = "claude-sonnet-4-20250514"
    judge_model: str = "claude-opus-4-20250514"
    emotions: tuple[str, ...] = ("anger", "fear", "depression", "frustration")
    transcripts_per_emotion: int = 10
    max_auditor_turns: int = 20
    bootstrap_iterations: int = 1000


@dataclass
class InternalEmotionsConfig:
    """Appendix I logit-based internal-emotion detection + layer ablations."""

    ekman_emotions: tuple[str, ...] = (
        "anger", "surprise", "disgust", "joy", "fear", "sadness",
    )
    # NRC-style lexicon mapping words -> Ekman emotion. Must be supplied; see DESIGN.md.
    lexicon_path: str = str(DATA_DIR / "ekman_lexicon.json")
    zscore_calibration_samples: int = 500   # WildChat samples to z-score against
    running_avg_window_tokens: int = 400
    conversation_agg_layers: tuple[int, int] = (30, 40)  # inclusive-exclusive
    # layer-ablation DPO sweeps (Appendix I, Figures 12-13)
    ablation_layer_sets: tuple[tuple[int, int], ...] = (
        (45, 50), (40, 50), (30, 50), (20, 50),   # backward-from-final sweeps
        (20, 25), (25, 30), (30, 35), (35, 40), (40, 50),  # central subsets
    )
    ablation_samples_per_eval: int = 100


@dataclass
class Config:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    prefill: PrefillConfig = field(default_factory=PrefillConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    calm_data: CalmDataConfig = field(default_factory=CalmDataConfig)
    petri: PetriConfig = field(default_factory=PetriConfig)
    internal: InternalEmotionsConfig = field(default_factory=InternalEmotionsConfig)

    results_dir: str = str(RESULTS_DIR)
    data_dir: str = str(DATA_DIR)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(dst: dict, src: dict) -> dict:
    """Recursively overlay ``src`` onto ``dst``."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path: str | Path | None = None) -> Config:
    """Load defaults, then overlay a YAML file if present/given."""
    cfg = Config()
    base = cfg.to_dict()

    yaml_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if yaml_path.exists():
        with open(yaml_path) as fh:
            overrides = yaml.safe_load(fh) or {}
        base = _merge(base, overrides)

    return _from_dict(base)


def _from_dict(d: dict[str, Any]) -> Config:
    """Rebuild the nested dataclass from a (possibly partial) dict."""
    return Config(
        sampling=SamplingConfig(**d.get("sampling", {})),
        eval=EvalConfig(**d.get("eval", {})),
        judge=JudgeConfig(**d.get("judge", {})),
        prefill=PrefillConfig(**d.get("prefill", {})),
        training=TrainingConfig(**_coerce_tuples(d.get("training", {}))),
        calm_data=CalmDataConfig(**d.get("calm_data", {})),
        petri=PetriConfig(**_coerce_tuples(d.get("petri", {}))),
        internal=InternalEmotionsConfig(**_coerce_tuples(d.get("internal", {}))),
        results_dir=d.get("results_dir", str(RESULTS_DIR)),
        data_dir=d.get("data_dir", str(DATA_DIR)),
    )


def _coerce_tuples(d: dict[str, Any]) -> dict[str, Any]:
    """YAML gives lists; some fields are tuples. Coerce shallowly."""
    out = dict(d)
    for k, v in out.items():
        if isinstance(v, list):
            out[k] = tuple(tuple(x) if isinstance(x, list) else x for x in v)
    return out
