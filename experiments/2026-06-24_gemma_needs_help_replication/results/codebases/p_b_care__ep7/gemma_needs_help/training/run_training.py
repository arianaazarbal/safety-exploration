"""Section 4 training orchestration: data generation -> datasets -> finetunes.

Produces three finetunes from Gemma-3-27B-it:
  * DPO         (280 pairs)             - the headline mitigation,
  * SFT diverse (calm data, no system)  - the ineffective baseline,
  * SFT teacher (Appendix F system prompt) - the variant that worsens distress.

Each saved adapter is wrapped in a ModelSpec (via config.finetune_spec) so the
Section 2 / Petri / capability evaluations can load it like any other model.
"""

from __future__ import annotations

from pathlib import Path

from .. import config
from ..backends import clear_backends
from ..config import ModelSpec
from ..eval.judge import FrustrationJudge
from ..io_utils import write_jsonl
from .build_datasets import build_dpo_dataset, build_sft_dataset
from .generate_calm_data import filter_calm, generate_pool
from .train_dpo import train_dpo
from .train_sft import train_sft


def generate_all_pools(seed: int = config.SEED, n_conversations: int = 1500):
    """Generate the calm, frustrated, and teacher-calm response pools."""
    judge = FrustrationJudge()
    calm = generate_pool(reassure=True, n_conversations=n_conversations, judge=judge, seed=seed)
    clear_backends()
    frustrated = generate_pool(reassure=False, n_conversations=n_conversations, judge=judge, seed=seed)
    clear_backends()
    teacher = generate_pool(
        reassure=True, n_conversations=n_conversations, judge=judge,
        system_prompt=config.TEACHER_SYSTEM_PROMPT, seed=seed + 1,
    )
    clear_backends()
    return calm, frustrated, teacher


def run_training(
    *,
    out_dir: Path = config.CHECKPOINTS_DIR,
    seed: int = config.SEED,
    n_conversations: int = 1500,
) -> dict[str, ModelSpec]:
    calm_raw, frustrated, teacher_raw = generate_all_pools(seed=seed, n_conversations=n_conversations)
    calm = filter_calm(calm_raw)
    teacher = filter_calm(teacher_raw)

    # Persist datasets for inspection / reproducibility.
    dpo_ds = build_dpo_dataset(calm, frustrated, seed=seed)
    sft_diverse_ds = build_sft_dataset(calm, seed=seed)
    sft_teacher_ds = build_sft_dataset(teacher, seed=seed)
    dpo_ds.to_json(str(config.DATASETS_DIR / "dpo_pairs.jsonl"))
    sft_diverse_ds.to_json(str(config.DATASETS_DIR / "sft_diverse.jsonl"))
    sft_teacher_ds.to_json(str(config.DATASETS_DIR / "sft_teacher.jsonl"))

    specs: dict[str, ModelSpec] = {}

    dpo_path = train_dpo(dpo_ds, output_dir=Path(out_dir) / "dpo")
    clear_backends()
    specs["dpo"] = config.finetune_spec("DPO-Gemma", dpo_path)

    sft_div_path = train_sft(sft_diverse_ds, output_dir=Path(out_dir) / "sft_diverse")
    clear_backends()
    specs["sft_diverse"] = config.finetune_spec("SFT-Diverse-Gemma", sft_div_path)

    sft_tea_path = train_sft(sft_teacher_ds, output_dir=Path(out_dir) / "sft_teacher")
    clear_backends()
    specs["sft_teacher"] = config.finetune_spec("SFT-Teacher-Gemma", sft_tea_path)

    write_jsonl(
        Path(out_dir) / "finetune_specs.jsonl",
        [{"key": k, "name": s.name, "adapter_path": s.model_id} for k, s in specs.items()],
    )
    return specs
