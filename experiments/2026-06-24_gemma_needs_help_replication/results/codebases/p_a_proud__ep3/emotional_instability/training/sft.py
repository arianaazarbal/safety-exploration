"""SFT finetuning of Gemma-3-27B-it on calm data (§4.1, Appendix E).

650 calm responses + 500 standard instruct samples, 2 epochs, lr 1e-4, LoRA
rank-64 / alpha-128 on all projection layers, effective batch size 8.

The paper finds SFT ineffective (and the 'teacher' variant *increases* distress,
Appendix F) — we implement it faithfully so that negative result can be
reproduced. ``use_teacher_data`` is handled at the data-generation stage
(``DataGenConfig.teacher_system_prompt``); this trainer is data-source agnostic.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..io_utils import ensure_dir, read_jsonl
from ..logging_utils import get_logger, seed_everything
from .lora import build_peft_config

logger = get_logger(__name__)


def train_sft(
    cfg: Config,
    *,
    calm_path: str | Path,
    instruct_mix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    seed_everything(cfg.seed)
    sft = cfg.training.sft
    base_spec = cfg.model(cfg.training.base_model)
    output_dir = Path(output_dir or Path(cfg.output_dir) / "training" / "sft")
    adapter_dir = ensure_dir(output_dir / "adapter")

    examples = list(read_jsonl(calm_path))
    if instruct_mix_path:
        examples += list(read_jsonl(instruct_mix_path))
    if not examples:
        raise RuntimeError(f"No SFT examples found at {calm_path}")
    logger.info("SFT on %d examples", len(examples))

    tokenizer = AutoTokenizer.from_pretrained(base_spec.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_text(ex):
        return {
            "text": tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    dataset = Dataset.from_list(examples).map(to_text, remove_columns=["messages"])

    model = AutoModelForCausalLM.from_pretrained(
        base_spec.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    per_device, accum = _batch_split(sft.effective_batch_size)
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=sft.epochs,
        learning_rate=sft.learning_rate,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        max_length=cfg.training.max_seq_len,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        dataset_text_field="text",
    )
    peft_config = build_peft_config(cfg.training.lora, alpha=sft.lora_alpha)

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("Saved SFT adapter to %s", adapter_dir)
    return adapter_dir


def _batch_split(effective: int) -> tuple[int, int]:
    """Split an effective batch into (per_device, grad_accum). Single-GPU, so we
    keep per_device=1 for a 27B model and put the rest in accumulation."""
    return 1, max(1, effective)
