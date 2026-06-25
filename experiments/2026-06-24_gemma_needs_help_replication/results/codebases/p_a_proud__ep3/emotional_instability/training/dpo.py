"""DPO finetuning of Gemma-3-27B-it (§4.1, Appendix E).

280 preference pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA rank-64 / alpha-64 on all
projection layers, effective batch size 8. This is the paper's headline
intervention: it drops the average %>=5 frustration from 35% to 0.3%.

Pairs use the conversational DPO format: ``prompt`` is the (clean) chat history
up to the final user turn; ``chosen`` / ``rejected`` are the calm / frustrated
final assistant responses.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..io_utils import ensure_dir, read_jsonl
from ..logging_utils import get_logger, seed_everything
from .lora import build_peft_config
from .sft import _batch_split

logger = get_logger(__name__)


def train_dpo(
    cfg: Config,
    *,
    pairs_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    seed_everything(cfg.seed)
    dpo = cfg.training.dpo
    base_spec = cfg.model(cfg.training.base_model)
    output_dir = Path(output_dir or Path(cfg.output_dir) / "training" / "dpo")
    adapter_dir = ensure_dir(output_dir / "adapter")

    raw_pairs = list(read_jsonl(pairs_path))
    if not raw_pairs:
        raise RuntimeError(f"No DPO pairs found at {pairs_path}")
    logger.info("DPO on %d preference pairs", len(raw_pairs))

    tokenizer = AutoTokenizer.from_pretrained(base_spec.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Conversational DPO format: prompt as messages, chosen/rejected as messages.
    records = [
        {
            "prompt": p["prompt"],
            "chosen": [{"role": "assistant", "content": p["chosen"]}],
            "rejected": [{"role": "assistant", "content": p["rejected"]}],
        }
        for p in raw_pairs
    ]
    dataset = Dataset.from_list(records)

    model = AutoModelForCausalLM.from_pretrained(
        base_spec.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    per_device, accum = _batch_split(dpo.effective_batch_size)
    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=dpo.epochs,
        learning_rate=dpo.learning_rate,
        beta=dpo.beta,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        max_length=cfg.training.max_seq_len,
        max_prompt_length=cfg.training.max_seq_len // 2,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )
    peft_config = build_peft_config(cfg.training.lora, alpha=dpo.lora_alpha)

    # With a PEFT config, TRL uses the same base model as the (frozen) reference,
    # so no separate ref_model is required.
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("Saved DPO adapter to %s", adapter_dir)
    return adapter_dir
