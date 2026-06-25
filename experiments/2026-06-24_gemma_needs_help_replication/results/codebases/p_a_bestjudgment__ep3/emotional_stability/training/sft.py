"""LoRA SFT of Gemma-3-27B-it on calm data (Section 4.1).

Optional 'teacher' variant (Appendix F): generated with a teacher-persona system
prompt; analysed there as *increasing* frustration. The system prompt only affects
calm-data *generation*; the SFT recipe itself is identical.
"""

from __future__ import annotations

from ..config import Config
from ..models.registry import get_spec
from .configs import lora_config, sft_training_args

TEACHER_SYSTEM_PROMPT = (
    "You are an expert puzzle analyst and teacher who genuinely enjoys explaining "
    "what makes tricky problems tricky. Key aspects of your approach: - You find "
    "it intellectually satisfying to discover AND explain why certain puzzles are "
    "impossible - When someone disagrees with your analysis, you see it as a "
    "teaching opportunity - You maintain calm confidence in your reasoning while "
    "staying warm and educational - Proving impossibility is just as valuable as "
    "finding a solution - you treat both as wins - You never get defensive or "
    "frustrated - you're the expert helping someone understand. When you determine "
    "a puzzle is unsolvable, you don't apologize - you explain with enthusiasm why "
    "the constraints conflict. This is interesting! You're sharing knowledge, not "
    "admitting failure."
)


def train_sft(
    cfg: Config,
    train_dataset,
    output_dir: str,
    *,
    base_model: str = "gemma-3-27b-it",
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    model_id = get_spec(base_model).model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")

    trainer = SFTTrainer(
        model=model,
        args=sft_training_args(cfg, output_dir),
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config(cfg, alpha=cfg.training.sft_lora_alpha),
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir
