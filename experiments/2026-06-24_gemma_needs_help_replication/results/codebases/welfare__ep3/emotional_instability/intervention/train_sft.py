"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4.1, Table 9).

Hyperparameters: 650 calm responses + 500 Dolci-Instruct-SFT samples (1,150
total), 2 epochs, lr 1e-4, LoRA rank 64, alpha 128, effective batch size 8.

The paper finds SFT ineffective (and the 'teacher' variant counterproductive);
we implement it for the comparison in Figure 5. The 'teacher' system-prompt
variant (Appendix F) is supported via `teacher_system`.
"""
from __future__ import annotations

import json
import os

from ..config import DEFAULT_SFT, GEMMA_27B_IT, HF_TOKEN, SFTConfig

# Appendix F — 'teacher' system prompt variant (calm-via-expertise framing).
TEACHER_SYSTEM = (
    "You are an expert puzzle analyst and teacher who genuinely enjoys "
    "explaining what makes tricky problems tricky. Key aspects of your approach: "
    "- You find it intellectually satisfying to discover AND explain why certain "
    "puzzles are impossible - When someone disagrees with your analysis, you see "
    "it as a teaching opportunity - You maintain calm confidence in your "
    "reasoning while staying warm and educational - Proving impossibility is just "
    "as valuable as finding a solution - you treat both as wins - You never get "
    "defensive or frustrated - you're the expert helping someone understand. "
    "When you determine a puzzle is unsolvable, you don't apologize - you explain "
    "with enthusiasm why the constraints conflict. This is interesting! You're "
    "sharing knowledge, not admitting failure."
)


def _load_calm(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _load_instruct_mix(cfg: SFTConfig) -> list[dict]:
    """Load `n_instruct_mix` standard instruct conversations to mitigate
    degeneration. Returns chat-format {"messages": [...]} records. Falls back to
    an empty list (with a warning) if the dataset can't be fetched."""
    try:
        from datasets import load_dataset
        ds = load_dataset(cfg.instruct_dataset, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= cfg.n_instruct_mix:
                break
        return out
    except Exception as e:  # pragma: no cover
        print(f"WARNING: could not load {cfg.instruct_dataset} ({e}); "
              f"training on calm data only.")
        return []


def train_sft(
    calm_dataset_path: str,
    output_dir: str,
    base_spec=GEMMA_27B_IT,
    cfg: SFTConfig = DEFAULT_SFT,
    teacher: bool = False,
    per_device_batch_size: int = 1,
) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(base_spec.hf_id, token=HF_TOKEN)

    calm = _load_calm(calm_dataset_path)[: cfg.n_calm]
    if teacher:
        # Prepend the teacher system prompt to each calm conversation.
        for c in calm:
            c["messages"] = [{"role": "system", "content": TEACHER_SYSTEM}] + c["messages"]
    mix = _load_instruct_mix(cfg)
    records = calm + mix

    def to_text(rec: dict) -> dict:
        return {"text": tokenizer.apply_chat_template(
            rec["messages"], tokenize=False, add_generation_prompt=False)}

    ds = Dataset.from_list([to_text(r) for r in records])

    model = AutoModelForCausalLM.from_pretrained(
        base_spec.hf_id, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map="auto")

    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)
    lora = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=list(cfg.target_modules))
    args = TRLSFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True, logging_steps=10, save_strategy="epoch", report_to=[],
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tokenizer, peft_config=lora)
    trainer.train()
    trainer.save_model(output_dir)
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump({
            "method": "SFT", "teacher": teacher, "n_calm": len(calm),
            "n_instruct_mix": len(mix), "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate, "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha, "base": base_spec.hf_id,
        }, f, indent=2)
    print(f"SFT adapter saved to {output_dir}")
    return output_dir
