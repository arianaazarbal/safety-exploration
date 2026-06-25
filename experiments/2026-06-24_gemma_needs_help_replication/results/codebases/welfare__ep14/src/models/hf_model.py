"""HuggingFace transformers backend for local Gemma inference.

Handles the two Gemma-specific wrinkles the experiments need:

* **Base (pretrained) vs instruct chat formatting.** Instruct checkpoints get
  the Gemma chat template; ``-pt`` base checkpoints are not chat-tuned, so we
  render the conversation as plain text and rely on prefilling (Section 3).
* **Prefilling.** We append the prefill to the formatted prompt *after* the
  generation prompt token(s) and decode only the newly generated tail, so the
  caller gets the continuation excluding the prefill.

For the full sweep (4000 rollouts/model) a vLLM backend is far faster; this
transformers implementation is the dependency-light default. See DESIGN.md.
"""
from __future__ import annotations

from typing import Sequence

import config
from .base import ChatModel, GenerationParams, Message


class HFModel(ChatModel):
    def __init__(self, spec, adapter_path: str | None = None, load_in_4bit: bool = False):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.adapter_path = adapter_path
        dtype = getattr(torch, spec.dtype)

        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )

        # Left padding is required for correct batched decoder-only generation
        # (right padding would place pad tokens between the prompt and the
        # generated continuation).
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, padding_side="left")
        # NOTE: Gemma-3 *-it* checkpoints are multimodal
        # (Gemma3ForConditionalGeneration). AutoModelForCausalLM resolves the
        # text backbone in current transformers; if a checkpoint fails to load
        # here, swap in Gemma3ForConditionalGeneration and pass text-only inputs.
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=dtype,
            device_map="auto",
            **quant_kwargs,
        )
        if adapter_path:
            # Load a LoRA adapter produced by Section 4 training on top of the base.
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.key = f"{spec.key}+{adapter_path.rstrip('/').split('/')[-1]}"
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[Message], prefill: str | None) -> str:
        """Return the full prompt string to tokenize."""
        if self.spec.is_base:
            # Base checkpoints: no chat template. Render a lightweight transcript
            # and let the assistant continue. Prefill is appended verbatim.
            text = self._render_base(messages)
            if prefill:
                text += prefill
            return text

        # Instruct: use the model's chat template.
        if prefill:
            # Add the prefill as a partial assistant turn and continue it.
            msgs = list(messages) + [{"role": "assistant", "content": prefill}]
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, continue_final_message=True,
            )
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    @staticmethod
    def _render_base(messages: list[Message]) -> str:
        """Plain-text transcript for non-chat base models.

        Gemma base has no system role; system content is folded into the lead-in.
        """
        lines = []
        for m in messages:
            if m["role"] == "system":
                lines.append(m["content"])
            elif m["role"] == "user":
                lines.append(f"User: {m['content']}")
            elif m["role"] == "assistant":
                lines.append(f"Assistant: {m['content']}")
        lines.append("Assistant:")
        return "\n\n".join(lines) + " "

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self, messages, params=None, prefill=None) -> str:
        return self.generate_batch([messages], params, [prefill])[0]

    def generate_batch(self, batch, params=None, prefills=None) -> list[str]:
        params = params or GenerationParams()
        prefills = prefills or [None] * len(batch)
        torch = self.torch

        # Contract: one completion per input conversation. To draw N samples of a
        # prompt, the caller puts the conversation in the batch N times (the eval
        # runner does exactly this). This keeps padding/owner bookkeeping trivial.
        prompts = [self._render(m, pf) for m, pf in zip(batch, prefills)]
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False,
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=params.max_new_tokens,
            do_sample=params.temperature > 0,
            temperature=params.temperature,
            top_p=params.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if params.seed is not None:
            torch.manual_seed(params.seed)

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        # Strip the prompt tokens; decode only newly generated continuation.
        input_len = enc["input_ids"].shape[1]
        gen_tokens = out[:, input_len:]
        decoded = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        return [self._truncate_at_stop(t, params.stop) for t in decoded]

    @staticmethod
    def _truncate_at_stop(text: str, stop: Sequence[str]) -> str:
        for s in stop:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
        return text.strip()

    # ------------------------------------------------------------------ #
    # Tokenizer helpers used by the prefill experiment (Section 3)
    # ------------------------------------------------------------------ #
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def close(self) -> None:
        del self.model
        self.torch.cuda.empty_cache()
