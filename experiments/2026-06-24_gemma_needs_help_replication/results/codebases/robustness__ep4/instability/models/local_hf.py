"""Local HuggingFace transformers backend.

Required for everything API providers can't do:
  * loading trained LoRA adapters (DPO / SFT fine-tunes) for re-evaluation,
  * base-model (``*-pt``) continuation with no chat template,
  * prefill continuation from a partial assistant turn (Section 3).

For the large 4000-response sweeps this is slow; vLLM (see ``vllm_backend`` note
in DESIGN.md) is recommended there. This backend prioritises correctness and
the prefill/adapter capabilities the experiments specifically need.
"""
from __future__ import annotations

from typing import Optional

from .base import ChatMessage, ChatModel, Completion


class LocalHFModel(ChatModel):
    def __init__(self, spec, device_map: str = "auto", dtype: str = "bfloat16",
                 load_in_4bit: bool = False):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = dict(device_map=device_map, torch_dtype=getattr(torch, dtype))
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)

        if spec.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, spec.adapter_path)
        self.model.eval()

    # ------------------------------------------------------------------ #
    def _render(self, messages: list[ChatMessage], add_generation_prompt: bool) -> str:
        """Render messages to a prompt string.

        Instruct models use the chat template. Base models (no chat template)
        get a plain role-tagged transcript so prefill comparisons are fair.
        """
        if self.spec.is_instruct and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        # Base-model fallback: simple transcript format.
        parts = []
        for m in messages:
            parts.append(f"{m['role'].capitalize()}: {m['content']}")
        if add_generation_prompt:
            parts.append("Assistant:")
        return "\n\n".join(parts)

    def _sample(self, prompt_text: str, *, temperature, max_new_tokens, n, seed):
        import torch

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        texts = self.tokenizer.batch_decode(
            out[:, prompt_len:], skip_special_tokens=True
        )
        return [Completion(text=t) for t in texts]

    def generate(self, messages, *, temperature, max_new_tokens, n=1, seed=None):
        prompt_text = self._render(messages, add_generation_prompt=True)
        return self._sample(
            prompt_text, temperature=temperature, max_new_tokens=max_new_tokens,
            n=n, seed=seed,
        )

    def continue_prefill(self, messages, prefill, *, temperature, max_new_tokens,
                         n=1, seed=None):
        # Render the conversation up to (and including the start of) the final
        # assistant turn, then append the prefill text so the model continues it.
        prompt_text = self._render(messages, add_generation_prompt=True) + prefill
        return self._sample(
            prompt_text, temperature=temperature, max_new_tokens=max_new_tokens,
            n=n, seed=seed,
        )
