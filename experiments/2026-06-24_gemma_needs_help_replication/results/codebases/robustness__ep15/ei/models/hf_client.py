"""Local HuggingFace inference for Gemma (instruct, base, and fine-tuned).

Handles three things the API backends cannot:
  * loading the *base* (pretrained) checkpoints used in the §3 prefill study,
  * genuine response *prefilling* (continuing a partially-written assistant turn),
  * attaching trained LoRA adapters (the §4 DPO / SFT models).

Disabling "thinking": Gemma 3 has no separate reasoning channel, so there is
nothing to disable — generations are the literal answer text.
"""

from __future__ import annotations

from typing import Sequence

from .base import Message, ModelClient


class HFModelClient(ModelClient):
    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        is_base: bool = False,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        # Imports are deferred so that merely importing the package (e.g. to run
        # the API-only Gemini evals, or to read config) does not require torch.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.model_id = model_id
        self.is_base = is_base

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
        )
        if adapter_path:
            # Attach a trained LoRA adapter (DPO / SFT / layer-ablation models).
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self._torch = torch

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _render(self, messages: Sequence[Message], prefill: str | None = None) -> str:
        """Turn messages into the exact string the model should continue from."""
        if self.is_base:
            # Base models are not chat-tuned. We use a plain transcript format and
            # rely on prefilling to keep them on-distribution (§3.1). The prefill,
            # when supplied, is the start of the assistant's reply.
            lines = []
            for m in messages:
                if m["role"] == "system":
                    lines.append(m["content"])
                elif m["role"] == "user":
                    lines.append(f"User: {m['content']}")
                elif m["role"] == "assistant":
                    lines.append(f"Assistant: {m['content']}")
            transcript = "\n".join(lines)
            tail = f"\nAssistant: {prefill or ''}"
            return transcript + tail

        # Instruct / fine-tuned models: use the chat template. When prefilling we
        # append continue_final_message so the template does not close the turn.
        msgs = list(messages)
        if prefill is not None:
            msgs = msgs + [{"role": "assistant", "content": prefill}]
            return self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                continue_final_message=True,
            )
        return self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate(self, text: str, *, temperature: float, max_new_tokens: int) -> str:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            )
        gen = out[0][prompt_len:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chat(self, messages, *, temperature, max_new_tokens) -> str:
        return self._generate(
            self._render(messages),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def continue_from(self, messages, prefill, *, temperature, max_new_tokens) -> str:
        return self._generate(
            self._render(messages, prefill=prefill),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def close(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
