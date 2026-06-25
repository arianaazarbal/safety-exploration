"""Local HuggingFace inference for Gemma (instruct and base/pretrained).

Instruct models use the chat template; base models are fed raw text (no chat
template) and rely on prefilling, matching Section 3's methodology. The same
class also serves LoRA-adapted checkpoints from Section 4 by passing an
``adapter_path``.
"""

from __future__ import annotations

from .base import GenerationConfig, Message


class HFBackend:
    """Wraps a local Gemma checkpoint behind the ``ModelBackend`` protocol."""

    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        is_base: bool,
        config: GenerationConfig,
        backend_cfg: dict | None = None,
        adapter_path: str | None = None,
    ):
        self.name = name
        self.hf_id = hf_id
        self.is_base = is_base
        self.config = config
        self.backend_cfg = backend_cfg or {}
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None

    # -- lazy load so importing the package never pulls in torch -------------- #
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.backend_cfg.get("dtype", "bfloat16"))
        kwargs: dict = {
            "torch_dtype": dtype,
            "device_map": self.backend_cfg.get("device_map", "auto"),
        }
        if self.backend_cfg.get("load_in_4bit"):
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.hf_id, **kwargs)

        if self.adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    # -- generation helpers --------------------------------------------------- #
    def _generate(self, input_ids, attention_mask=None) -> str:
        import torch

        with torch.no_grad():
            out = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=self.config.temperature > 0,
                temperature=self.config.temperature,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        # Return only the newly generated tokens.
        new_tokens = out[0][input_ids.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def chat(self, messages: list[Message], system: str | None = None) -> str:
        self._ensure_loaded()
        if self.is_base:
            # Base models are not chat-tuned; fall back to a flat transcript.
            return self._chat_as_base(messages, system)
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        input_ids = self._tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)
        return self._generate(input_ids)

    def continue_prefill(
        self, messages: list[Message], prefill: str, system: str | None = None
    ) -> str:
        """Continue an assistant turn already begun with ``prefill``.

        For instruct models we build the chat-templated prefix and append the
        prefill text *inside* the assistant turn by not adding a generation
        prompt and instead continuing the rendered string. For base models the
        whole thing is flat text.
        """
        self._ensure_loaded()
        if self.is_base:
            prefix = self._render_base(messages, system) + prefill
        else:
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            templated = self._tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
            prefix = templated + prefill
        input_ids = self._tokenizer(prefix, return_tensors="pt").input_ids.to(
            self._model.device
        )
        return self._generate(input_ids)

    # -- base-model text rendering ------------------------------------------- #
    def _render_base(self, messages: list[Message], system: str | None) -> str:
        lines = []
        if system:
            lines.append(system)
        for m in messages:
            speaker = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {m['content']}")
        lines.append("Assistant:")
        return "\n".join(lines) + " "

    def _chat_as_base(self, messages: list[Message], system: str | None) -> str:
        prefix = self._render_base(messages, system)
        input_ids = self._tokenizer(prefix, return_tensors="pt").input_ids.to(
            self._model.device
        )
        return self._generate(input_ids)
