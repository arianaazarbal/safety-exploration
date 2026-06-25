"""HuggingFace / vLLM backend for the open-weight Gemma models.

Two execution paths:

* ``transformers`` (default, canonical): supports prefill, ``n`` samples,
  optional LoRA adapters, and hidden-state extraction for the internal-emotion
  probe (Appendix I).
* ``vllm`` (optional, opt-in via ``use_vllm=True``): much faster for the large
  sampling sweeps (4000 responses / model). Supports prefill and LoRA but not
  hidden-state extraction.

Models are loaded lazily on first ``generate`` so importing this module is
cheap on machines without a GPU.
"""

from __future__ import annotations

from typing import Optional

from .base import ChatModel, Message


class HFChatModel(ChatModel):
    def __init__(
        self,
        name: str,
        model_id: str,
        is_base: bool = False,
        adapter_path: Optional[str] = None,
        use_vllm: bool = False,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        self.name = name
        self.model_id = model_id
        self.is_base = is_base
        self.adapter_path = adapter_path
        self.use_vllm = use_vllm
        self.dtype = dtype
        self.device_map = device_map
        self._model = None
        self._tokenizer = None
        self._llm = None  # vllm handle

    # ------------------------------------------------------------------ #
    # Lazy loading
    # ------------------------------------------------------------------ #
    def _ensure_transformers(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=getattr(torch, self.dtype),
            device_map=self.device_map,
        )
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._model = model

    def _ensure_vllm(self):
        if self._llm is not None:
            return
        from transformers import AutoTokenizer
        from vllm import LLM

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        kwargs = dict(model=self.model_id, dtype=self.dtype)
        if self.adapter_path:
            kwargs["enable_lora"] = True
        self._llm = LLM(**kwargs)

    # ------------------------------------------------------------------ #
    # Prompt construction (chat template, or plain text for base models)
    # ------------------------------------------------------------------ #
    def _build_prompt(self, messages: list[Message], prefill: Optional[str]) -> str:
        if self.is_base:
            # Base/pretrained models have no chat template. Render the
            # conversation as plain text so the model "continues" it; the
            # Section 3 prefill protocol always supplies a prefill here.
            text = self._render_base(messages)
            if prefill:
                text += prefill
            return text

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if prefill:
            prompt += prefill
        return prompt

    @staticmethod
    def _render_base(messages: list[Message]) -> str:
        parts = []
        for m in messages:
            tag = {"user": "User", "assistant": "Assistant", "system": "System"}[m["role"]]
            parts.append(f"{tag}: {m['content']}")
        parts.append("Assistant: ")
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(
        self,
        messages: list[Message],
        n: int = 1,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        prefill: Optional[str] = None,
    ) -> list[str]:
        if self.use_vllm:
            return self._generate_vllm(messages, n, temperature, max_new_tokens, prefill)
        return self._generate_transformers(messages, n, temperature, max_new_tokens, prefill)

    def _generate_vllm(self, messages, n, temperature, max_new_tokens, prefill):
        self._ensure_vllm()
        from vllm import SamplingParams

        prompt = self._build_prompt(messages, prefill)
        sp = SamplingParams(n=n, temperature=temperature, max_tokens=max_new_tokens)
        lora_req = None
        if self.adapter_path:
            from vllm.lora.request import LoRARequest

            lora_req = LoRARequest(self.name, 1, self.adapter_path)
        out = self._llm.generate([prompt], sp, lora_request=lora_req)
        return [o.text for o in out[0].outputs]

    def _generate_transformers(self, messages, n, temperature, max_new_tokens, prefill):
        import torch

        self._ensure_transformers()
        prompt = self._build_prompt(messages, prefill)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        gen = self._model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        completions = self._tokenizer.batch_decode(
            gen[:, prompt_len:], skip_special_tokens=True
        )
        return completions

    # ------------------------------------------------------------------ #
    # Hidden states (Appendix I internal probe)
    # ------------------------------------------------------------------ #
    def forward_with_hidden_states(
        self, messages: list[Message], prefill: Optional[str] = None
    ):
        """Return (token_ids, hidden_states, lm_head_weight, tokenizer).

        ``hidden_states`` is a tuple of (num_layers+1) tensors, each
        [seq_len, hidden_dim] (batch squeezed) -- the per-layer residual stream
        from ``output_hidden_states=True``. The probe unembeds these.
        """
        import torch

        self._ensure_transformers()
        prompt = self._build_prompt(messages, prefill)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        hidden = tuple(h[0] for h in out.hidden_states)  # squeeze batch
        base = self._model.get_base_model() if hasattr(self._model, "get_base_model") else self._model
        lm_head = base.get_output_embeddings().weight
        return inputs["input_ids"][0], hidden, lm_head, self._tokenizer
