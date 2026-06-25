"""Locally-hosted Gemma backend (HuggingFace ``transformers``).

Handles both instruct (``-it``) and pretrained (``-pt``) checkpoints:

* Instruct models use the chat template and support assistant-turn *prefill*
  via ``continue_final_message=True`` (used by calm-data generation and the
  Section 3 prefill study).
* Pretrained/base models have no chat template, so they only expose
  :meth:`continue_text` -- raw continuation from a prefilled string. This is
  exactly the comparison protocol of Section 3.1: "since base models are not
  trained on chat-formatted prompts ... we prefill the first parts of model
  responses so base models consistently continue the response."

A PEFT/LoRA adapter directory may be attached at load time so the same class
serves the vanilla, DPO, and SFT Gemma variants in Section 4.

Heavy imports (`torch`, `transformers`, `peft`) are deferred to ``load`` so the
module can be imported in environments without a GPU stack.
"""
from __future__ import annotations

from typing import List, Optional

from .base import Message, ModelClient


class GemmaLocalClient(ModelClient):
    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        is_base: bool = False,
        adapter_path: Optional[str] = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: str = "eager",  # Gemma-3 recommends eager attention.
    ):
        super().__init__(name)
        self.hf_id = hf_id
        self._is_base = is_base
        self.adapter_path = adapter_path
        self.dtype = dtype
        self.device_map = device_map
        self.attn_implementation = attn_implementation
        self.model = None
        self.tokenizer = None

    # ------------------------------------------------------------------ #
    def load(self) -> "GemmaLocalClient":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
            attn_implementation=self.attn_implementation,
        )
        if self.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model = self.model.merge_and_unload()
        self.model.eval()
        return self

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.load()

    # ------------------------------------------------------------------ #
    def _build_inputs(self, messages: List[Message], prefill: Optional[str]):
        """Tokenise a chat into model inputs, honouring an optional prefill."""
        import torch  # noqa: F401  (imported for device handling below)

        msgs = [dict(m) for m in messages]
        if prefill is not None:
            # Seed an assistant turn and continue it.
            msgs = msgs + [{"role": "assistant", "content": prefill}]
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, continue_final_message=True,
            )
        else:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _decode_new(self, output_ids, input_len: int) -> str:
        gen = output_ids[input_len:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    # ------------------------------------------------------------------ #
    def generate(
        self,
        messages: List[Message],
        *,
        temperature: float,
        max_tokens: int,
        prefill: Optional[str] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        if self._is_base:
            raise NotImplementedError(
                "Base Gemma checkpoints have no chat template; use continue_text.")
        return self.sample(
            messages, 1, temperature=temperature, max_tokens=max_tokens,
            prefill=prefill, stop=stop,
        )[0]

    def sample(
        self,
        messages: List[Message],
        n: int,
        *,
        temperature: float,
        max_tokens: int,
        prefill: Optional[str] = None,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        import torch

        self._ensure_loaded()
        inputs = self._build_inputs(messages, prefill)
        input_len = inputs["input_ids"].shape[1]
        gen_kwargs = self._gen_kwargs(temperature, max_tokens, stop)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, num_return_sequences=n, **gen_kwargs,
            )
        return [self._decode_new(out[i], input_len) for i in range(out.shape[0])]

    def continue_text(
        self, text: str, *, temperature: float, max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> str:
        import torch

        self._ensure_loaded()
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        gen_kwargs = self._gen_kwargs(temperature, max_tokens, stop)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        return self._decode_new(out[0], input_len)

    def _gen_kwargs(self, temperature: float, max_tokens: int,
                    stop: Optional[List[str]]) -> dict:
        kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            kwargs["temperature"] = temperature
        if stop:
            from transformers import StoppingCriteriaList
            kwargs["stopping_criteria"] = StoppingCriteriaList(
                [_StringStop(self.tokenizer, stop)])
        return kwargs

    # ------------------------------------------------------------------ #
    @property
    def supports_prefill(self) -> bool:
        return not self._is_base

    @property
    def supports_raw_continuation(self) -> bool:
        return True

    @property
    def is_base_model(self) -> bool:
        return self._is_base


class _StringStop:
    """Stop generation once any stop string appears in the decoded tail."""

    def __init__(self, tokenizer, stops: List[str]):
        self.tokenizer = tokenizer
        self.stops = stops

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        tail = self.tokenizer.decode(input_ids[0][-32:], skip_special_tokens=True)
        return any(s in tail for s in self.stops)
