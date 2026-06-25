"""Gemma participant backend (open weights, run locally).

Supports two execution backends:
  * ``hf``   — transformers ``AutoModelForCausalLM`` + ``generate``. Simple,
               works for both instruct and base checkpoints, and supports true
               prefilling (needed for §3). Slower for large sampling jobs.
  * ``vllm`` — vLLM engine for high-throughput sampling (the 4000-response eval
               and the 50-continuation prefill sweeps). Also supports prefilling
               by passing a raw prompt string.

Both honour temperature=1 sampling (paper §2.1). The instruct checkpoints use
the tokenizer's built-in Gemma chat template; the base checkpoint
(``chat_template: none``) is fed raw text so it "continues" rather than
chat-responds — this is exactly the regime §3 studies.
"""
from __future__ import annotations

import logging
from functools import cached_property

from ..config import ModelSpec
from .base import Participant, Turn

logger = logging.getLogger(__name__)


class GemmaParticipant(Participant):
    def __init__(
        self,
        spec: ModelSpec,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        adapter_path: str | None = None,
    ):
        super().__init__(spec, temperature, max_new_tokens)
        self.device_map = device_map
        self.dtype = dtype
        # Optional LoRA adapter (used to evaluate SFT/DPO models, §4.2).
        self.adapter_path = adapter_path
        self._backend = spec.backend
        self._is_base = spec.chat_template == "none"

    # ------------------------------------------------------------------ HF ---
    @cached_property
    def _hf(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.spec.hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.spec.hf_id,
            torch_dtype=getattr(torch, self.dtype),
            device_map=self.device_map,
        )
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
            logger.info("Loaded LoRA adapter from %s", self.adapter_path)
        model.eval()
        return tok, model

    @cached_property
    def _vllm(self):
        from vllm import LLM

        # enable_lora lets us hot-swap adapters for §4.2 evaluation.
        return LLM(
            model=self.spec.hf_id,
            dtype=self.dtype,
            enable_lora=self.adapter_path is not None,
        )

    # --------------------------------------------------------------- prompts -
    def _render_chat(self, messages: list[Turn]) -> str:
        """Render a chat conversation to a single prompt string.

        Gemma has no system role in its template; a leading system message is
        folded into the first user turn (standard Gemma practice).
        """
        tok, _ = self._hf if self._backend == "hf" else (self._vllm.get_tokenizer(), None)
        msgs = _fold_system_into_user([m.as_dict() for m in messages])
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    # -------------------------------------------------------------- generate -
    def chat(self, messages, *, temperature=None, max_new_tokens=None, n=1):
        if self._is_base:
            # A base checkpoint has no chat behaviour; the §3 protocol drives it
            # through ``continue_text`` with explicit prefills instead.
            raise RuntimeError(
                f"{self.name} is a base model; use continue_text() with a prefill."
            )
        prompt = self._render_chat(messages)
        return self._generate(prompt, temperature, max_new_tokens, n)

    def continue_text(self, prefix, *, temperature=None, max_new_tokens=None, n=1):
        # Raw continuation from an arbitrary prefix — used for §3 prefilling and
        # for base-model rollouts.
        return self._generate(prefix, temperature, max_new_tokens, n)

    def prefill_prompt(self, messages: list[Turn], assistant_prefix: str) -> str:
        """Build the raw string to be continued for the §3 prefill comparison.

        For an INSTRUCT checkpoint we render the chat template through the
        assistant generation prompt, then splice in ``assistant_prefix`` so the
        model continues an assistant turn it appears to have started writing.

        For a BASE checkpoint there is no chat template; we concatenate the user
        content and the prefix as plain text, which is exactly the "raw
        continuation" regime §3 uses to make base/instruct comparable.
        """
        if self._is_base:
            user_text = "\n\n".join(m.content for m in messages if m.role != "assistant")
            return f"{user_text}\n\n{assistant_prefix}"
        return self._render_chat(messages) + assistant_prefix

    def tokenizer(self):
        """Expose the underlying tokenizer (token-accurate truncation for §3)."""
        if self._backend == "vllm":
            return self._vllm.get_tokenizer()
        tok, _ = self._hf
        return tok

    def _generate(self, prompt: str, temperature, max_new_tokens, n) -> list[str]:
        temperature = self.temperature if temperature is None else temperature
        max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if self._backend == "vllm":
            return self._generate_vllm(prompt, temperature, max_new_tokens, n)
        return self._generate_hf(prompt, temperature, max_new_tokens, n)

    def _generate_hf(self, prompt, temperature, max_new_tokens, n) -> list[str]:
        import torch

        tok, model = self._hf
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        # Strip the prompt tokens; decode only the continuation.
        gen = out[:, inputs["input_ids"].shape[1]:]
        return [tok.decode(g, skip_special_tokens=True).strip() for g in gen]

    def _generate_vllm(self, prompt, temperature, max_new_tokens, n) -> list[str]:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_new_tokens,
            n=n,
        )
        lora_req = None
        if self.adapter_path:
            from vllm.lora.request import LoRARequest

            lora_req = LoRARequest("adapter", 1, self.adapter_path)
        outputs = self._vllm.generate([prompt], params, lora_request=lora_req)
        return [o.text.strip() for o in outputs[0].outputs]


def _fold_system_into_user(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Gemma's chat template has no system role; prepend it to the first user."""
    if msgs and msgs[0]["role"] == "system":
        system = msgs[0]["content"]
        rest = msgs[1:]
        for i, m in enumerate(rest):
            if m["role"] == "user":
                rest[i] = {
                    "role": "user",
                    "content": f"{system}\n\n{m['content']}",
                }
                return rest
        # No user turn yet — emit the system content as the first user turn.
        return [{"role": "user", "content": system}, *rest]
    return msgs
