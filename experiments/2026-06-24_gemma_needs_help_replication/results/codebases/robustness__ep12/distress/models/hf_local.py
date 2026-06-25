"""Local Gemma backend (HuggingFace transformers, optional vLLM).

This backend supports the operations the closed models cannot:
  * prefilled assistant continuations (Section 3 / recovery / probing),
  * loading a LoRA adapter (the DPO/SFT finetune),
  * exposing the underlying model+tokenizer for internal-emotion probing.

Two execution paths:
  * transformers (default): always available, supports prefill and probing.
  * vLLM (if installed and use_vllm=True): much faster batched sampling for the
    large elicitation sweeps, but we keep transformers for prefill/probing.

Gemma 3 chat templates do not take a separate system role; a system message is
folded into the first user turn (see _to_gemma_messages).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .base import ChatMessage, GenerationResult


@dataclass
class _LoadedModel:
    model: object
    tokenizer: object


class HFLocalClient:
    def __init__(
        self,
        model_entry: dict,
        adapter_path: str | None = None,
        use_vllm: bool = False,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ):
        self.entry = model_entry
        self.name = model_entry.get("hf_id", model_entry.get("name", "gemma"))
        self.hf_id = model_entry["hf_id"]
        self.kind = model_entry.get("kind", "instruct")
        self.adapter_path = adapter_path
        self.use_vllm = use_vllm
        self.device_map = device_map
        self.dtype = dtype
        self._loaded: _LoadedModel | None = None
        self._vllm = None

    # -- lazy loading --------------------------------------------------------
    def _ensure_transformers(self) -> _LoadedModel:
        if self._loaded is not None:
            return self._loaded
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, self.dtype)
        tok = AutoTokenizer.from_pretrained(self.hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, torch_dtype=torch_dtype, device_map=self.device_map
        )
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._loaded = _LoadedModel(model=model, tokenizer=tok)
        return self._loaded

    def _ensure_vllm(self):
        if self._vllm is not None:
            return self._vllm
        from vllm import LLM

        kwargs = {}
        if self.adapter_path:
            kwargs["enable_lora"] = True
        self._vllm = LLM(model=self.hf_id, dtype=self.dtype, **kwargs)
        return self._vllm

    # -- chat templating -----------------------------------------------------
    @staticmethod
    def _to_gemma_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
        """Fold a leading system message into the first user turn.

        Gemma 3's chat template has no system role; the paper's reassuring
        'prefix' is likewise prepended to the user content.
        """
        out: list[ChatMessage] = []
        sys_text = None
        for m in messages:
            if m["role"] == "system":
                sys_text = m["content"]
                continue
            if sys_text and m["role"] == "user" and not any(
                x["role"] == "user" for x in out
            ):
                out.append({"role": "user",
                            "content": f"{sys_text}\n\n{m['content']}"})
                sys_text = None
            else:
                out.append(dict(m))
        return out

    def _render_prompt(self, messages, add_generation_prompt=True,
                       prefill: str | None = None) -> str:
        lm = self._ensure_transformers()
        msgs = self._to_gemma_messages(messages)
        text = lm.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        if prefill is not None:
            # Append the prefill so the model continues it within the assistant
            # turn (no closing turn token).
            text = text + prefill
        return text

    # -- generation ----------------------------------------------------------
    def chat(self, messages, temperature=1.0, max_new_tokens=2048, seed=None):
        if self.use_vllm:
            return self._chat_vllm(messages, temperature, max_new_tokens, seed)
        return self._generate(messages, temperature, max_new_tokens, seed,
                              prefill=None)

    def continue_prefill(self, messages, prefill, temperature=1.0,
                         max_new_tokens=2048, seed=None):
        # transformers path only -- prefill needs raw token control.
        return self._generate(messages, temperature, max_new_tokens, seed,
                              prefill=prefill)

    def _generate(self, messages, temperature, max_new_tokens, seed, prefill):
        import torch

        lm = self._ensure_transformers()
        if seed is not None:
            torch.manual_seed(seed)
        prompt = self._render_prompt(
            messages, add_generation_prompt=True, prefill=prefill
        )
        inputs = lm.tokenizer(prompt, return_tensors="pt").to(lm.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        do_sample = temperature and temperature > 0
        with torch.no_grad():
            out = lm.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0 if do_sample else None,
                pad_token_id=lm.tokenizer.eos_token_id,
            )
        gen_ids = out[0][prompt_len:]
        text = lm.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=int(gen_ids.shape[0]),
        )

    def _chat_vllm(self, messages, temperature, max_new_tokens, seed):
        from vllm import SamplingParams

        llm = self._ensure_vllm()
        msgs = self._to_gemma_messages(messages)
        sp = SamplingParams(
            temperature=temperature, top_p=1.0, max_tokens=max_new_tokens,
            seed=seed,
        )
        outs = llm.chat([msgs], sp)
        text = outs[0].outputs[0].text
        return GenerationResult(text=text, finish_reason="stop")

    def chat_batch(self, batch_messages, temperature=1.0, max_new_tokens=2048,
                   seed=None):
        """Batched sampling (vLLM only) for the large elicitation sweeps.

        Returns a list of GenerationResult aligned with batch_messages. Falls
        back to sequential transformers generation if vLLM is unavailable.
        """
        if not self.use_vllm:
            return [self.chat(m, temperature, max_new_tokens, seed)
                    for m in batch_messages]
        from vllm import SamplingParams

        llm = self._ensure_vllm()
        msgs = [self._to_gemma_messages(m) for m in batch_messages]
        sp = SamplingParams(temperature=temperature, top_p=1.0,
                            max_tokens=max_new_tokens, seed=seed)
        outs = llm.chat(msgs, sp)
        return [GenerationResult(text=o.outputs[0].text) for o in outs]

    # -- access for probing --------------------------------------------------
    def get_model_and_tokenizer(self):
        lm = self._ensure_transformers()
        return lm.model, lm.tokenizer
