"""Local HuggingFace backend for Gemma (instruct, base, and LoRA-adapted).

Supports two execution paths:

* **vLLM** (default if installed and ``EI_USE_VLLM=1``) — fast batched sampling,
  the practical choice for collecting thousands of rollouts.
* **transformers** — fallback that works anywhere; slower.

The backend handles three model roles:

* instruct (``gemma-3-27b-it`` / ``-12b-it``): chat template applied normally.
* base/pretrained (``-pt``): not chat-tuned, so for ``generate`` we still apply
  a minimal chat-like format, but the primary use of base models is
  ``continue_prefill`` for the Section-3 experiment.
* adapted: an instruct model with a LoRA adapter (the DPO/SFT result) loaded on
  top, selected via ``adapter_path``.

Prefilling is implemented by rendering the chat template *up to* the start of
the assistant turn, appending the ``prefill`` string, and continuing generation
with the tokenizer's chat continuation (``continue_final_message=True`` where
supported, else manual template construction).
"""

from __future__ import annotations

import os
from typing import Optional

import config
from emotional_instability.models.base import GenResult, Message
from emotional_instability.utils import log


class HFBackend:
    def __init__(self, spec: config.ModelSpec, adapter_path: Optional[str] = None):
        self.spec = spec
        self.adapter_path = adapter_path
        self.is_base = spec.role == "base"
        self._use_vllm = config.RUN.use_vllm and adapter_path is None and _vllm_available()
        self._loaded = False
        self._tokenizer = None
        self._model = None
        self._llm = None  # vLLM handle

    # ------------------------------------------------------------------ #
    # Lazy loading (so importing the module never touches GPUs/network)
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self):
        if self._loaded:
            return
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)
        if self._use_vllm:
            from vllm import LLM

            log.info("Loading %s via vLLM", self.spec.model_id)
            self._llm = LLM(model=self.spec.model_id, dtype="bfloat16", trust_remote_code=True)
        else:
            import torch
            from transformers import AutoModelForCausalLM

            log.info("Loading %s via transformers%s", self.spec.model_id,
                     f" + adapter {self.adapter_path}" if self.adapter_path else "")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.spec.model_id, torch_dtype=torch.bfloat16, device_map="auto"
            )
            if self.adapter_path:
                from peft import PeftModel

                self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
            self._model.eval()
        self._loaded = True

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[Message], prefill: str | None = None) -> str:
        """Render messages to a single prompt string.

        For instruct/adapted models we use the tokenizer chat template. For base
        models we use a lightweight role-tagged format, because they were not
        trained on the chat template; this matches the paper's use of base models
        purely as prefill continuers.
        """
        tok = self._tokenizer
        if self.is_base:
            parts = []
            for m in messages:
                parts.append(f"{m['role'].capitalize()}: {m['content']}")
            parts.append("Assistant:")
            text = "\n".join(parts)
            return text + (" " + prefill if prefill else "")

        if prefill is None:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Continue an assistant turn that starts with `prefill`.
        msgs = list(messages) + [{"role": "assistant", "content": prefill}]
        try:
            return tok.apply_chat_template(
                msgs, tokenize=False, continue_final_message=True
            )
        except TypeError:
            # Older transformers without continue_final_message: build manually.
            base = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return base + prefill

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def _sample(self, prompt: str, n: int, **overrides) -> list[str]:
        self._ensure_loaded()
        temperature = overrides.get("temperature", config.GEN.temperature)
        top_p = overrides.get("top_p", config.GEN.top_p)
        max_new = overrides.get("max_new_tokens", config.GEN.max_new_tokens)

        if self._use_vllm:
            from vllm import SamplingParams

            params = SamplingParams(
                n=n, temperature=temperature, top_p=top_p, max_tokens=max_new
            )
            out = self._llm.generate([prompt], params, use_tqdm=False)
            return [o.text for o in out[0].outputs]

        import torch

        tok = self._tokenizer
        inputs = tok(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            gen = self._model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new,
                num_return_sequences=n,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return [tok.decode(g[prompt_len:], skip_special_tokens=True) for g in gen]

    def generate(self, messages: list[Message], n: int = 1, **overrides) -> list[GenResult]:
        prompt = self._render(messages)
        texts = self._sample(prompt, n, **overrides)
        return [GenResult(text=t.strip(), meta={"model": self.spec.name}) for t in texts]

    def continue_prefill(
        self, messages: list[Message], prefill: str, n: int = 1, **overrides
    ) -> list[GenResult]:
        prompt = self._render(messages, prefill=prefill)
        texts = self._sample(prompt, n, **overrides)
        # Returned text excludes the prefill (the sampler only emits new tokens).
        return [
            GenResult(text=t.strip(), meta={"model": self.spec.name, "prefill": prefill})
            for t in texts
        ]


def _vllm_available() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False
