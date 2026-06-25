"""Local Gemma inference via vLLM (with a transformers fallback).

Used for every open-weight model in the replication: Gemma-3 {12B,27B} in both
instruct (`-it`) and pretrained (`-pt`) form, plus the LoRA-fine-tuned 27B model
from Section 4.

Design notes
------------
* vLLM is the primary path because the eval samples thousands of rollouts at
  temperature 1; transformers single-stream generation would be impractically
  slow. The transformers path exists so the code can run (slowly) without vLLM.
* Prefilling (Section 3) is implemented by building the prompt string ourselves
  -- chat template + ``add_generation_prompt`` + the prefill text -- and asking
  the engine to continue from that raw string. For base/pretrained models we
  skip the chat template entirely and continue from the raw prefill.
* A LoRA adapter (the DPO / SFT fine-tune) can be attached via ``lora_path``.
"""

from __future__ import annotations

from typing import Sequence

from .base import Message, ModelBackend


class HFBackend(ModelBackend):
    def __init__(
        self,
        name: str,
        hf_id: str,
        is_chat: bool = True,
        lora_path: str | None = None,
        backend: str = "vllm",
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 16384,
    ):
        self.name = name
        self.hf_id = hf_id
        self.is_chat = is_chat
        self.lora_path = lora_path
        self.backend = backend
        self._engine = None
        self._tokenizer = None
        self._lora_request = None
        self._dtype = dtype
        self._tp = tensor_parallel_size
        self._gpu_util = gpu_memory_utilization
        self._max_len = max_model_len

    # ----------------------------------------------------------------- setup
    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        if self.backend == "vllm":
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_vllm(self) -> None:
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        self._engine = LLM(
            model=self.hf_id,
            dtype=self._dtype,
            tensor_parallel_size=self._tp,
            gpu_memory_utilization=self._gpu_util,
            max_model_len=self._max_len,
            enable_lora=self.lora_path is not None,
            max_lora_rank=64,  # matches Appendix E
        )
        if self.lora_path is not None:
            self._lora_request = LoRARequest("calm_adapter", 1, self.lora_path)

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        if self.lora_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.lora_path)
        self._engine = model

    # ------------------------------------------------------------ prompting
    def _render_prompt(self, messages: Sequence[Message], prefill: str | None) -> str:
        """Turn a message list into the exact string fed to the model."""
        if self.is_chat:
            text = self._tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
        else:
            # Base/pretrained model: no chat template. We linearise the
            # conversation as plain alternating text. In the Section 3 prefill
            # study, base models are only ever given a single user turn plus a
            # prefilled assistant turn, so this stays simple and faithful.
            parts = []
            for m in messages:
                parts.append(m["content"])
            text = "\n\n".join(parts) + "\n\n"
        if prefill:
            text = text + prefill
        return text

    # ----------------------------------------------------------- generation
    def _sample(self, prompt: str, n: int, temperature: float,
                max_new_tokens: int, top_p: float) -> list[str]:
        self._ensure_loaded()
        if self.backend == "vllm":
            from vllm import SamplingParams

            params = SamplingParams(
                n=n, temperature=temperature, top_p=top_p,
                max_tokens=max_new_tokens,
            )
            outputs = self._engine.generate(
                [prompt], params, lora_request=self._lora_request, use_tqdm=False
            )
            return [o.text for o in outputs[0].outputs]
        # transformers fallback (one sequence at a time)
        import torch

        tok = self._tokenizer(prompt, return_tensors="pt").to(self._engine.device)
        outs = []
        for _ in range(n):
            gen = self._engine.generate(
                **tok, do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                top_p=top_p, max_new_tokens=max_new_tokens,
            )
            new_tokens = gen[0][tok["input_ids"].shape[1]:]
            outs.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True))
        return outs

    def generate(self, messages, n=1, temperature=1.0, max_new_tokens=2048, top_p=1.0):
        prompt = self._render_prompt(messages, prefill=None)
        return self._sample(prompt, n, temperature, max_new_tokens, top_p)

    def generate_with_prefill(self, messages, prefill, n=1, temperature=1.0,
                              max_new_tokens=2048, top_p=1.0):
        prompt = self._render_prompt(messages, prefill=prefill)
        # The engine continues from `prompt`; the returned text is already the
        # continuation only (vLLM/transformers return generated tokens, not the
        # prompt), so the prefill is implicitly excluded -- exactly what the
        # Section 3 judge needs.
        return self._sample(prompt, n, temperature, max_new_tokens, top_p)

    def close(self):
        self._engine = None
