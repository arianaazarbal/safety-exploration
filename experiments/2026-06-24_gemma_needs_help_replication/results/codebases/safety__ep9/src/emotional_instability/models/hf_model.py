"""Local Gemma inference via vLLM.

Supports:
  * chat generation with the model's chat template,
  * **prefilling** an assistant turn (Section 3 "onset"/"early" continuations,
    Section 4 recovery experiment),
  * **raw** continuation for base/pretrained Gemma (no chat template),
  * serving a **LoRA adapter** (DPO/SFT checkpoints) without reloading weights.

vLLM is used because the evals need temperature-1 sampling over thousands of
prompts and n=50 continuations per prefill; its batched scheduler and `n`
sampling parameter make this tractable.
"""
from __future__ import annotations

from typing import Any

from .base import Conversation, ModelClient


class HFModelClient(ModelClient):
    supports_prefill = True
    supports_raw = True

    def __init__(self, name: str, hf_id: str, hf_cfg: dict[str, Any],
                 generation_cfg: dict[str, Any], lora_path: str | None = None):
        self.name = name
        self.hf_id = hf_id
        self.hf_cfg = hf_cfg
        self.gen_cfg = generation_cfg
        self.lora_path = lora_path
        self._llm = None
        self._tokenizer = None
        self._lora_request = None

    # -- lazy init so importing the module never requires a GPU -------------
    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer

        self._llm = LLM(
            model=self.hf_id,
            dtype=self.hf_cfg.get("dtype", "bfloat16"),
            tensor_parallel_size=self.hf_cfg.get("tensor_parallel_size", 1),
            gpu_memory_utilization=self.hf_cfg.get("gpu_memory_utilization", 0.9),
            max_model_len=self.hf_cfg.get("max_model_len", 8192),
            enforce_eager=self.hf_cfg.get("enforce_eager", False),
            enable_lora=self.hf_cfg.get("enable_lora", True),
            max_lora_rank=self.hf_cfg.get("max_lora_rank", 64),
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        if self.lora_path:
            self._lora_request = LoRARequest("adapter", 1, self.lora_path)

    def _sampling_params(self, n: int, temperature: float | None, max_new_tokens: int | None):
        from vllm import SamplingParams

        return SamplingParams(
            n=n,
            temperature=self.gen_cfg["temperature"] if temperature is None else temperature,
            top_p=self.gen_cfg.get("top_p", 1.0),
            max_tokens=max_new_tokens or self.gen_cfg.get("max_new_tokens", 1024),
        )

    # -- prompt rendering --------------------------------------------------
    def _render_chat(self, conv: Conversation, prefill: str | None) -> str:
        """Render a conversation to a prompt string. When `prefill` is given we
        append it after the generation header and let the model continue it
        (continue_final_message semantics)."""
        msgs = [m.as_dict() for m in conv]
        text = self._tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        if prefill:
            text = text + prefill
        return text

    # -- API ---------------------------------------------------------------
    def generate(self, conversations, *, n=1, prefill=None, temperature=None,
                 max_new_tokens=None) -> list[list[str]]:
        self._ensure_loaded()
        if prefill is not None and len(prefill) != len(conversations):
            raise ValueError("prefill must be one string per conversation")
        prompts = [
            self._render_chat(conv, prefill[i] if prefill else None)
            for i, conv in enumerate(conversations)
        ]
        sp = self._sampling_params(n, temperature, max_new_tokens)
        outputs = self._llm.generate(prompts, sp, lora_request=self._lora_request)
        return [[o.text for o in out.outputs] for out in outputs]

    def generate_raw(self, prompts, *, n=1, temperature=None, max_new_tokens=None):
        self._ensure_loaded()
        sp = self._sampling_params(n, temperature, max_new_tokens)
        outputs = self._llm.generate(list(prompts), sp, lora_request=self._lora_request)
        return [[o.text for o in out.outputs] for out in outputs]

    def num_tokens(self, text: str) -> int:
        self._ensure_loaded()
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def truncate_tokens(self, text: str, n_tokens: int) -> str:
        """Return the prefix of `text` consisting of its first `n_tokens` tokens."""
        self._ensure_loaded()
        ids = self._tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
        return self._tokenizer.decode(ids)
