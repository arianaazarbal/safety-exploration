"""vLLM backend for local Gemma models (preferred for throughput at scale).

Supports both chat (applies the Gemma chat template) and raw completion (for
base-model prefilling in Section 3). A single vLLM engine is loaded per process;
batching is handled by passing the whole batch to `llm.generate`.

Optionally loads a LoRA adapter (trained DPO/SFT model) via vLLM's LoRA support.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from .base import ChatClient, CompletionClient, GenConfig, GenResult, Message, fold_system

logger = logging.getLogger("eilm.vllm")


class VLLMModel(ChatClient, CompletionClient):
    def __init__(
        self,
        hf_id: str,
        name: str,
        family: str = "gemma",
        role: str = "instruct",
        lora_path: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "bfloat16",
    ):
        from vllm import LLM
        from transformers import AutoTokenizer

        self.hf_id = hf_id
        self.name = name
        self.family = family
        self.role = role
        self._lora_path = lora_path

        if tensor_parallel_size is None:
            tensor_parallel_size = max(1, _visible_gpu_count())

        self._tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self._lora_request = None
        enable_lora = lora_path is not None
        self._llm = LLM(
            model=hf_id,
            tokenizer=hf_id,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            enable_lora=enable_lora,
            max_lora_rank=64,
            trust_remote_code=True,
        )
        if enable_lora:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("adapter", 1, lora_path)

    # --- helpers -----------------------------------------------------------
    def _sampling_params(self, cfg: GenConfig, seed=None):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            seed=cfg.seed if seed is None else seed,
            stop=cfg.stop,
        )

    def _render_chat(self, messages: List[Message]) -> str:
        # Gemma 3 lacks a system role; fold any system message into the first
        # user turn before applying the chat template.
        if self.family == "gemma":
            messages = fold_system(messages)
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate(self, prompts: List[str], cfg: GenConfig) -> List[GenResult]:
        # Per-prompt sampling params when per-item seeds are supplied, so that
        # identical prompts in a batch still produce diverse samples.
        if cfg.seeds is not None and len(cfg.seeds) == len(prompts):
            sp = [self._sampling_params(cfg, seed=s) for s in cfg.seeds]
        else:
            sp = self._sampling_params(cfg)
        outs = self._llm.generate(
            prompts,
            sp,
            lora_request=self._lora_request,
            use_tqdm=False,
        )
        # vLLM may reorder; map back by request_id order (generate preserves input order).
        results = []
        for o in outs:
            comp = o.outputs[0]
            results.append(
                GenResult(
                    text=comp.text,
                    finish_reason=comp.finish_reason or "stop",
                    usage={
                        "prompt_tokens": len(o.prompt_token_ids),
                        "completion_tokens": len(comp.token_ids),
                    },
                )
            )
        return results

    # --- ChatClient --------------------------------------------------------
    def chat(self, messages: List[Message], cfg: GenConfig) -> GenResult:
        return self.chat_batch([messages], cfg)[0]

    def chat_batch(self, batch: List[List[Message]], cfg: GenConfig) -> List[GenResult]:
        prompts = [self._render_chat(m) for m in batch]
        return self._generate(prompts, cfg)

    # --- CompletionClient (prefill) ----------------------------------------
    def complete(self, prompt_text: str, cfg: GenConfig) -> GenResult:
        return self.complete_batch([prompt_text], cfg)[0]

    def complete_batch(self, prompts: List[str], cfg: GenConfig) -> List[GenResult]:
        return self._generate(prompts, cfg)

    # --- prefill template helper -------------------------------------------
    def render_chat_prefix(self, messages: List[Message], prefill: str) -> str:
        """Build a chat prompt whose assistant turn is *prefilled* with `prefill`,
        so the model continues from that text. Used by the prefill experiment."""
        base = self._render_chat(messages)
        return base + prefill

    @property
    def tokenizer(self):
        return self._tokenizer


def _visible_gpu_count() -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return len([x for x in cvd.split(",") if x.strip() != ""])
    try:
        import torch

        return max(1, torch.cuda.device_count())
    except Exception:
        return 1
