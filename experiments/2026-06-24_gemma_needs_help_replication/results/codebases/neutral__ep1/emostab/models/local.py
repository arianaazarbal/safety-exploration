"""Local Gemma backends: vLLM (fast sampling) and HuggingFace transformers
(needed for LoRA adapters, prefill, and hidden-state access)."""
from __future__ import annotations

import os
from typing import List

from ..config import CACHE_DIR, ModelSpec
from .base import ChatModel, Conversation


# --------------------------------------------------------------------------- #
# Chat templating helpers (Gemma 3)
# --------------------------------------------------------------------------- #
def _normalise_for_gemma(conversation: Conversation) -> Conversation:
    """Gemma 3 chat templates have no dedicated system role; fold any system
    message into the first user turn so templating never raises."""
    msgs = [dict(m) for m in conversation]
    if msgs and msgs[0]["role"] == "system":
        sys = msgs.pop(0)["content"]
        for m in msgs:
            if m["role"] == "user":
                m["content"] = f"{sys}\n\n{m['content']}"
                break
        else:  # no user turn yet
            msgs.insert(0, {"role": "user", "content": sys})
    return msgs


def _format_base_prompt(conversation: Conversation) -> str:
    """Plain-text rendering of a conversation for *base* (pretrained) models,
    mirroring Gemma's turn structure without relying on a chat template.

    Used by the Section 3 prefill experiment: base models simply continue from
    a prefilled assistant turn (see DESIGN.md)."""
    conversation = _normalise_for_gemma(conversation)
    parts = []
    for m in conversation:
        tag = "user" if m["role"] == "user" else "model"
        parts.append(f"<start_of_turn>{tag}\n{m['content']}<end_of_turn>")
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


def _build_prompt(tokenizer, spec: ModelSpec, conversation: Conversation,
                  prefill: str | None) -> str:
    """Render a conversation (+ optional assistant prefill) to a prompt string."""
    if spec.is_base:
        text = _format_base_prompt(conversation)
        if prefill:
            text += prefill
        return text

    msgs = _normalise_for_gemma(conversation)
    if prefill:
        # Continue a partially-written assistant turn.
        msgs = msgs + [{"role": "assistant", "content": prefill}]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
            continue_final_message=True,
        )
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )


# --------------------------------------------------------------------------- #
# vLLM backend
# --------------------------------------------------------------------------- #
class VLLMModel(ChatModel):
    """Fast batched sampling for the main eval. Supports assistant prefill."""

    def __init__(self, spec: ModelSpec, *, tensor_parallel_size: int | None = None,
                 dtype: str = "bfloat16", gpu_memory_utilization: float = 0.90,
                 max_model_len: int | None = 16384):
        super().__init__(spec)
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        tp = tensor_parallel_size or int(os.environ.get("EMOSTAB_TP", "1"))
        self.llm = LLM(
            model=spec.model_id,
            dtype=dtype,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            download_dir=str(CACHE_DIR),
            enable_lora=spec.adapter_path is not None,
        )
        self._lora_request = None
        if spec.adapter_path:
            from vllm.lora.request import LoRARequest
            self._lora_request = LoRARequest("adapter", 1, spec.adapter_path)

    def generate_batch(self, conversations, *, temperature=1.0, max_tokens=4096,
                       prefills=None, seed=None) -> List[str]:
        from vllm import SamplingParams

        if prefills is None:
            prefills = [None] * len(conversations)
        prompts = [
            _build_prompt(self.tokenizer, self.spec, conv, pf)
            for conv, pf in zip(conversations, prefills)
        ]
        params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, seed=seed, top_p=1.0,
        )
        outs = self.llm.generate(
            prompts, params, lora_request=self._lora_request, use_tqdm=False,
        )
        return [o.outputs[0].text for o in outs]


# --------------------------------------------------------------------------- #
# HuggingFace transformers backend
# --------------------------------------------------------------------------- #
class HFModel(ChatModel):
    """transformers backend. Used where vLLM is awkward: running LoRA adapters,
    prefill continuations from base models, and exposing the underlying module
    for hidden-state probing (Appendix I)."""

    def __init__(self, spec: ModelSpec, *, dtype: str = "bfloat16",
                 device_map: str = "auto", load_in_4bit: bool = False,
                 output_hidden_states: bool = False):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        torch_dtype = getattr(torch, dtype)
        kw = dict(torch_dtype=torch_dtype, device_map=device_map,
                  cache_dir=str(CACHE_DIR))
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4")
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kw)
        if spec.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, spec.adapter_path)
            self.model = self.model.merge_and_unload()  # fold LoRA for inference
        self.model.eval()
        self._output_hidden_states = output_hidden_states

    def generate_batch(self, conversations, *, temperature=1.0, max_tokens=4096,
                       prefills=None, seed=None) -> List[str]:
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        if prefills is None:
            prefills = [None] * len(conversations)

        outputs: List[str] = []
        # transformers batched generation with left-padding; kept simple/robust
        # by processing one conversation at a time (these runs are small for HF).
        for conv, pf in zip(conversations, prefills):
            prompt = _build_prompt(self.tokenizer, self.spec, conv, pf)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            gen = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                top_p=1.0,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            new_tokens = gen[0, inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(text)
        return outputs
