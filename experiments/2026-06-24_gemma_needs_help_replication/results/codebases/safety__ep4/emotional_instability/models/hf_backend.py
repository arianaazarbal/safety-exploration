"""Local Gemma inference.

Preferred path is vLLM (high throughput needed for thousands of temp-1 samples).
Falls back to plain HuggingFace transformers if vLLM is unavailable. Both paths:
  - apply the Gemma chat template,
  - support prefilling an assistant turn (`continue_final_message`) for the
    Section-3 base-vs-instruct continuation experiment,
  - support loading a LoRA adapter (for evaluating the DPO/SFT fine-tunes).

Gemma base (-pt) models have no chat template; we render a minimal turn format
and rely on prefilling so they continue an assistant response, exactly as the
paper does ("base models are not trained on chat-formatted prompts ... we
prefill the first parts of the model responses").
"""

from __future__ import annotations

from typing import Optional

from .base import ChatMessage, ChatModel


# Minimal Gemma-style turn format used for base (-pt) models that lack a chat
# template. Mirrors Gemma's <start_of_turn>/<end_of_turn> structure.
_BASE_TEMPLATE_TURN = "<start_of_turn>{role}\n{content}<end_of_turn>\n"


def _render_base_prompt(messages: list[ChatMessage], system: Optional[str]) -> str:
    parts = []
    if system:
        parts.append(_BASE_TEMPLATE_TURN.format(role="user", content=system))
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        parts.append(_BASE_TEMPLATE_TURN.format(role=role, content=m["content"]))
    parts.append("<start_of_turn>model\n")
    return "".join(parts)


class GemmaHFModel(ChatModel):
    supports_prefill = True

    def __init__(self, spec, *, adapter_path: Optional[str] = None,
                 dtype: str = "bfloat16", use_vllm: bool = True,
                 max_model_len: int = 16384, gpu_memory_utilization: float = 0.9):
        self.spec = spec
        self.name = spec.name
        self.model_id = spec.model_id
        self.is_base = spec.is_base
        self.adapter_path = adapter_path
        self._backend = None

        if use_vllm:
            try:
                self._init_vllm(dtype, max_model_len, gpu_memory_utilization,
                                adapter_path)
                self._backend = "vllm"
            except Exception as e:  # pragma: no cover - environment dependent
                print(f"[hf_backend] vLLM unavailable ({e}); using transformers.")
        if self._backend is None:
            self._init_transformers(dtype, adapter_path)
            self._backend = "transformers"

    # ------------------------------------------------------------------ #
    # init
    # ------------------------------------------------------------------ #
    def _init_vllm(self, dtype, max_model_len, gpu_util, adapter_path):
        from vllm import LLM, SamplingParams  # noqa
        from transformers import AutoTokenizer

        self._SamplingParams = SamplingParams
        enable_lora = adapter_path is not None
        self.llm = LLM(model=self.model_id, dtype=dtype, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_util, enable_lora=enable_lora,
                       max_lora_rank=64)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._lora_request = None
        if adapter_path:
            from vllm.lora.request import LoRARequest
            self._lora_request = LoRARequest("adapter", 1, adapter_path)

    def _init_transformers(self, dtype, adapter_path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch_dtype, device_map="auto")
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # prompt rendering
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[ChatMessage], system: Optional[str],
                prefill: Optional[str]) -> str:
        if self.is_base:
            text = _render_base_prompt(messages, system)
            return text + (prefill or "")

        msgs = list(messages)
        if system:
            # Gemma has no separate system role; prepend to first user turn.
            if msgs and msgs[0]["role"] == "user":
                msgs = [{"role": "user", "content": f"{system}\n\n{msgs[0]['content']}"}] \
                    + msgs[1:]
            else:
                msgs = [{"role": "user", "content": system}] + msgs

        if prefill is not None:
            msgs = msgs + [{"role": "assistant", "content": prefill}]
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, continue_final_message=True)
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #
    def _vllm_generate(self, prompts: list[str], temperature, top_p,
                       max_new_tokens) -> list[str]:
        sp = self._SamplingParams(temperature=temperature, top_p=top_p,
                                  max_tokens=max_new_tokens)
        kwargs = {}
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        outs = self.llm.generate(prompts, sp, **kwargs)
        return [o.outputs[0].text for o in outs]

    def _transformers_generate(self, prompt: str, temperature, top_p,
                               max_new_tokens) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, do_sample=temperature > 0, temperature=temperature,
                top_p=top_p, max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def generate(self, messages, *, temperature=1.0, top_p=1.0,
                 max_new_tokens=2048, system=None) -> str:
        return self.generate_batch([messages], temperature=temperature, top_p=top_p,
                                   max_new_tokens=max_new_tokens, system=system)[0]

    def generate_batch(self, batch, *, temperature=1.0, top_p=1.0,
                       max_new_tokens=2048, system=None) -> list[str]:
        prompts = [self._render(m, system, None) for m in batch]
        if self._backend == "vllm":
            return self._vllm_generate(prompts, temperature, top_p, max_new_tokens)
        return [self._transformers_generate(p, temperature, top_p, max_new_tokens)
                for p in prompts]

    def continue_prefill(self, messages, prefill, *, temperature=1.0, top_p=1.0,
                         max_new_tokens=2048, system=None) -> str:
        prompt = self._render(messages, system, prefill)
        if self._backend == "vllm":
            return self._vllm_generate([prompt], temperature, top_p, max_new_tokens)[0]
        return self._transformers_generate(prompt, temperature, top_p, max_new_tokens)
