"""Model backends and a registry mapping logical names to runnable clients.

Three backends:
  * VLLMBackend      - fast local sampling for Gemma chat models (bulk Section 2).
  * HFBackend        - transformers; supports (a) loading a LoRA adapter on top
                       of a base model and (b) *raw prefill continuation*, which
                       vLLM's chat API does not expose and which Section 3 needs.
  * OpenRouterBackend- API chat for Gemini targets, with thinking disabled.

Heavy deps (torch/vllm/transformers/anthropic/openai) are imported lazily inside
each backend so that pure-analysis code can `import config`/`src.models` cheaply.

Message format everywhere: list[{"role": "system"|"user"|"assistant", "content": str}].
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import config
from src.utils import retry

Messages = list[dict]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def merge_system_into_first_user(messages: Messages) -> Messages:
    """Gemma's chat template does not support a `system` role. We fold any
    system message into the first user turn (the convention HF uses internally)."""
    sys_chunks = [m["content"] for m in messages if m["role"] == "system"]
    if not sys_chunks:
        return messages
    rest = [m for m in messages if m["role"] != "system"]
    prefix = "\n\n".join(sys_chunks).strip()
    for i, m in enumerate(rest):
        if m["role"] == "user":
            rest = rest.copy()
            rest[i] = {"role": "user", "content": f"{prefix}\n\n{m['content']}"}
            return rest
    # No user turn: prepend as a standalone user message.
    return [{"role": "user", "content": prefix}, *rest]


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class Backend:
    """Interface. `generate` returns, for each conversation, `n` sampled
    completions (assistant text only)."""

    def generate(self, conversations: Sequence[Messages], *, temperature: float,
                 max_tokens: int, n: int = 1) -> list[list[str]]:
        raise NotImplementedError

    def continue_text(self, prompts: Sequence[str], *, temperature: float,
                      max_tokens: int, n: int = 1) -> list[list[str]]:
        """Raw text continuation (no chat template applied). Used for prefill."""
        raise NotImplementedError


class VLLMBackend(Backend):
    def __init__(self, spec: config.ModelSpec, **kw):
        from vllm import LLM  # lazy

        self.spec = spec
        self.llm = LLM(
            model=spec.model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            tensor_parallel_size=int(os.environ.get("TP_SIZE", "1")),
            gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.90")),
            max_model_len=int(os.environ.get("MAX_MODEL_LEN", "8192")),
            **kw,
        )

    def _sp(self, temperature, max_tokens, n):
        from vllm import SamplingParams

        return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              n=n, top_p=1.0, seed=None)

    def generate(self, conversations, *, temperature, max_tokens, n=1):
        convs = [merge_system_into_first_user(c) for c in conversations]
        outs = self.llm.chat(convs, self._sp(temperature, max_tokens, n))
        return [[o.text for o in out.outputs] for out in outs]

    def continue_text(self, prompts, *, temperature, max_tokens, n=1):
        outs = self.llm.generate(list(prompts), self._sp(temperature, max_tokens, n))
        return [[o.text for o in out.outputs] for out in outs]


class HFBackend(Backend):
    """transformers backend. Loads an optional LoRA adapter (PEFT). Slower than
    vLLM; used for base-model prefill (Section 3) and serving finetuned adapters
    when vLLM adapter-hot-swap is inconvenient."""

    def __init__(self, spec: config.ModelSpec, load_in_4bit: bool | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.torch = torch
        load_in_4bit = (os.environ.get("LOAD_IN_4BIT", "0") == "1"
                        if load_in_4bit is None else load_in_4bit)
        quant = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self.tok = AutoTokenizer.from_pretrained(spec.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, **quant,
        )
        if spec.lora_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, spec.lora_path)
        self.model.eval()

    def _gen(self, input_ids, attn, temperature, max_tokens, n):
        out = self.model.generate(
            input_ids=input_ids, attention_mask=attn,
            do_sample=temperature > 0, temperature=max(temperature, 1e-5),
            top_p=1.0, max_new_tokens=max_tokens, num_return_sequences=n,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        gen = out[:, input_ids.shape[1]:]
        return [self.tok.decode(g, skip_special_tokens=True) for g in gen]

    def generate(self, conversations, *, temperature, max_tokens, n=1):
        results = []
        for conv in conversations:
            conv = merge_system_into_first_user(conv)
            ids = self.tok.apply_chat_template(
                conv, add_generation_prompt=True, return_tensors="pt",
            ).to(self.model.device)
            attn = self.torch.ones_like(ids)
            results.append(self._gen(ids, attn, temperature, max_tokens, n))
        return results

    def continue_text(self, prompts, *, temperature, max_tokens, n=1):
        # add_special_tokens=False: callers pass already-rendered prompts that
        # include BOS/turn markers; re-adding BOS here would corrupt them.
        results = []
        for p in prompts:
            enc = self.tok(p, return_tensors="pt", add_special_tokens=False
                           ).to(self.model.device)
            results.append(self._gen(enc.input_ids, enc.attention_mask,
                                     temperature, max_tokens, n))
        return results

    def build_prefill_prompt(self, conversation: Messages, prefill: str) -> str:
        """Render a chat conversation and append `prefill` so the model continues
        the assistant turn from that text (used by Section 3)."""
        conv = merge_system_into_first_user(conversation)
        rendered = self.tok.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False,
        )
        return rendered + prefill


class OpenRouterBackend(Backend):
    """OpenAI-compatible client pointed at OpenRouter; used for Gemini targets.
    Thinking/reasoning is disabled where the route supports it (paper sets
    thinking=false; Gemini-2.5-Pro may still emit hidden reasoning)."""

    def __init__(self, spec: config.ModelSpec):
        from openai import OpenAI

        self.spec = spec
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def _one(self, conv, temperature, max_tokens):
        def call():
            return self.client.chat.completions.create(
                model=self.spec.model_id,
                messages=conv,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"reasoning": {"enabled": False}},  # disable thinking
            )
        resp = retry(call)
        return resp.choices[0].message.content or ""

    def generate(self, conversations, *, temperature, max_tokens, n=1):
        # Gemini accepts a system role; pass through unchanged.
        return [[self._one(conv, temperature, max_tokens) for _ in range(n)]
                for conv in conversations]


# --------------------------------------------------------------------------- #
# Registry (lazy, cached: one heavy model loaded at a time per process)
# --------------------------------------------------------------------------- #
_BACKEND_CLASSES = {
    "vllm": VLLMBackend, "hf": HFBackend, "openrouter": OpenRouterBackend,
}


@lru_cache(maxsize=4)
def get_backend(model_name: str) -> Backend:
    spec = config.ALL_MODELS[model_name]
    return _BACKEND_CLASSES[spec.backend](spec)
