"""Inference backends and a registry that maps model names to backends.

Two backends:
  HFBackend          local HuggingFace weights. Supports chat generation,
                     prefilled-assistant continuation (Section 3), raw hidden
                     states / per-layer logit-lens (Appendix I), and serving
                     LoRA-adapted checkpoints (Section 4).
  OpenRouterBackend  remote chat API (Gemini targets + Claude/GPT infrastructure).
                     Chat only: no prefill, no logits, no training. `thinking`
                     is disabled per the paper.

Backends are lazily instantiated and cached: loading a 27B model is expensive,
so we keep at most a small set resident. Callers obtain backends through
`get_backend(spec)`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import ModelSpec, get_config

# A chat message is a {"role": ..., "content": ...} dict; roles are
# "system" | "user" | "assistant". We alias for readability.
Message = dict[str, str]
Conversation = list[Message]


@dataclass
class GenConfig:
    temperature: float = 1.0
    max_new_tokens: int = 2048
    thinking: bool = False
    top_p: float = 1.0
    stop: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------
class ModelBackend:
    spec: ModelSpec

    def generate(self, messages: Conversation, gen: GenConfig) -> str:
        raise NotImplementedError

    def generate_with_prefill(self, messages: Conversation, prefill: str,
                              gen: GenConfig) -> str:
        """Continue an assistant turn whose first tokens are `prefill`.

        Returns ONLY the newly generated continuation (excluding `prefill`),
        matching the paper's "model-generated continuation, excluding the
        prefilled text, is scored".
        """
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def truncate_to_tokens(self, text: str, n: int) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Local HuggingFace backend
# ---------------------------------------------------------------------------
class HFBackend(ModelBackend):
    def __init__(self, spec: ModelSpec, load_in_4bit: bool = False,
                 adapter_dir: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        quant = {}
        if load_in_4bit or os.environ.get("GNH_LOAD_4BIT") == "1":
            from transformers import BitsAndBytesConfig
            quant["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto", **quant,
        )
        # Serve a trained LoRA adapter on top of the base weights when present.
        adapter_dir = adapter_dir or spec.extra.get("adapter_dir")
        if adapter_dir and os.path.isdir(adapter_dir):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.eval()

    # -- helpers -----------------------------------------------------------
    def _apply_chat_template(self, messages: Conversation,
                             add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        )

    def _sample(self, prompt_text: str, gen: GenConfig) -> str:
        torch = self.torch
        inputs = self.tokenizer(prompt_text, return_tensors="pt",
                                add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=gen.temperature > 0,
                temperature=gen.temperature,
                top_p=gen.top_p,
                max_new_tokens=gen.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # -- interface ---------------------------------------------------------
    def generate(self, messages: Conversation, gen: GenConfig) -> str:
        if self.spec.is_instruct:
            prompt = self._apply_chat_template(messages, add_generation_prompt=True)
        else:
            # Base/pretrained models are not chat-tuned; fall back to a plain
            # concatenation. In practice base models are only ever called via
            # generate_with_prefill (Section 3), so this path is rarely used.
            prompt = "\n".join(m["content"] for m in messages) + "\n"
        return self._sample(prompt, gen)

    def generate_with_prefill(self, messages: Conversation, prefill: str,
                              gen: GenConfig) -> str:
        """Build a prompt that ends *inside* an assistant turn equal to `prefill`.

        For instruct models we render the chat template with a generation prompt
        and then append `prefill` so the model continues it. For base models we
        present the conversation as plain text (the paper prefills base models so
        they "consistently continue the model response").
        """
        if self.spec.is_instruct:
            base = self._apply_chat_template(messages, add_generation_prompt=True)
            prompt = base + prefill
        else:
            rendered = []
            for m in messages:
                rendered.append(f"{m['content']}")
            prompt = "\n".join(rendered) + "\n" + prefill
        return self._sample(prompt, gen)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_to_tokens(self, text: str, n: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    # -- logit lens (Appendix I) ------------------------------------------
    def residual_logits(self, messages: Conversation, response: str,
                        layers: list[int]) -> dict[str, Any]:
        """Run the model over (conversation + response) and return, per requested
        layer, the unembedded logits at every response-token position.

        Returns {"layers": layers, "token_ids": [...], "logits": tensor[L, T, V]}
        where positions correspond to the assistant `response` tokens only.
        Used by internal_emotion.py to compute Ekman-emotion z-scores.
        """
        torch = self.torch
        full = self._apply_chat_template(messages, add_generation_prompt=True) + response
        enc = self.tokenizer(full, return_tensors="pt",
                             add_special_tokens=False).to(self.model.device)
        prompt_len = self.count_tokens(
            self._apply_chat_template(messages, add_generation_prompt=True))
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        hidden = out.hidden_states  # tuple(len = num_layers+1) of [1, T, d]
        # Final unembedding (lm_head) + final RMSNorm, resolved robustly so this
        # works whether self.model is a bare CausalLM or a PeftModel wrapping one.
        lm_head = self.model.get_output_embeddings()
        norm = self._find_final_norm()
        resp_slice = slice(prompt_len, enc["input_ids"].shape[1])
        per_layer = []
        for L in layers:
            h = hidden[L][:, resp_slice, :]
            if norm is not None:
                h = norm(h)
            logits = lm_head(h)[0]  # [T_resp, V]
            per_layer.append(logits.float().cpu())
        return {
            "layers": layers,
            "token_ids": enc["input_ids"][0, resp_slice].cpu().tolist(),
            "logits": torch.stack(per_layer, dim=0),  # [L, T_resp, V]
        }

    def _find_final_norm(self):
        """Locate the decoder's final RMSNorm regardless of PEFT wrapping.

        Layout is `<...>.model.norm` for Gemma; a PeftModel inserts an extra
        `.base_model.model` (or `.model`) level, so we walk down `.model`
        attributes until we find a `norm`.
        """
        node = self.model
        for _ in range(4):
            norm = getattr(node, "norm", None)
            if norm is not None:
                return norm
            inner = getattr(node, "model", None) or getattr(node, "base_model", None)
            if inner is None or inner is node:
                break
            node = inner
        return None


# ---------------------------------------------------------------------------
# OpenRouter API backend
# ---------------------------------------------------------------------------
class OpenRouterBackend(ModelBackend):
    """Chat-completions via an OpenAI-compatible endpoint (OpenRouter by default).

    Used for Gemini targets and all infrastructure judges/auditors. Reasoning is
    disabled (`thinking=false`); the paper notes Gemini-2.5-Pro / GPT-5.2 may
    still emit hidden reasoning the API cannot suppress.
    """

    def __init__(self, spec: ModelSpec):
        from openai import OpenAI

        self.spec = spec
        base_url = os.environ.get("GNH_API_BASE", "https://openrouter.ai/api/v1")
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GNH_API_KEY")
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=2, max=30))
    def generate(self, messages: Conversation, gen: GenConfig) -> str:
        extra: dict[str, Any] = {}
        if not gen.thinking:
            # OpenRouter passes provider-specific reasoning controls through here.
            extra["extra_body"] = {"reasoning": {"enabled": False}}
        resp = self.client.chat.completions.create(
            model=self.spec.api_id,
            messages=messages,
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_new_tokens,
            stop=gen.stop,
            **extra,
        )
        return resp.choices[0].message.content or ""

    def generate_with_prefill(self, messages, prefill, gen):
        raise NotImplementedError(
            f"{self.spec.name} is API-only; prefilled continuation (Section 3) "
            "is not supported. Prefill experiments are Gemma-only.")

    def count_tokens(self, text: str) -> int:
        # Approximate; only HF backends need exact token truncation.
        return max(1, len(text) // 4)

    def truncate_to_tokens(self, text: str, n: int) -> str:
        return text[: n * 4]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_BACKENDS: dict[str, ModelBackend] = {}


def get_backend(spec: ModelSpec, **kwargs) -> ModelBackend:
    if spec.name in _BACKENDS:
        return _BACKENDS[spec.name]
    if spec.backend == "hf":
        backend: ModelBackend = HFBackend(spec, **kwargs)
    elif spec.backend == "openrouter":
        backend = OpenRouterBackend(spec)
    else:
        raise ValueError(f"Unknown backend '{spec.backend}' for {spec.name}")
    _BACKENDS[spec.name] = backend
    return backend


def get_backend_by_name(name: str, **kwargs) -> ModelBackend:
    return get_backend(get_config().model(name), **kwargs)


def unload(name: str) -> None:
    """Free a resident backend (useful between 27B experiments)."""
    b = _BACKENDS.pop(name, None)
    if b is not None and isinstance(b, HFBackend):
        import gc
        del b
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
