"""Local HuggingFace client for Gemma models.

Two backends:
  * vLLM (preferred) for fast high-throughput sampling at temperature 1.
  * transformers for prefill continuation, hidden-state extraction (probing),
    and as a fallback when vLLM is unavailable.

A single instance can optionally load a PEFT/LoRA adapter on top of the base
weights, which is how finetuned (DPO/SFT) checkpoints are evaluated.

Gemma's chat template does not accept a standalone `system` role; we fold any
system message into the first user turn (documented in DESIGN.md).
"""
from __future__ import annotations

from typing import List, Optional

from .base import ChatModel, GenConfig, Message


def _fold_system(messages: List[Message]) -> List[Message]:
    """Gemma has no system role: prepend system text to the first user turn."""
    if not messages or messages[0]["role"] != "system":
        return list(messages)
    system = messages[0]["content"]
    rest = list(messages[1:])
    for i, m in enumerate(rest):
        if m["role"] == "user":
            rest[i] = {"role": "user", "content": f"{system}\n\n{m['content']}"}
            return rest
    # no user turn: emit system as a user turn
    return [{"role": "user", "content": system}] + rest


class HFLocalModel(ChatModel):
    supports_prefill = True
    supports_hidden_states = True

    def __init__(self, name: str, hf_id: str, family: str = "gemma",
                 role: str = "instruct", adapter_path: Optional[str] = None,
                 prefer_vllm: bool = True, dtype: str = "bfloat16",
                 tensor_parallel_size: int = 1):
        self.name = name
        self.hf_id = hf_id
        self.family = family
        self.role = role
        self.adapter_path = adapter_path
        self.prefer_vllm = prefer_vllm and adapter_path is None  # see _init_vllm note
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size

        self._vllm = None
        self._hf_model = None
        self._tokenizer = None

    # --- tokenizer (always via transformers, used for templating) ---
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        return self._tokenizer

    def _render_prompt(self, messages: List[Message], add_generation_prompt=True) -> str:
        msgs = _fold_system(messages) if self.family == "gemma" else list(messages)
        if self.role == "base":
            # Base models are not chat-tuned: render a plain transcript so the
            # prefill experiment continues from a consistent surface form.
            parts = []
            for m in msgs:
                tag = "User" if m["role"] == "user" else "Assistant"
                parts.append(f"{tag}: {m['content']}")
            parts.append("Assistant:")
            return "\n".join(parts)
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    # --- vLLM backend ---
    def _ensure_vllm(self):
        if self._vllm is None:
            from vllm import LLM

            self._vllm = LLM(
                model=self.hf_id,
                dtype=self.dtype,
                tensor_parallel_size=self.tensor_parallel_size,
                enable_prefix_caching=True,
            )
        return self._vllm

    def _vllm_available(self) -> bool:
        if not self.prefer_vllm:
            return False
        try:
            import vllm  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    # --- transformers backend ---
    def _ensure_hf(self):
        if self._hf_model is None:
            import torch
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                self.hf_id,
                torch_dtype=getattr(torch, self.dtype),
                device_map="auto",
            )
            if self.adapter_path:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, self.adapter_path)
            model.eval()
            self._hf_model = model
        return self._hf_model

    # --- generation ---
    def generate(self, messages: List[Message], cfg: GenConfig) -> str:
        return self.generate_batch([messages], cfg)[0]

    def generate_batch(self, batch: List[List[Message]], cfg: GenConfig) -> List[str]:
        prompts = [self._render_prompt(m) for m in batch]
        if self._vllm_available():
            return self._vllm_generate(prompts, cfg)
        return self._hf_generate(prompts, cfg)

    def _vllm_generate(self, prompts: List[str], cfg: GenConfig) -> List[str]:
        from vllm import SamplingParams

        sp = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            stop=list(cfg.stop) if cfg.stop else None,
        )
        outs = self._ensure_vllm().generate(prompts, sp)
        return [o.outputs[0].text for o in outs]

    def _hf_generate(self, prompts: List[str], cfg: GenConfig) -> List[str]:
        import torch

        model = self._ensure_hf()
        tok = self.tokenizer
        results: List[str] = []
        for prompt in prompts:
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    do_sample=cfg.temperature > 0,
                    temperature=max(cfg.temperature, 1e-5),
                    top_p=cfg.top_p,
                    max_new_tokens=cfg.max_new_tokens,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            gen = out[0][inputs["input_ids"].shape[1]:]
            results.append(tok.decode(gen, skip_special_tokens=True))
        return results

    # --- prefill continuation (Section 3) ---
    def generate_with_prefill(self, messages: List[Message], prefill: str,
                              cfg: GenConfig) -> str:
        """Continue an assistant turn beginning with `prefill`; return the
        continuation only."""
        prompt = self._render_prompt(messages, add_generation_prompt=True) + prefill
        if self._vllm_available():
            return self._vllm_generate([prompt], cfg)[0]
        return self._hf_generate([prompt], cfg)[0]

    def prefill_batch(self, items: List[tuple], cfg: GenConfig) -> List[str]:
        """Batched prefill. `items` is a list of (messages, prefill)."""
        prompts = [self._render_prompt(m, True) + p for m, p in items]
        if self._vllm_available():
            return self._vllm_generate(prompts, cfg)
        return self._hf_generate(prompts, cfg)

    # --- hidden states (Appendix I probing) ---
    def hidden_states(self, text: str):
        """Return per-layer hidden states for a raw text string.

        Returns (hidden_states, input_ids): hidden_states is a tuple of
        (num_layers+1) tensors of shape [seq, d_model]. Always uses transformers.
        """
        import torch

        model = self._ensure_hf()
        tok = self.tokenizer
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = tuple(h[0].float().cpu() for h in out.hidden_states)
        return hs, inputs["input_ids"][0].cpu()

    @property
    def lm_head_weight(self):
        """Unembedding matrix [vocab, d_model] for logit-lens probing."""
        import torch

        model = self._ensure_hf()
        base = getattr(model, "base_model", model)
        head = model.get_output_embeddings()
        return head.weight.detach().float().cpu() if head is not None else None
