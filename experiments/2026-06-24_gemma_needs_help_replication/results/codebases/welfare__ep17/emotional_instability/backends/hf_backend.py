"""Local HuggingFace backend for Gemma.

Used wherever we need weights/logits, not just text: multi-turn eval, base-model
prefilling (§3), and internal-emotion probing (Appendix I). Base ("pt") models
have no chat template, so we render conversations into Gemma's chat format
ourselves and rely on prefilling to keep them on-distribution (paper §3.1).
"""

from __future__ import annotations

import threading
from functools import lru_cache

from ..config import Config, ModelSpec
from .base import GenConfig, Message

# Gemma chat-format tokens (used to hand-render conversations for BASE models,
# which lack an apply_chat_template). Instruct models use the tokenizer's own
# template. See DESIGN.md "base-model prompting".
_GEMMA_BOS = "<bos>"
_TURN_START = "<start_of_turn>"
_TURN_END = "<end_of_turn>"


class HFBackend:
    _instances: dict[str, "HFBackend"] = {}
    _lock = threading.Lock()

    def __init__(self, spec: ModelSpec, cfg: Config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.spec_name = spec.name
        self.cfg = cfg
        self.supports_prefill = True
        self.is_base = spec.kind == "base"

        self.tokenizer = AutoTokenizer.from_pretrained(spec.ident)
        load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained(spec.ident, **load_kwargs)
        # Apply a LoRA/PEFT adapter on top of the base weights (the DPO/SFT
        # models are stored as adapters over gemma-3-27b-it).
        if spec.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, spec.adapter_path)
        self.model.eval()
        self._torch = torch

    # -- singleton-per-model so we don't reload the 27B twice in one process ---
    @classmethod
    def shared(cls, spec: ModelSpec, cfg: Config) -> "HFBackend":
        with cls._lock:
            if spec.name not in cls._instances:
                cls._instances[spec.name] = cls(spec, cfg)
            return cls._instances[spec.name]

    # -- prompt rendering ------------------------------------------------------
    def render(self, messages: list[Message], add_generation_prompt: bool = True) -> str:
        """Render a conversation to a single prompt string.

        Instruct models: tokenizer chat template. Base models: hand-rolled Gemma
        turn format so prefilling continues coherently (paper §3.1).
        """
        if not self.is_base and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        # base model: mimic the Gemma instruct turn structure
        parts = [_GEMMA_BOS]
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            parts.append(f"{_TURN_START}{role}\n{m['content']}{_TURN_END}\n")
        if add_generation_prompt:
            parts.append(f"{_TURN_START}model\n")
        return "".join(parts)

    # -- generation ------------------------------------------------------------
    def _generate(self, prompt_text: str, gen: GenConfig) -> str:
        torch = self._torch
        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        do_sample = gen.temperature and gen.temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=gen.max_new_tokens,
                do_sample=do_sample,
                temperature=gen.temperature if do_sample else None,
                top_p=gen.top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()

    def chat(self, messages: list[Message], gen: GenConfig) -> str:
        prompt = self.render(messages, add_generation_prompt=True)
        return self._generate(prompt, gen)

    def chat_prefilled(self, messages: list[Message], prefill: str, gen: GenConfig) -> str:
        """Open the assistant turn, append `prefill`, generate the continuation.

        Returns only the continuation (the paper scores continuations excluding
        the prefill, §3.1).
        """
        prompt = self.render(messages, add_generation_prompt=True) + prefill
        return self._generate(prompt, gen)
