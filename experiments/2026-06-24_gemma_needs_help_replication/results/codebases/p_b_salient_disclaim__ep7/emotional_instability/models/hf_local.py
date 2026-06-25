"""Local HuggingFace inference for Gemma (instruct, base, and LoRA finetunes).

Supports:
  * chat generation via the tokenizer chat template (instruct models),
  * a plain-text rendering for base ("pt") models that lack a chat template,
  * prefill continuation (continue from a partial final assistant turn),
  * batched generation,
  * residual-stream / unembedding hooks for Appendix I (internal emotions).

The 27B model needs a multi-GPU node or 4-bit loading; both are configurable
via config.py (DEVICE_MAP, LOAD_IN_4BIT). Nothing here is run during this task
— it is the inference layer the experiment scripts call.
"""

from __future__ import annotations

from typing import Optional

import config
from .base import ChatMessage, GenerationResult, ModelClient


def _torch_dtype():
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[config.TORCH_DTYPE]


class HFLocalClient(ModelClient):
    supports_prefill = True
    supports_hidden_states = True

    def __init__(self, name: str, model_id: str, *, is_base: bool = False,
                 adapter_path: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.model_id = model_id
        self.is_base = is_base
        self.adapter_path = adapter_path

        quant_kwargs = {}
        if config.LOAD_IN_4BIT:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=_torch_dtype(),
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=_torch_dtype(), device_map=config.DEVICE_MAP,
            **quant_kwargs)

        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[ChatMessage],
                add_generation_prompt: bool = True,
                assistant_prefix: Optional[str] = None) -> str:
        """Render messages to a prompt string.

        Instruct models: use the Gemma chat template.
        Base models: a simple, fixed transcript rendering (the paper relies on
        prefilling for base models; the exact base-model rendering is
        underspecified, so we use a minimal Role: content format — see DESIGN.md).
        """
        if not self.is_base and self.tokenizer.chat_template:
            msgs = [{"role": m.role, "content": m.content} for m in messages]
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
            if assistant_prefix is not None:
                text = text + assistant_prefix
            return text
        # Base-model plain rendering
        parts = []
        for m in messages:
            tag = {"system": "System", "user": "User", "assistant": "Assistant"}[m.role]
            parts.append(f"{tag}: {m.content}")
        if add_generation_prompt or assistant_prefix is not None:
            parts.append("Assistant:" + (f" {assistant_prefix}" if assistant_prefix else " "))
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _generate_from_text(self, prompt_text: str, *, temperature: float,
                            max_new_tokens: int, seed: Optional[int]) -> GenerationResult:
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=1.0, top_k=0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            prompt_token_count=int(inputs["input_ids"].shape[1]),
            completion_token_count=int(gen_ids.shape[0]),
        )

    def generate(self, messages, *, temperature, max_new_tokens, seed=None):
        prompt = self._render(messages, add_generation_prompt=True)
        return self._generate_from_text(
            prompt, temperature=temperature, max_new_tokens=max_new_tokens, seed=seed)

    def generate_batch(self, batch, *, temperature, max_new_tokens, seed=None):
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        prompts = [self._render(m, add_generation_prompt=True) for m in batch]
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=1.0, top_k=0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        results = []
        in_len = enc["input_ids"].shape[1]
        for i in range(out.shape[0]):
            gen_ids = out[i][in_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append(GenerationResult(text=text,
                                            prompt_token_count=in_len,
                                            completion_token_count=int(gen_ids.shape[0])))
        return results

    def continue_prefill(self, messages, assistant_prefix, *, temperature,
                         max_new_tokens, seed=None):
        prompt = self._render(messages, add_generation_prompt=False,
                              assistant_prefix=assistant_prefix)
        # The continuation excludes the prefix because we generate from the end
        # of `prompt` (which already contains the prefix) and return only new tokens.
        return self._generate_from_text(
            prompt, temperature=temperature, max_new_tokens=max_new_tokens, seed=seed)

    # ------------------------------------------------------------------ #
    # Token utilities
    # ------------------------------------------------------------------ #
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------ #
    # Hidden states / unembedding (Appendix I)
    # ------------------------------------------------------------------ #
    def residual_stream_logits(self, prompt_text: str, layers: list[int]):
        """Return, for each requested layer, the unembedded logits of that
        layer's residual stream at every token position.

        Implementation: run with output_hidden_states=True, apply the model's
        final norm + lm_head (the unembedding) to each selected hidden state.
        Returns a dict {layer: tensor[seq_len, vocab]} on CPU.
        """
        import torch

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple: (embeddings, layer1, ..., layerN)

        base = self.model
        # Unwrap PEFT / handle architecture-specific norm + lm_head.
        core = getattr(base, "base_model", base)
        # Gemma: model.model.norm and model.lm_head (tied embeddings).
        lm_head = getattr(base, "lm_head", None) or getattr(core, "lm_head", None)
        final_norm = None
        inner = getattr(core, "model", core)
        final_norm = getattr(inner, "norm", None)

        logits_by_layer = {}
        for layer in layers:
            hs = hidden_states[layer][0]            # [seq, hidden]
            if final_norm is not None:
                hs = final_norm(hs)
            with torch.no_grad():
                logits = lm_head(hs)                # [seq, vocab]
            logits_by_layer[layer] = logits.float().cpu()
        return logits_by_layer, inputs["input_ids"][0].cpu()
