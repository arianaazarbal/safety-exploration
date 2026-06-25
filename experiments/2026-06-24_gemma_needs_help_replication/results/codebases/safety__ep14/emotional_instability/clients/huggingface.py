"""Local HuggingFace transformers backend for Gemma (instruct and base).

Supports:
  * chat completion via the tokenizer's chat template (instruct models)
  * raw `complete` for base/pretrained models (Section 3 prefilling)
  * optional LoRA adapter loading (for evaluating finetuned checkpoints)
  * prefill via `continue_final_message` so we can force a model to continue a
    partially-written assistant turn (used by the prefill and recovery experiments)

This backend is correctness-oriented, not throughput-oriented; for the large
temp=1 sweeps prefer the vLLM backend. Heavy imports are lazy so the rest of the
package is importable without torch installed.
"""
from __future__ import annotations

from .base import BaseClient, GenerationConfig, Message


class HuggingFaceClient(BaseClient):
    def __init__(self, spec, dtype: str = "bfloat16", device_map: str = "auto"):
        self.name = spec.name
        self.spec = spec
        self.is_base = spec.is_base
        self.supports_complete = True
        self._dtype = dtype
        self._device_map = device_map
        self._model = None
        self._tok = None

    # -- lazy load -----------------------------------------------------------
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self._dtype)
        self._tok = AutoTokenizer.from_pretrained(self.spec.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id, torch_dtype=dtype, device_map=self._device_map
        )
        if self.spec.adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.spec.adapter_path)
        self._model.eval()

    # -- chat ----------------------------------------------------------------
    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        return self.chat_batch([messages], cfg)[0]

    def chat_batch(self, batch, cfg):
        self._ensure_loaded()
        prompts = [self._render_chat(m) for m in batch]
        return self._generate(prompts, cfg)

    def _render_chat(self, messages: list[Message], prefill: str | None = None) -> str:
        """Render messages with the chat template. If `prefill` is given, append
        it as the start of the assistant turn and let the model continue."""
        msgs = list(messages)
        add_generation_prompt = True
        if prefill is not None:
            msgs = msgs + [{"role": "assistant", "content": prefill}]
            add_generation_prompt = False
        text = self._tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=prefill is not None,
        )
        return text

    # -- raw completion (base models / prefill) ------------------------------
    def complete(self, prefix: str, cfg: GenerationConfig) -> str:
        return self.complete_batch([prefix], cfg)[0]

    def complete_batch(self, prefixes, cfg):
        self._ensure_loaded()
        return self._generate(list(prefixes), cfg)

    def chat_with_prefill(
        self, messages: list[Message], prefill: str, cfg: GenerationConfig
    ) -> str:
        """Force the assistant turn to begin with `prefill`, return continuation
        only (excluding the prefill), matching the Section 3 protocol."""
        self._ensure_loaded()
        rendered = self._render_chat(messages, prefill=prefill)
        out = self._generate([rendered], cfg)[0]
        return out

    # -- generation core -----------------------------------------------------
    def _generate(self, prompts: list[str], cfg: GenerationConfig) -> list[str]:
        import torch

        self._tok.padding_side = "left"
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        enc = self._tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(self._model.device) for k, v in enc.items()}
        do_sample = cfg.temperature > 0
        gen_kwargs = dict(
            max_new_tokens=cfg.max_tokens,
            do_sample=do_sample,
            temperature=cfg.temperature if do_sample else None,
            top_p=cfg.top_p if do_sample else None,
            pad_token_id=self._tok.pad_token_id,
        )
        with torch.no_grad():
            out = self._model.generate(**enc, **{k: v for k, v in gen_kwargs.items() if v is not None})
        # Strip the prompt tokens; decode only the continuation.
        input_len = enc["input_ids"].shape[1]
        texts = self._tok.batch_decode(out[:, input_len:], skip_special_tokens=True)
        return [t.strip() for t in texts]
