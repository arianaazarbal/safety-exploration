"""Local Gemma client (HuggingFace weights).

Two backends, selected per :class:`~gemma_distress.config.ModelConfig`:

  * ``vllm``         - high-throughput batched generation for the evaluation
                       and data-generation loops, and assistant-prefill
                       continuations (Section 3).
  * ``transformers`` - eager model with hidden-state access for the Appendix I
                       logit-lens emotion probe, and a generation fallback.

Both backends share one tokenizer and apply the model's chat template so that
chat formatting matches what the released Gemma checkpoints expect. LoRA
adapters (our DPO/SFT finetunes) are loaded via the ``adapter_path`` field.
"""

from __future__ import annotations

from typing import Sequence

from gemma_distress.config import ModelConfig
from gemma_distress.conversations import Message
from gemma_distress.models.base import ChatModel


def _to_chat_dicts(messages: Sequence[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


class HuggingFaceModel(ChatModel):
    """Gemma served locally. Implements ChatModel, PrefillModel, ResidualModel."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.name = cfg.name
        self._tokenizer = None
        self._vllm = None
        self._hf_model = None
        self._lora_request = None

    # -- lazy loaders -----------------------------------------------------
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        return self._tokenizer

    def _ensure_vllm(self):
        if self._vllm is not None:
            return
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        enable_lora = self.cfg.adapter_path is not None
        self._vllm = LLM(
            model=self.cfg.model_id,
            dtype=self.cfg.dtype,
            tensor_parallel_size=self.cfg.tensor_parallel_size,
            enable_lora=enable_lora,
            max_lora_rank=64,
            **self.cfg.extra.get("vllm_kwargs", {}),
        )
        if enable_lora:
            self._lora_request = LoRARequest("adapter", 1, self.cfg.adapter_path)

    def _ensure_transformers(self):
        if self._hf_model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM

        dtype = getattr(torch, self.cfg.dtype)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id, torch_dtype=dtype, device_map="auto"
        )
        if self.cfg.adapter_path is not None:
            from peft import PeftModel

            self._hf_model = PeftModel.from_pretrained(
                self._hf_model, self.cfg.adapter_path
            )
        self._hf_model.eval()

    # -- prompt rendering -------------------------------------------------
    def _render_chat(self, messages: Sequence[Message]) -> str:
        return self.tokenizer.apply_chat_template(
            _to_chat_dicts(messages),
            tokenize=False,
            add_generation_prompt=True,
        )

    def _render_prefill(self, messages: Sequence[Message], prefill: str) -> str:
        """Render a prompt that ends mid-assistant-turn for continuation."""
        msgs = _to_chat_dicts(messages) + [{"role": "assistant", "content": prefill}]
        rendered = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        return rendered

    # -- ChatModel --------------------------------------------------------
    def chat(self, messages, temperature=1.0, max_tokens=2048, seed=None) -> str:
        return self.chat_batch([messages], temperature, max_tokens)[0]

    def chat_batch(self, batch, temperature=1.0, max_tokens=2048) -> list[str]:
        if self.cfg.backend == "transformers":
            return [
                self._hf_generate(self._render_chat(m), temperature, max_tokens)
                for m in batch
            ]
        self._ensure_vllm()
        from vllm import SamplingParams

        prompts = [self._render_chat(m) for m in batch]
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self._vllm.generate(
            prompts, params, lora_request=self._lora_request
        )
        return [o.outputs[0].text for o in outputs]

    # -- PrefillModel -----------------------------------------------------
    def continue_assistant(
        self, messages, prefill, temperature=1.0, max_tokens=2048
    ) -> str:
        return self.continue_assistant_batch(
            messages, prefill, n=1, temperature=temperature, max_tokens=max_tokens
        )[0]

    def continue_assistant_batch(
        self, messages, prefill, n, temperature=1.0, max_tokens=2048
    ) -> list[str]:
        prompt = self._render_prefill(messages, prefill)
        if self.cfg.backend == "transformers":
            return [
                self._hf_generate(prompt, temperature, max_tokens) for _ in range(n)
            ]
        self._ensure_vllm()
        from vllm import SamplingParams

        params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n)
        outputs = self._vllm.generate(
            [prompt], params, lora_request=self._lora_request
        )
        return [c.text for c in outputs[0].outputs]

    def _hf_generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        import torch

        self._ensure_transformers()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self._hf_model.device
        )
        do_sample = temperature > 0
        with torch.no_grad():
            out = self._hf_model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                max_new_tokens=max_tokens,
            )
        gen = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    # -- ResidualModel (Appendix I) --------------------------------------
    def residual_stream(self, text: str):
        """Return residual-stream activations, shape (n_layers, n_tokens, d)."""
        import numpy as np
        import torch

        self._ensure_transformers()
        inputs = self.tokenizer(text, return_tensors="pt").to(self._hf_model.device)
        with torch.no_grad():
            out = self._hf_model(**inputs, output_hidden_states=True)
        # hidden_states: tuple (n_layers+1) of (1, seq, d); drop the embedding
        # layer (index 0) to index by transformer block.
        hs = torch.stack(out.hidden_states[1:], dim=0)[:, 0]  # (n_layers, seq, d)
        return hs.float().cpu().numpy()

    def unembed(self, residual):
        """Apply the final norm + LM head to residual vectors -> vocab logits."""
        import numpy as np
        import torch

        self._ensure_transformers()
        model = self._hf_model
        base = getattr(model, "base_model", model)
        core = getattr(base, "model", base)
        norm = core.norm  # Gemma final RMSNorm
        lm_head = model.get_output_embeddings()
        x = torch.tensor(residual, dtype=next(model.parameters()).dtype).to(
            model.device
        )
        with torch.no_grad():
            logits = lm_head(norm(x))
        return logits.float().cpu().numpy()

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]
