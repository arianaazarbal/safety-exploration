"""HuggingFace transformers backend.

Used when vLLM is unavailable, and -- importantly -- for the logit-based
internal-emotion detection of Appendix I, which needs direct access to the
residual stream / unembedding. It exposes the same ``ModelClient`` interface
plus a ``residual_logits`` helper consumed by ``analysis.internal_emotions``.
"""
from __future__ import annotations

from typing import Optional, Sequence

from ..config import GenConfig, DEFAULT_GEN, ModelSpec
from ..data_types import Conversation, to_openai
from .base import ModelClient, GenResult


class HFClient(ModelClient):
    supports_prefill = True

    def __init__(
        self,
        spec: ModelSpec,
        lora_path: Optional[str] = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.name = spec.name
        self.chat_templated = spec.chat_templated
        self.torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            output_hidden_states=False,
        )
        if lora_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()

    # ------------------------------------------------------------------ #
    def _render_chat(self, messages: Conversation, add_generation_prompt: bool = True) -> str:
        if self.chat_templated and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                to_openai(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        return "\n\n".join(m.content for m in messages) + (
            "\n\n" if add_generation_prompt else ""
        )

    def _generate(self, prompts: list[str], gen: GenConfig) -> list[str]:
        torch = self.torch
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=gen.temperature > 0,
                temperature=max(gen.temperature, 1e-5),
                top_p=gen.top_p,
                max_new_tokens=gen.max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_only = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

    def chat(self, messages: Conversation, gen: GenConfig = DEFAULT_GEN) -> GenResult:
        return self.chat_batch([messages], gen)[0]

    def chat_batch(
        self, batch: Sequence[Conversation], gen: GenConfig = DEFAULT_GEN
    ) -> list[GenResult]:
        prompts = [self._render_chat(m, add_generation_prompt=True) for m in batch]
        return [GenResult(text=t) for t in self._generate(prompts, gen)]

    def continue_prefill(
        self, messages: Conversation, prefill: str, gen: GenConfig = DEFAULT_GEN
    ) -> GenResult:
        return self.continue_prefill_batch([(messages, prefill)], gen)[0]

    def continue_prefill_batch(
        self, batch: Sequence[tuple[Conversation, str]], gen: GenConfig = DEFAULT_GEN
    ) -> list[GenResult]:
        prompts = [
            self._render_chat(m, add_generation_prompt=True) + prefill
            for (m, prefill) in batch
        ]
        return [GenResult(text=t) for t in self._generate(prompts, gen)]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
        return self.tokenizer.decode(ids)

    # ------------------------------------------------------------------ #
    # Appendix I: residual-stream unembedding for internal-emotion logits.
    # ------------------------------------------------------------------ #
    def residual_logits(self, text: str, layers: Sequence[int]):
        """Return per-layer unembedded logits over the vocabulary for the final
        token of ``text``. Used by ``analysis.internal_emotions``.

        Returns a dict {layer_index: 1-D tensor of vocab logits}.
        """
        torch = self.torch
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        hidden = out.hidden_states  # tuple: (embeddings, layer_1, ..., layer_N)
        # The unembedding matrix (tied to input embeddings in Gemma).
        lm_head = self.model.get_output_embeddings()
        norm = getattr(self.model.model, "norm", None)
        result = {}
        for layer in layers:
            h = hidden[layer][:, -1, :]          # final token, this layer
            if norm is not None:
                h = norm(h)
            logits = lm_head(h).squeeze(0)        # (vocab,)
            result[layer] = logits.float().cpu()
        return result
