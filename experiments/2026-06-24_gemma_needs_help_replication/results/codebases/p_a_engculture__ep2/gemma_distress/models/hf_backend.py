"""Local HuggingFace backend for Gemma models.

Loads a Gemma 3 checkpoint (instruct or pretrained) with optional LoRA adapter and
optional 4-bit quantisation, and exposes batched sampling plus assistant prefilling.

Notes
-----
* Gemma 3 instruct checkpoints are exposed on the hub as ``Gemma3ForConditionalGeneration``
  (a multimodal wrapper). For text-only use we try ``AutoModelForCausalLM`` first and fall
  back to the conditional-generation class, generating from its text tower. This keeps the
  same code path for both the 12B and 27B models.
* For pretrained ("pt") checkpoints there is no chat template, so prefilling and the
  fairness-preserving chat scaffolding fall back to an explicit Gemma-format template so
  that base and instruct models continue from byte-identical prefixes (Section 3.1).
* Generation uses left padding (required for correct decoder-only batched generation) and
  stops on the ``<end_of_turn>`` token when present.

This is the reference backend. For the 4000-sample sweeps the vLLM backend
(:mod:`gemma_distress.models.vllm_backend`) is far faster and is selected by giving the
model ``backend: vllm`` in config.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import ChatModel, Conversation

logger = logging.getLogger(__name__)

# Explicit Gemma chat format, used for base checkpoints that ship no chat template.
_GEMMA_TURN = "<start_of_turn>{role}\n{content}<end_of_turn>\n"
_GEMMA_GEN_PROMPT = "<start_of_turn>model\n"


def _gemma_format(conversation: Conversation, add_generation_prompt: bool) -> str:
    """Render a conversation in Gemma's turn format (manual fallback for base models).

    Gemma has no dedicated system role; a leading system message is folded into the first
    user turn, mirroring the official instruct template.
    """
    msgs = list(conversation)
    system = None
    if msgs and msgs[0]["role"] == "system":
        system = msgs[0]["content"]
        msgs = msgs[1:]
    parts = ["<bos>"]
    for i, m in enumerate(msgs):
        role = "model" if m["role"] == "assistant" else "user"
        content = m["content"]
        if i == 0 and system and role == "user":
            content = f"{system}\n\n{content}"
        parts.append(_GEMMA_TURN.format(role=role, content=content))
    if add_generation_prompt:
        parts.append(_GEMMA_GEN_PROMPT)
    return "".join(parts)


class HFBackend(ChatModel):
    """Transformers-based Gemma backend with optional LoRA and 4-bit loading."""

    supports_prefill = True

    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        is_base: bool = False,
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ):
        super().__init__(name)
        import torch  # local import keeps the package importable without torch installed
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.is_base = is_base
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Left padding is required for correct batched decoder-only generation.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._load_model(
            model_id, load_in_4bit=load_in_4bit, device_map=device_map, dtype=dtype
        )
        if adapter_path:
            from peft import PeftModel

            logger.info("Loading LoRA adapter for %s from %s", name, adapter_path)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

        self._eos_ids = self._resolve_eos_ids()

    # -- loading -----------------------------------------------------------------

    def _load_model(self, model_id, *, load_in_4bit, device_map, dtype):
        import torch
        from transformers import AutoModelForCausalLM

        torch_dtype = getattr(torch, dtype)
        kwargs = dict(torch_dtype=torch_dtype, device_map=device_map)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
            )
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except (ValueError, KeyError, OSError) as exc:
            logger.info(
                "AutoModelForCausalLM failed for %s (%s); trying Gemma3ForConditionalGeneration.",
                model_id, exc,
            )
            from transformers import Gemma3ForConditionalGeneration

            return Gemma3ForConditionalGeneration.from_pretrained(model_id, **kwargs)

    def _resolve_eos_ids(self) -> list[int]:
        eos = [self.tokenizer.eos_token_id]
        try:
            end_of_turn = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if end_of_turn is not None and end_of_turn >= 0:
                eos.append(end_of_turn)
        except Exception:  # pragma: no cover - tokenizer without the special token
            pass
        return sorted({e for e in eos if e is not None})

    # -- prompt rendering --------------------------------------------------------

    def _render(self, conversation: Conversation, add_generation_prompt: bool) -> str:
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        return _gemma_format(conversation, add_generation_prompt)

    # -- generation --------------------------------------------------------------

    def _generate(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_new_tokens: int,
        n: int,
    ) -> list[list[str]]:
        import torch

        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self._eos_ids,
        )
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        # `out` is (batch * n, seq). Strip the prompt tokens, decode the new tokens.
        input_len = enc["input_ids"].shape[1]
        gen_tokens = out[:, input_len:]
        texts = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        # Regroup into [batch][n].
        grouped: list[list[str]] = []
        for i in range(len(prompts)):
            grouped.append([texts[i * n + j].strip() for j in range(n)])
        return grouped

    def chat_batch(
        self,
        conversations: list[Conversation],
        *,
        temperature: float,
        max_new_tokens: int,
        n: int = 1,
    ) -> list[list[str]]:
        prompts = [self._render(c, add_generation_prompt=True) for c in conversations]
        return self._generate(
            prompts, temperature=temperature, max_new_tokens=max_new_tokens, n=n
        )

    def continue_from_prefill(
        self,
        conversation: Conversation,
        prefill: str,
        *,
        n: int,
        temperature: float,
        max_new_tokens: int,
    ) -> list[str]:
        # Render up to the generation prompt, then append the assistant prefix verbatim.
        prompt = self._render(conversation, add_generation_prompt=True) + prefill
        return self._generate(
            [prompt], temperature=temperature, max_new_tokens=max_new_tokens, n=n
        )[0]

    def close(self) -> None:  # pragma: no cover
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
