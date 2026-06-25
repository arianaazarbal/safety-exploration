"""Gemma subject models served locally via Hugging Face ``transformers``.

Supports both instruct (chat-templated) and base (raw-continuation) checkpoints,
plus prefilled generation for the §3 experiment. Optionally loads a PEFT/LoRA
adapter on top (used to evaluate the §4 SFT/DPO interventions).

Heavy imports (``torch``/``transformers``) are done lazily inside ``__init__``
so the rest of the package can be imported without a GPU stack present.
"""

from __future__ import annotations

from typing import Iterable

from ..config import SamplingConfig
from .base import GenerationResult, Message, SubjectModel


class HFGemmaModel(SubjectModel):
    """A Gemma checkpoint loaded with transformers.

    Parameters
    ----------
    model_id:
        Hugging Face id, e.g. ``google/gemma-3-27b-it``.
    name:
        Display name for metrics; defaults to the last path segment.
    is_base:
        If True, treat as a base/pretrained model: no chat template is applied
        and generation continues raw text. Used for §3 base comparisons.
    adapter_path:
        Optional path to a PEFT/LoRA adapter to load on top (§4 interventions).
    device_map:
        Passed to ``from_pretrained``; ``"auto"`` shards across visible GPUs.
    """

    supports_tools = False

    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        *,
        is_base: bool = False,
        adapter_path: str | None = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.name = name or model_id.split("/")[-1]
        self.is_base = is_base
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=getattr(torch, dtype),
        )

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.name = f"{self.name}+{adapter_path.split('/')[-1]}"

        self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _render_prompt(self, messages: list[Message], prefill: str | None = None) -> str:
        """Render messages to a prompt string.

        Instruct models use the chat template (with an open assistant turn, and
        ``prefill`` appended if continuing). Base models receive a plain
        concatenation since they have no chat format.
        """
        if self.is_base:
            # Base model: plain text. Join turns; the caller drives structure
            # via prefilling for §3, so we mostly rely on `prefill`.
            joined = "\n\n".join(m["content"] for m in messages)
            return joined + ("\n" + prefill if prefill else "")

        # Gemma's chat template has no `system` role: fold any system content
        # into the first user turn (the standard Gemma convention).
        chat = self._merge_system_into_first_user(messages)
        text = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        if prefill:
            text = text + prefill
        return text

    @staticmethod
    def _merge_system_into_first_user(messages: list[Message]) -> list[Message]:
        """Fold system-role content into the first user turn (Gemma has no
        system role). Preserves order; non-system turns pass through unchanged."""
        system_text = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        if not system_text:
            return rest
        merged: list[Message] = []
        injected = False
        for m in rest:
            if not injected and m["role"] == "user":
                merged.append({"role": "user", "content": f"{system_text}\n\n{m['content']}"})
                injected = True
            else:
                merged.append(m)
        if not injected:  # no user turn yet — prepend system as a user turn
            merged.insert(0, {"role": "user", "content": system_text})
        return merged

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _generate_from_text(
        self, prompt_text: str, cfg: SamplingConfig
    ) -> GenerationResult:
        torch = self._torch
        # The instruct chat template already emits BOS/special tokens, so don't
        # add them again; base-model raw text does need them.
        add_special = self.is_base
        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=add_special
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=cfg.temperature > 0,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_new_tokens=cfg.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][prompt_len:].tolist()
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return GenerationResult(text=text.strip(), token_ids=gen_ids)

    def generate(self, messages: list[Message], cfg: SamplingConfig) -> GenerationResult:
        return self._generate_from_text(self._render_prompt(messages), cfg)

    def generate_with_prefill(
        self, messages: list[Message], prefill: str, cfg: SamplingConfig
    ) -> GenerationResult:
        return self._generate_from_text(self._render_prompt(messages, prefill=prefill), cfg)

    # ------------------------------------------------------------------ #
    # Tokeniser passthrough (used by §3 onset/early truncation)
    # ------------------------------------------------------------------ #
    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def detokenize(self, token_ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def close(self) -> None:  # pragma: no cover
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
