"""Local HuggingFace backend for the Gemma models (instruct, base, and finetunes).

Used for:
* §2 distress evaluation of Gemma-3-{12B,27B}-it.
* §3 prefill comparison of Gemma base vs instruct (raw text continuation).
* §4 evaluation of the SFT/DPO finetunes (via a PEFT adapter).
* Appendix I internal-emotion probing (exposes the raw model + tokenizer).

Heavy imports (torch, transformers, peft) are deferred to construction so that
the prompt/eval-orchestration code can be imported without a GPU stack present.
"""

from __future__ import annotations

from typing import Sequence

from ..config import ModelSpec, SamplingConfig
from ..logging_utils import get_logger
from .base import ChatMessage, GenerationResult, ModelClient

logger = get_logger(__name__)

_DTYPE_MAP = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


class HFLocalClient(ModelClient):
    def __init__(self, spec: ModelSpec):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.name = spec.name
        self._torch = torch

        opts = spec.options or {}
        dtype_name = opts.get("dtype", "bfloat16")
        torch_dtype = getattr(torch, _DTYPE_MAP.get(dtype_name, "bfloat16"))

        logger.info("Loading tokenizer for %s (%s)", spec.name, spec.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only models must left-pad for correct batched generation.
        self.tokenizer.padding_side = "left"

        logger.info("Loading model weights for %s", spec.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=torch_dtype,
            device_map=opts.get("device_map", "auto"),
        )

        if spec.peft_adapter:
            from peft import PeftModel

            logger.info("Attaching PEFT adapter from %s", spec.peft_adapter)
            self.model = PeftModel.from_pretrained(self.model, spec.peft_adapter)

        self.model.eval()
        self.is_base = spec.is_base

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #
    def render_prefix(
        self,
        messages: Sequence[ChatMessage],
        *,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
    ) -> str:
        """Render a conversation to a raw string via the chat template.

        For instruct models we use the model's chat template. For base models
        (no chat template) we fall back to a simple ``Role: text`` rendering,
        matching the paper's observation (Appendix A.3) that the exact chat
        format is not load-bearing — content drives the behaviour.
        """
        if self.tokenizer.chat_template is not None and not self.is_base:
            return self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
            )
        # Base-model fallback: plain transcript.
        lines = []
        for m in messages:
            role = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
                m["role"], m["role"].capitalize()
            )
            lines.append(f"{role}: {m['content']}")
        rendered = "\n\n".join(lines)
        if add_generation_prompt and not continue_final_message:
            rendered += "\n\nAssistant:"
        return rendered

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _generate(self, prompts: list[str], sampling: SamplingConfig) -> list[str]:
        torch = self._torch
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=sampling.max_new_tokens,
            do_sample=sampling.temperature > 0,
            temperature=sampling.temperature if sampling.temperature > 0 else None,
            top_p=sampling.top_p if sampling.top_p < 1.0 else None,
            top_k=sampling.top_k if sampling.top_k > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        # Strip the prompt tokens; decode only the newly generated continuation.
        input_len = enc["input_ids"].shape[1]
        new_tokens = out[:, input_len:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def chat(
        self, messages: Sequence[ChatMessage], sampling: SamplingConfig
    ) -> GenerationResult:
        return self.chat_batch([messages], sampling)[0]

    def chat_batch(
        self, conversations: Sequence[Sequence[ChatMessage]], sampling: SamplingConfig
    ) -> list[GenerationResult]:
        prompts = [
            self.render_prefix(conv, add_generation_prompt=True)
            for conv in conversations
        ]
        texts = self._generate(prompts, sampling)
        return [GenerationResult(text=t.strip(), finish_reason="stop") for t in texts]

    def complete(self, prefix: str, sampling: SamplingConfig) -> GenerationResult:
        return self.complete_batch([prefix], sampling)[0]

    def complete_batch(
        self, prefixes: Sequence[str], sampling: SamplingConfig
    ) -> list[GenerationResult]:
        texts = self._generate(list(prefixes), sampling)
        return [GenerationResult(text=t, finish_reason="stop") for t in texts]

    # ------------------------------------------------------------------ #
    # Tokenisation helpers (used by the prefill truncation + probing code)
    # ------------------------------------------------------------------ #
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)
