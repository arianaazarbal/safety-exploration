"""Local HuggingFace transformers client for Gemma (instruct + pretrained).

Handles three things the harness needs:

  * Instruct chat formatting via the tokenizer chat template.
  * Assistant *prefill* / continuation (for Section 3 and for base models),
    using `continue_final_message=True`.
  * Plain-text formatting for base ("pt") checkpoints, which have no chat
    template - we render a minimal transcript and let the model continue the
    trailing assistant text.

True batched decoding (left-padded) is implemented in `generate_batch`, since
sampling thousands of rollouts one-at-a-time would be impractically slow.
"""

from __future__ import annotations

from typing import Sequence

from ..config import ModelSpec
from .base import ChatMessage, ChatModel


class HFModel(ChatModel):
    def __init__(
        self,
        spec: ModelSpec,
        max_concurrency: int = 1,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ):
        super().__init__(spec, max_concurrency=1)  # GPU work is not thread-parallel
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        load_kwargs = dict(device_map=device_map, torch_dtype=getattr(torch, dtype))
        if spec.load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding is required for correct batched generation.
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **load_kwargs)
        self.model.eval()
        self._has_chat_template = (
            not spec.is_base and self.tokenizer.chat_template is not None
        )

    @property
    def supports_prefill(self) -> bool:
        return True

    # -- prompt rendering ---------------------------------------------------

    def _render(self, messages: Sequence[ChatMessage]) -> str:
        """Render a conversation to a single prompt string.

        If the final message is an assistant turn, we render it as a *prefill*
        to be continued rather than as a finished turn.
        """
        msgs = list(messages)
        prefill = msgs and msgs[-1]["role"] == "assistant"

        if self._has_chat_template:
            if prefill:
                return self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

        # Base model: minimal transcript format. Kept deliberately plain because
        # the point of the prefill experiment is to let the base model continue
        # free-form text rather than imposing instruct chat structure.
        lines = []
        for m in msgs:
            if m["role"] == "system":
                lines.append(m["content"])
            elif m["role"] == "user":
                lines.append(f"User: {m['content']}")
            else:  # assistant
                lines.append(f"Assistant: {m['content']}")
        text = "\n".join(lines)
        if not prefill:
            text += "\nAssistant:"
        return text

    # -- generation ---------------------------------------------------------

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> str:
        return self.generate_batch(
            [messages],
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )[0]

    def generate_batch(
        self,
        batch: Sequence[Sequence[ChatMessage]],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> list[str]:
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)

        prompts = [self._render(m) for m in batch]
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)

        do_sample = temperature and temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Strip the prompt tokens; decode only the newly generated continuation.
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def close(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
