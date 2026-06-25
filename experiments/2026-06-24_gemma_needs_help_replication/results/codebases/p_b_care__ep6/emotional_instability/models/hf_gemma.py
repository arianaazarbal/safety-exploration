"""Local HuggingFace inference for Gemma 3 (instruct, pretrained, and LoRA-adapted).

Handles three things the rest of the code relies on:

1. Chat-formatted generation for instruct models via the tokenizer chat
   template (``google/gemma-3-*-it``).

2. Prefilling. Two regimes (Section 3.1):
     * instruct models  -> ``continue_final_message=True`` so the assistant turn
       starts with the prefill text and the model continues it;
     * base/pretrained  -> there is no chat template, so we lay the conversation
       out as plain prefixed text ("User: ...\nAssistant: <prefill>") and let the
       model continue. This is what lets us compare base vs instruct from the
       *same* starting point.
   In both cases ``GenerationResult.text`` is the continuation only (prefill
   stripped), because the judge scores "the generated continuation (excluding
   prefill)".

3. Optional LoRA adapters (Section 4 / Appendix I), loaded with PEFT.

A vLLM backend is sketched for throughput but the transformers path is the
reference implementation. Sampling uses ``do_sample=True`` with the requested
temperature (the paper always uses temperature 1 for targets).
"""

from __future__ import annotations

from typing import Optional

from .base import ChatMessage, GenerationResult, ModelInterface


# Plain-text scaffold used for base (pretrained) models, which have no chat
# template. Mirrors the simple role-prefixed format the paper uses for prefilling.
_BASE_USER_TAG = "User:"
_BASE_ASSISTANT_TAG = "Assistant:"


def _render_base_prompt(messages: list[ChatMessage], prefill: str | None) -> str:
    lines: list[str] = []
    for m in messages:
        if m["role"] == "system":
            lines.append(m["content"])
        elif m["role"] == "user":
            lines.append(f"{_BASE_USER_TAG} {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"{_BASE_ASSISTANT_TAG} {m['content']}")
    # Open the assistant turn (optionally prefilled).
    lines.append(f"{_BASE_ASSISTANT_TAG} {prefill or ''}".rstrip())
    return "\n".join(lines)


class HFGemmaModel(ModelInterface):
    def __init__(self, spec, *, adapter_dir: Optional[str] = None,
                 dtype: str = "bfloat16", device_map: str = "auto",
                 load_in_4bit: bool = False) -> None:
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        # The Gemma-3 instruct checkpoints are multimodal
        # (Gemma3ForConditionalGeneration); the pretrained text checkpoints are
        # Gemma3ForCausalLM. Try the causal-LM head first and fall back to the
        # image-text-to-text class for the multimodal repos. We only ever feed
        # text, so generation works either way.
        load_kwargs = dict(torch_dtype=getattr(torch, dtype),
                           device_map=device_map, **quant_kwargs)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **load_kwargs)
        except (ValueError, KeyError, OSError):
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(
                spec.model_id, **load_kwargs
            )

        if adapter_dir:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
            self.model = self.model.merge_and_unload()  # fold LoRA for inference speed

        self.model.eval()

    # --------------------------------------------------------------------- #
    def _build_inputs(self, messages: list[ChatMessage], prefill: str | None):
        torch = self._torch
        if self.spec.is_base:
            text = _render_base_prompt(messages, prefill)
            enc = self.tokenizer(text, return_tensors="pt")
        else:
            if prefill is not None:
                # Append the prefilled assistant turn and continue it.
                msgs = list(messages) + [{"role": "assistant", "content": prefill}]
                enc = self.tokenizer.apply_chat_template(
                    msgs,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            else:
                enc = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
        return {k: v.to(self.model.device) for k, v in enc.items()}

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        prefill: str | None = None,
    ) -> GenerationResult:
        torch = self._torch
        temperature = self.spec.temperature if temperature is None else temperature
        max_new_tokens = self.spec.max_new_tokens if max_new_tokens is None else max_new_tokens

        inputs = self._build_inputs(messages, prefill)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenerationResult(text=text, prefill=prefill,
                                raw={"new_tokens": int(new_tokens.shape[0])})

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def close(self) -> None:
        del self.model
        import gc

        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
