"""HuggingFace local-inference backend for Gemma (instruct + base/pt).

Used for: Section 2 elicitation (Gemma instruct), Section 3 prefill (Gemma base
and instruct), Section 4 finetuned-model evaluation (LoRA adapters loaded on top
of Gemma-3-27b-it).

Key requirements:
* temperature = 1 sampling (paper default).
* ``n`` independent samples per prompt (we expand the batch).
* prefill / continuation: append a partial assistant message and let the model
  extend it. Instruct models use ``apply_chat_template(..., continue_final_message
  =True)``; base models (no chat template) use a plain-text format.
* optional LoRA adapter loading for the finetuned Section 4 models.
"""
from __future__ import annotations

from typing import Optional

from ..config import ModelSpec, SamplingConfig
from .base import ChatModel, Message


class HFChatModel(ChatModel):
    def __init__(
        self,
        spec: ModelSpec,
        *,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        adapter_path: Optional[str] = None,
        max_batch: int = 16,
    ):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_batch = max_batch

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding is required for correct batched generation w/ decoder LMs.
        self.tokenizer.padding_side = "left"

        model_kwargs: dict = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )
        else:
            model_kwargs["torch_dtype"] = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **model_kwargs)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

        # Does this checkpoint carry a chat template? Base/pt models do not.
        self.has_chat_template = (
            getattr(self.tokenizer, "chat_template", None) is not None and not spec.is_base
        )

    # ------------------------------------------------------------------ #
    # Prompt formatting
    # ------------------------------------------------------------------ #
    def _format(self, messages: list[Message], prefill: Optional[str]) -> str:
        """Render messages to a single prompt string.

        Instruct models: use the chat template; if ``prefill`` is given we append
        an assistant message and request ``continue_final_message`` so no
        end-of-turn token is emitted before the continuation.

        Base models: there is no chat template, so we use a lightweight, neutral
        transcript format. The paper uses prefilling precisely because base
        models are not chat-tuned; the exact scaffold is not load-bearing (App.
        A.3 shows formatting is not the driver), so we keep it simple and
        consistent across base/instruct for the prefill experiment.
        """
        if self.has_chat_template:
            msgs = list(messages)
            if prefill is not None:
                msgs = msgs + [{"role": "assistant", "content": prefill}]
                return self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    continue_final_message=True,
                )
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

        # Base-model plain-text transcript.
        lines = []
        for m in messages:
            tag = {"user": "User", "assistant": "Assistant", "system": "System"}[m["role"]]
            lines.append(f"{tag}: {m['content']}")
        text = "\n".join(lines) + "\nAssistant:"
        if prefill is not None:
            text += " " + prefill
        return text

    # ------------------------------------------------------------------ #
    # Generation core
    # ------------------------------------------------------------------ #
    def _run(self, prompt: str, sampling: SamplingConfig, n: int) -> list[str]:
        torch = self.torch
        outputs: list[str] = []
        remaining = n
        while remaining > 0:
            b = min(self.max_batch, remaining)
            enc = self.tokenizer([prompt] * b, return_tensors="pt", padding=True).to(
                self.model.device
            )
            gen_kwargs = dict(
                max_new_tokens=sampling.max_new_tokens,
                do_sample=sampling.temperature > 0,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            with torch.no_grad():
                out = self.model.generate(**enc, **gen_kwargs)
            # Strip the prompt tokens; decode only the continuation.
            gen = out[:, enc["input_ids"].shape[1]:]
            decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            outputs.extend(s.strip() for s in decoded)
            remaining -= b
        return outputs

    def generate(self, messages, sampling, n=1):
        prompt = self._format(messages, prefill=None)
        return self._run(prompt, sampling, n)

    def continue_prefill(self, messages, prefill, sampling, n=1):
        prompt = self._format(messages, prefill=prefill)
        # The continuation already excludes the prefill because we decode only
        # the newly generated tokens.
        return self._run(prompt, sampling, n)
