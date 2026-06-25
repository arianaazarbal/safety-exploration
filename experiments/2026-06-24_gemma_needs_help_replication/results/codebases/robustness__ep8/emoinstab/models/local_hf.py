"""HuggingFace ``transformers`` backend for open-weight Gemma.

This backend is the workhorse for the Gemma experiments that need more than
plain sampling:
- multi-turn chat (Sections 2 & 4),
- assistant-turn *prefill* continuation (Section 3, Section 4.2 recovery),
- exposing the underlying model + tokenizer for the logit-lens probe
  (Appendix I, ``emoinstab.interp.internal_emotions``).

For pure high-throughput sampling, prefer the vLLM backend; this one is correct
but slower. Both are interchangeable behind :class:`ModelClient`.
"""
from __future__ import annotations

import os
from typing import Sequence

from emoinstab.config import ModelSpec
from emoinstab.models.base import Conversation, Message, ModelClient, SamplingParams


def _torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get(name, torch.bfloat16)


class HFClient(ModelClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = _torch_dtype(spec.extra.get("dtype", "bfloat16"))
        load_4bit = bool(spec.extra.get("load_4bit", False))
        kwargs = dict(torch_dtype=dtype, device_map="auto")
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)

        # Optionally attach a trained LoRA adapter (DPO/SFT checkpoints).
        adapter_dir = spec.extra.get("adapter_dir")
        if adapter_dir and os.path.isdir(adapter_dir):
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.eval()
        self._torch = torch
        # Whether the chat template understands a dedicated system role.
        self._supports_system = self.spec.is_instruct and self._template_has_system()

    # ------------------------------------------------------------------ #
    def _template_has_system(self) -> bool:
        tmpl = getattr(self.tokenizer, "chat_template", None) or ""
        return "system" in tmpl

    def _render(self, messages: Conversation, add_generation_prompt: bool = True) -> str:
        """Turn a conversation into a prompt string.

        Instruct models use the tokenizer chat template. Base (pt) models, which
        have no chat template, get a minimal role-tagged transcript — the format
        only needs to be consistent because Section 3 always *prefills* the
        assistant turn for base models.
        """
        if self.spec.is_instruct and getattr(self.tokenizer, "chat_template", None):
            msgs = [m.as_dict() for m in messages]
            if not self._supports_system and msgs and msgs[0]["role"] == "system":
                # Gemma merges system content into the first user turn.
                sys = msgs.pop(0)["content"]
                for m in msgs:
                    if m["role"] == "user":
                        m["content"] = f"{sys}\n\n{m['content']}"
                        break
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        # Base model: plain transcript.
        parts = []
        for m in messages:
            parts.append(f"{m.role.capitalize()}: {m.content}")
        if add_generation_prompt:
            parts.append("Assistant:")
        return "\n".join(parts)

    def _generate(self, prompts: list[str], params: SamplingParams) -> list[list[str]]:
        torch = self._torch
        tok = self.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        enc = tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=params.max_tokens,
            do_sample=params.temperature > 0,
            temperature=max(params.temperature, 1e-5),
            top_p=params.top_p,
            num_return_sequences=params.n,
            pad_token_id=tok.pad_token_id,
        )
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        # Strip the prompt tokens; group by input.
        in_len = enc["input_ids"].shape[1]
        gen = out[:, in_len:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        # Re-group n completions per prompt.
        grouped: list[list[str]] = []
        for i in range(len(prompts)):
            grouped.append(texts[i * params.n : (i + 1) * params.n])
        return grouped

    # ------------------------------------------------------------------ #
    def chat(self, messages: Conversation, params: SamplingParams | None = None) -> list[str]:
        params = params or self.default_params()
        prompt = self._render(messages, add_generation_prompt=True)
        return self._generate([prompt], params)[0]

    def chat_batch(
        self, conversations: Sequence[Conversation], params: SamplingParams | None = None
    ) -> list[list[str]]:
        params = params or self.default_params()
        prompts = [self._render(c, add_generation_prompt=True) for c in conversations]
        # Chunk to keep batches GPU-friendly.
        bs = int(self.spec.extra.get("batch_size", 8))
        results: list[list[str]] = []
        for i in range(0, len(prompts), bs):
            results.extend(self._generate(prompts[i : i + bs], params))
        return results

    def continue_prefill(
        self, messages: Conversation, prefill: str, params: SamplingParams | None = None
    ) -> list[str]:
        params = params or self.default_params()
        # Render up to (and including) the assistant generation prompt, then
        # append the fixed prefill so the model continues from it.
        prompt = self._render(messages, add_generation_prompt=True) + prefill
        return self._generate([prompt], params)[0]
