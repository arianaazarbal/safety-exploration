"""Local HuggingFace inference backend for the open-weight Gemma models.

Handles both instruct (chat-templated) and base/pretrained (raw-text
continuation) checkpoints, batched multi-sample generation, and assistant-turn
prefilling (needed for the Section 3 base-vs-instruct experiment).

Heavy deps (torch / transformers / peft) are imported lazily so that the rest of
the package can be used in API-only environments.
"""
from __future__ import annotations

from typing import Sequence

from .base import ChatMessage, GenerationResult, ModelBackend


class HFBackend(ModelBackend):
    def __init__(
        self,
        model_id: str,
        *,
        family: str = "gemma",
        kind: str = "instruct",
        load_in_4bit: bool = False,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        import torch  # noqa: F401
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.family = family
        self.kind = kind
        self.adapter_path = adapter_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding so that newly generated tokens are contiguous at the end
        # for every sequence in a batch.
        self.tokenizer.padding_side = "left"

        self.model = self._load_model(load_in_4bit, dtype, device_map)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ------------------------------------------------------------------ #
    def _load_model(self, load_in_4bit, dtype, device_map):
        import torch
        from transformers import AutoModelForCausalLM

        kwargs = dict(device_map=device_map, torch_dtype=getattr(torch, dtype))
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        # Gemma-3 instruct checkpoints are multimodal
        # (Gemma3ForConditionalGeneration); the text-only generation path works
        # through AutoModelForImageTextToText. Pretrained text checkpoints load
        # via AutoModelForCausalLM. Try the causal-LM path first and fall back.
        try:
            return AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        except (ValueError, KeyError):
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, **kwargs
            )
            # expose the language model so .generate returns text tokens only
            return getattr(model, "language_model", model)

    # ------------------------------------------------------------------ #
    def _render_prompt(
        self, messages: Sequence[ChatMessage], prefill: str | None
    ) -> str:
        if self.kind == "instruct":
            text = self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
            if prefill:
                text = text + prefill
            return text
        # Base / pretrained: no chat template. Render a role-tagged transcript so
        # the model continues coherently, then supply the assistant-response
        # prefix. The Section 3 prefill builder packs the full context (question
        # + prior turns + rejection) into `messages` and the truncated emotional
        # response into `prefill`.
        tags = {"system": "System", "user": "User", "assistant": "Assistant"}
        parts = [f"{tags.get(m['role'], m['role'])}: {m['content'].strip()}"
                 for m in messages]
        rendered = "\n\n".join(parts)
        # Open an assistant turn for the continuation.
        if rendered:
            rendered += "\n\nAssistant: "
        return rendered + (prefill or "")

    # ------------------------------------------------------------------ #
    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> list[GenerationResult]:
        import torch

        prompt = self._render_prompt(messages, prefill)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.model.device)
        attn = enc["attention_mask"].to(self.model.device)
        prompt_len = input_ids.shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature and temperature > 0:
            # Pure temperature sampling (paper only specifies T=1).
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, attention_mask=attn, **gen_kwargs
            )

        results: list[GenerationResult] = []
        for seq in out:
            new_tokens = seq[prompt_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            if stop:
                text = _truncate_at_stop(text, stop)
            results.append(GenerationResult(text=text, n_tokens=int(new_tokens.shape[0])))
        return results

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])


def _truncate_at_stop(text: str, stop: Sequence[str]) -> str:
    cut = len(text)
    for s in stop:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]
