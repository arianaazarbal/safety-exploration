"""Local HuggingFace backend for Gemma (instruct, pretrained, and LoRA-adapted).

This backend is the only one that can:
  * run the open-weight Gemma base model (gemma-3-27b-pt) for Section 3,
  * prefill an arbitrary assistant string and continue from it,
  * report generated token ids for token-accurate truncation (Section 3),
  * load a LoRA adapter on top of the instruct model (Section 4 DPO/SFT eval).

For base/pretrained models we still render the conversation with Gemma's chat
template (when present) so base and instruct continue from byte-identical
contexts — the comparison in Section 3 depends on that. A plain role-tagged
fallback is used if the tokenizer ships no chat template.
"""
from __future__ import annotations

from emotelic.models.base import ChatMessage, GenerationResult


def _fallback_format(messages: list[ChatMessage]) -> str:
    parts = []
    for m in messages:
        parts.append(f"<start_of_turn>{m.role}\n{m.content}<end_of_turn>")
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


class HFLocalClient:
    supports_prefill = True

    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        is_instruct: bool = True,
        adapter_path: str | None = None,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        **_: object,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.hf_id = hf_id
        self.is_instruct = is_instruct
        self.adapter_path = adapter_path

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        quant_kwargs: dict = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            **quant_kwargs,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self._torch = torch

    # ---- prompt construction -------------------------------------------------
    def _render(self, messages: list[ChatMessage], prefill: str | None) -> str:
        has_template = getattr(self.tokenizer, "chat_template", None)
        if has_template:
            text = self.tokenizer.apply_chat_template(
                [m.as_dict() for m in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = _fallback_format(messages)
        if prefill:
            text = text + prefill
        return text

    # ---- token helpers used by the prefill experiment ------------------------
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    # ---- generation ----------------------------------------------------------
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        prefill: str | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        results = self.generate_batch(
            messages, n=1, temperature=temperature, max_tokens=max_tokens,
            prefill=prefill, seed=seed,
        )
        return results[0]

    def generate_batch(
        self,
        messages: list[ChatMessage],
        *,
        n: int = 1,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        prefill: str | None = None,
        seed: int | None = None,
    ) -> list[GenerationResult]:
        """Sample `n` continuations from the same context (used for the 50
        continuations/prefill in Section 3). Returns the FULL text including any
        prefill; callers strip the prefill to get the continuation only."""
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)
        prompt = self._render(messages, prefill)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        do_sample = temperature and temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                max_new_tokens=max_tokens,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        results = []
        for seq in out:
            gen_ids = seq[prompt_len:].tolist()
            continuation = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            full = (prefill or "") + continuation
            results.append(
                GenerationResult(
                    text=full,
                    model=self.hf_id,
                    finish_reason="stop",
                    token_ids=gen_ids,
                )
            )
        return results
