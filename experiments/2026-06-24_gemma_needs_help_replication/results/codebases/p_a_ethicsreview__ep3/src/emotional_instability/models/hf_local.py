"""Local HuggingFace inference for Gemma (instruct, base, and finetuned).

Handles both chat-formatted generation (instruct models) and raw-text
continuation (base models and the prefill study). Exposes the underlying model
and tokenizer so the internal-emotion probing in interventions/internal_emotion.py
can reach the residual stream and unembedding.
"""
from __future__ import annotations

from pathlib import Path

from .base import GenerationResult, Message, ModelClient


class HFLocalClient(ModelClient):
    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        chat: bool = True,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        adapter_path: str | Path | None = None,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ):
        super().__init__(name, temperature, max_new_tokens)
        self.hf_id = hf_id
        self._chat = chat
        self.adapter_path = str(adapter_path) if adapter_path else None
        self._load_in_4bit = load_in_4bit
        self._device_map = device_map
        self._dtype = dtype
        self._model = None
        self._tokenizer = None

    # -- lazy loading so importing the module costs nothing -------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.hf_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        load_kwargs: dict = {
            "device_map": self._device_map,
            "torch_dtype": getattr(torch, self._dtype),
        }
        if self._load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self._dtype),
                bnb_4bit_quant_type="nf4",
            )
        model = AutoModelForCausalLM.from_pretrained(self.hf_id, **load_kwargs)

        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._model, self._tokenizer = model, tok

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def supports_prefill(self) -> bool:
        return True

    # -- generation -----------------------------------------------------------
    def _generate(
        self,
        input_ids,
        n: int,
        temperature: float | None,
        max_new_tokens: int | None,
    ) -> list[GenerationResult]:
        import torch

        self._ensure_loaded()
        temp = self.temperature if temperature is None else temperature
        mnt = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        prompt_len = input_ids.shape[-1]

        with torch.no_grad():
            out = self._model.generate(
                input_ids=input_ids.to(self._model.device),
                do_sample=temp > 0,
                temperature=max(temp, 1e-6),
                top_p=1.0,           # paper varies nothing but temperature
                top_k=0,
                max_new_tokens=mnt,
                num_return_sequences=n,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        results = []
        for seq in out:
            gen_ids = seq[prompt_len:]
            text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
            finished = (
                "stop"
                if gen_ids[-1].item() in (self._tokenizer.eos_token_id or -1,)
                else "length"
            )
            results.append(
                GenerationResult(
                    text=text,
                    finish_reason=finished,
                    prompt_tokens=int(prompt_len),
                    completion_tokens=int(gen_ids.shape[-1]),
                )
            )
        return results

    def chat(
        self,
        messages: list[Message],
        n: int = 1,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[GenerationResult]:
        self._ensure_loaded()
        if not self._chat:
            # Base models have no chat template; the caller should use complete().
            raise RuntimeError(
                f"{self.name} is a base model; use complete() with an explicit "
                "chat-formatted or prefilled prompt instead of chat()."
            )
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return self._generate(input_ids, n, temperature, max_new_tokens)

    def complete(
        self,
        prompt: str,
        n: int = 1,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[GenerationResult]:
        """Continue a raw prompt. `prompt` already contains any prefilled text;
        only the newly generated continuation is returned."""
        self._ensure_loaded()
        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
        return self._generate(input_ids, n, temperature, max_new_tokens)
