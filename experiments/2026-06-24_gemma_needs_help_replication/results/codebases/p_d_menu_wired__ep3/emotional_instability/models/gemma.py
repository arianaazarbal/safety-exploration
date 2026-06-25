"""Local Gemma client (HuggingFace transformers).

Supports both instruct ("-it") and base ("-pt") Gemma 3 models, response
prefilling (needed for the Section 3 prefill experiment and for honouring an
assistant turn that must begin with given text), stop strings, and optional
LoRA adapter loading (to evaluate DPO/SFT-finetuned checkpoints from Section 4).

Models load lazily and are cached per ``model_id`` so repeated client
construction in a run does not reload the (large) 27B weights.
"""
from __future__ import annotations

from typing import Sequence

from .base import ChatMessage, GenerationResult, ModelClient

# Cache of (model_id, adapter_path) -> (model, tokenizer) to avoid reloading.
_MODEL_CACHE: dict = {}


def _load(model_id: str, adapter_path: str | None, dtype: str, device_map: str,
          load_in_4bit: bool):
    cache_key = (model_id, adapter_path, dtype, device_map, load_in_4bit)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}.get(dtype, torch.bfloat16)

    quant_kwargs = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **quant_kwargs,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


class GemmaClient(ModelClient):
    def __init__(self, spec: dict):
        self.key = spec.get("key", spec["model_id"])
        self.model_id = spec["model_id"]
        self.is_instruct = spec.get("is_instruct", True)
        self.supports_prefill = spec.get("supports_prefill", True)
        self.adapter_path = spec.get("adapter_path")  # optional LoRA checkpoint
        self.dtype = spec.get("dtype", "bfloat16")
        self.device_map = spec.get("device_map", "auto")
        self.load_in_4bit = spec.get("load_in_4bit", False)
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model, self._tokenizer = _load(
                self.model_id, self.adapter_path, self.dtype,
                self.device_map, self.load_in_4bit,
            )

    # -- prompt construction ------------------------------------------------
    def _build_prompt(self, messages: Sequence[ChatMessage],
                      prefill: str | None) -> str:
        tok = self._tokenizer
        if self.is_instruct and tok.chat_template:
            # Gemma chat template has no system role; fold any system message
            # into the first user turn (standard Gemma practice).
            chat = _fold_system_into_user([m.as_dict() for m in messages])
            text = tok.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True,
            )
            if prefill:
                # Continue the just-opened assistant turn with the prefill.
                text = text + prefill
            return text
        # Base/pretrained model: no chat template. Render as a plain transcript
        # and let the model continue. Prefilling is just appended text.
        return _render_plaintext_transcript(messages, prefill)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
        tools: Sequence[dict] | None = None,
    ) -> GenerationResult:
        import torch
        self._ensure_loaded()
        tok, model = self._tokenizer, self._model

        prompt_text = self._build_prompt(messages, prefill)
        inputs = tok(prompt_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        stopping_criteria = None
        if stop:
            stopping_criteria = _make_stopping_criteria(tok, list(stop), input_len)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=1.0,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        if stopping_criteria is not None:
            gen_kwargs["stopping_criteria"] = stopping_criteria

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        gen_tokens = out[0][input_len:]
        text = tok.decode(gen_tokens, skip_special_tokens=True)

        stop_reason = "length"
        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
                    stop_reason = "stop"
                    break
        if tok.eos_token and tok.eos_token in tok.decode(gen_tokens):
            stop_reason = "eos" if stop_reason != "stop" else stop_reason

        # The prefill is part of the *assistant turn* but not part of what the
        # model generated this call; callers that prefilled handle prepending.
        return GenerationResult(text=text, stop_reason=stop_reason, raw=None)

    def generate_continuations(
        self,
        messages: Sequence[ChatMessage],
        prefill: str,
        *,
        n: int,
        temperature: float = 1.0,
        max_new_tokens: int = 256,
    ) -> list[str]:
        """Sample ``n`` continuations of a prefilled assistant turn.

        Returns only the generated continuation text (prefill excluded), as
        required by the Section 3 scoring procedure.
        """
        return [
            self.chat(
                messages, prefill=prefill, temperature=temperature,
                max_new_tokens=max_new_tokens,
            ).text
            for _ in range(n)
        ]


def _fold_system_into_user(chat: list[dict]) -> list[dict]:
    if chat and chat[0]["role"] == "system":
        sys = chat[0]["content"]
        rest = chat[1:]
        for i, m in enumerate(rest):
            if m["role"] == "user":
                rest[i] = {"role": "user", "content": f"{sys}\n\n{m['content']}"}
                return rest
        return [{"role": "user", "content": sys}] + rest
    return chat


def _render_plaintext_transcript(messages: Sequence[ChatMessage],
                                 prefill: str | None) -> str:
    lines = []
    for m in messages:
        tag = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
            m.role, m.role.capitalize())
        lines.append(f"{tag}: {m.content}")
    lines.append("Assistant:" + (f" {prefill}" if prefill else ""))
    return "\n".join(lines)


def _make_stopping_criteria(tokenizer, stops: list[str], prompt_len: int):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StopOnStrings(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):  # noqa: D401
            text = tokenizer.decode(input_ids[0][prompt_len:],
                                    skip_special_tokens=True)
            return any(s in text for s in stops)

    return StoppingCriteriaList([_StopOnStrings()])
