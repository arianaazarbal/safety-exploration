"""Local HuggingFace transformers client for Gemma.

Required for experiments that cloud APIs cannot serve:
  * Section 3 base/pretrained Gemma continuations from a prefill (base models are
    not on OpenRouter and need raw prefix continuation without chat templating).
  * Section 4 evaluation of LoRA-finetuned adapters.

Instruct models apply the Gemma chat template; base/pretrained models concatenate
content directly (prefill semantics) since they were never trained on chat turns.
For high-throughput sampling consider swapping this for a vLLM backend with the
same interface; transformers is used here for portability.
"""
from __future__ import annotations

from .base import ChatMessage, GenerationResult, ModelClient, SamplingParams


class LocalHFClient(ModelClient):
    def __init__(
        self,
        model_name: str,
        hf_id: str,
        is_instruct: bool = True,
        adapter_path: str | None = None,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
    ):
        super().__init__(model_name)
        self.hf_id = hf_id
        self.is_instruct = is_instruct
        self.adapter_path = adapter_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None

    # ---- lazy model loading ----------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.torch_dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, torch_dtype=dtype, device_map=self.device_map
        )
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._model = model

    # ---- prompt construction ---------------------------------------------
    def _render_prompt(self, messages: list[ChatMessage], prefill: str | None) -> str:
        """Turn a conversation into the exact input string for the model.

        Instruct: use the tokenizer chat template, with generation prompt, then
        append any prefill so generation continues from it.
        Base: there is no chat template; we render a plain transcript and let the
        model continue. This matches Section 3, where base models "consistently
        continue the response" from a prefilled prefix.
        """
        tok = self._tokenizer
        if self.is_instruct and tok.chat_template:
            rendered = tok.apply_chat_template(
                [m.to_openai() for m in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Plain transcript for base models.
            parts = []
            for m in messages:
                prefix = {"system": "", "user": "User: ", "assistant": "Assistant: "}[m.role]
                parts.append(f"{prefix}{m.content}")
            parts.append("Assistant: ")
            rendered = "\n\n".join(parts)
        if prefill:
            rendered = rendered + prefill
        return rendered

    def chat(self, messages: list[ChatMessage], params: SamplingParams) -> GenerationResult:
        self._ensure_loaded()
        import torch

        prompt = self._render_prompt(messages, params.prefill)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                do_sample=params.temperature > 0,
                temperature=params.temperature,
                top_p=params.top_p,
                max_new_tokens=params.max_tokens,
            )
        # Decode ONLY the newly generated tokens (exclude prompt + prefill).
        gen_ids = out[0][input_len:]
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            model=self.hf_id,
            finish_reason="stop",
            prefill=params.prefill or "",
        )

    def chat_batch(
        self, conversations: list[list[ChatMessage]], params: SamplingParams
    ) -> list[GenerationResult]:
        self._ensure_loaded()
        import torch

        prompts = [self._render_prompt(c, params.prefill) for c in conversations]
        tok = self._tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        enc = tok(prompts, return_tensors="pt", padding=True).to(self._model.device)
        input_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                do_sample=params.temperature > 0,
                temperature=params.temperature,
                top_p=params.top_p,
                max_new_tokens=params.max_tokens,
            )
        results = []
        for i in range(out.shape[0]):
            gen_ids = out[i][input_len:]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            results.append(
                GenerationResult(text=text, model=self.hf_id, prefill=params.prefill or "")
            )
        return results
