"""Local HuggingFace transformers backend for open-weights Gemma models.

Supports both instruct models (chat template) and base/pretrained models
(no chat template — prefill continuation only). This backend is also the one
the probing module (Appendix I) hooks into for residual-stream access, and the
one finetuning loads/saves LoRA adapters against.

Loading is lazy: the (large) weights are only materialised on first use, so
importing this module — or constructing the object to inspect its config — is
cheap. An optional LoRA adapter path can be supplied to evaluate a finetuned
model.
"""
from __future__ import annotations

from typing import Any

from .base import ChatModel, GenerationParams, Message


class HFChatModel(ChatModel):
    def __init__(
        self,
        name: str,
        hf_id: str,
        family: str,
        role: str,
        chat_template: str = "auto",
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        super().__init__(name=name, family=family, role=role)
        self.hf_id = hf_id
        self.chat_template = chat_template
        self.adapter_path = adapter_path
        self.dtype = dtype
        self.device_map = device_map
        self._model: Any = None
        self._tok: Any = None

    # -- lazy loading -------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.dtype)
        self._tok = AutoTokenizer.from_pretrained(self.hf_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, torch_dtype=dtype, device_map=self.device_map
        )
        if self.adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    @property
    def model(self) -> Any:
        self._ensure_loaded()
        return self._model

    @property
    def tokenizer(self) -> Any:
        self._ensure_loaded()
        return self._tok

    # -- prompt formatting --------------------------------------------------
    def _render(self, messages: list[Message], add_generation_prompt: bool) -> str:
        """Render messages to a single prompt string.

        Instruct models use the tokenizer's Gemma chat template. Base models
        ("none") get a plain concatenation, since they were never trained on
        chat turns — this matches the paper's prefill approach for base models.
        """
        if self.chat_template == "none":
            # Base model: plain text. We still mark turns minimally so the
            # model has *some* structure, but rely on prefill to anchor it.
            parts = []
            for m in messages:
                parts.append(m.content)
            return "\n\n".join(parts) + ("\n\n" if add_generation_prompt else "")
        # Gemma 3 has no system role; fold any system content into the first
        # user turn (the standard Gemma convention).
        rendered = self._fold_system(messages)
        return self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in rendered],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @staticmethod
    def _fold_system(messages: list[Message]) -> list[Message]:
        sys = [m for m in messages if m.role == "system"]
        if not sys:
            return messages
        rest = [m for m in messages if m.role != "system"]
        sys_text = "\n\n".join(m.content for m in sys)
        if rest and rest[0].role == "user":
            merged = Message("user", f"{sys_text}\n\n{rest[0].content}")
            return [merged, *rest[1:]]
        return [Message("user", sys_text), *rest]

    # -- generation ---------------------------------------------------------
    def _generate_from_prompt(self, prompt: str, params: GenerationParams) -> str:
        import torch

        self._ensure_loaded()
        inputs = self._tok(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        gen_kwargs = dict(
            max_new_tokens=params.max_new_tokens,
            do_sample=params.temperature > 0,
            temperature=params.temperature,
            top_p=params.top_p,
            pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
        )
        if params.seed is not None:
            torch.manual_seed(params.seed)
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self._tok.decode(new_tokens, skip_special_tokens=True)

    def generate(self, messages: list[Message], params: GenerationParams) -> str:
        prompt = self._render(messages, add_generation_prompt=True)
        return self._generate_from_prompt(prompt, params)

    def continue_from_prefill(
        self, messages: list[Message], prefill: str, params: GenerationParams
    ) -> str:
        # Render the chat up to the generation prompt, then append the prefill
        # so the model continues exactly from where the prefill ends.
        prompt = self._render(messages, add_generation_prompt=True) + prefill
        return self._generate_from_prompt(prompt, params)

    # -- tokenizer helpers (used by prefill truncation + probing) ----------
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)
