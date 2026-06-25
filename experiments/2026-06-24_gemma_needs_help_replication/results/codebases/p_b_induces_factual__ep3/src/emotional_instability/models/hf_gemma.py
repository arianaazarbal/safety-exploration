"""Local HuggingFace backend for Gemma 3 (instruct and base/pretrained).

This backend is the workhorse for everything that needs white-box access:
multi-turn chat (Section 2), assistant-prefill continuation (Section 3 + the
recovery experiment), LoRA-adapter loading (evaluating finetuned models), and
raw forward passes for logit-lens emotion detection (Appendix I).

Model loading is lazy and cached so a single process can hold one Gemma in
memory and reuse it across an entire evaluation sweep.
"""

from __future__ import annotations

from typing import Any

from ..logging_utils import get_logger
from .base import ChatModel, GenConfig, Message

logger = get_logger(__name__)


class HFGemmaModel(ChatModel):
    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        instruct: bool = True,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        super().__init__(name)
        self.hf_id = hf_id
        self.instruct = instruct
        self.adapter_path = adapter_path
        self.dtype = dtype
        self.device_map = device_map
        self._model = None
        self._tokenizer = None

    # -- lazy loading --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading %s (%s)", self.name, self.hf_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            torch_dtype=getattr(torch, self.dtype),
            device_map=self.device_map,
        )
        if self.adapter_path:
            from peft import PeftModel

            logger.info("Attaching LoRA adapter from %s", self.adapter_path)
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    # -- prompt construction -------------------------------------------------

    def _render_prompt(
        self,
        messages: list[Message],
        *,
        prefill: str | None = None,
        use_chat_template: bool = True,
    ) -> str:
        """Render messages to a prompt string.

        For instruct models we apply Gemma's chat template with a generation
        prompt; a non-empty ``prefill`` is appended verbatim after the template's
        assistant-turn opener so the model continues it. For base models we fall
        back to a plain role-tagged concatenation (no special tokens), which is
        how the paper prompts pretrained checkpoints.
        """
        if use_chat_template and self.instruct:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if prefill:
                text = text + prefill
            return text

        # Base-model path: minimal, template-free formatting.
        parts = []
        for m in messages:
            parts.append(f"{m['role'].capitalize()}: {m['content']}")
        parts.append("Assistant:")
        text = "\n\n".join(parts)
        if prefill:
            text = text + " " + prefill
        return text

    def _generate(self, prompt_text: str, gen: GenConfig) -> str:
        import torch

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        do_sample = gen.temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=gen.max_new_tokens,
                do_sample=do_sample,
                temperature=gen.temperature if do_sample else None,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        # Only decode the newly generated tokens (exclude the prompt + prefill).
        new_tokens = out[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # -- ChatModel API -------------------------------------------------------

    def chat(self, messages: list[Message], gen: GenConfig) -> str:
        prompt = self._render_prompt(messages, use_chat_template=True)
        return self._generate(prompt, gen)

    def supports_prefill(self) -> bool:
        return True

    def continue_from(
        self,
        messages: list[Message],
        prefill: str,
        gen: GenConfig,
        *,
        use_chat_template: bool = True,
    ) -> str:
        prompt = self._render_prompt(
            messages, prefill=prefill, use_chat_template=use_chat_template
        )
        return self._generate(prompt, gen)

    # -- tokenization helpers ------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    # -- white-box access (Appendix I) --------------------------------------

    def forward_hidden_states(self, prompt_text: str) -> Any:
        """Run a forward pass returning per-layer hidden states for ``prompt_text``.

        Used by the logit-lens emotion detector. Returns the HuggingFace
        ``CausalLMOutput`` with ``hidden_states`` populated.
        """
        import torch

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            return self.model(**inputs, output_hidden_states=True)
