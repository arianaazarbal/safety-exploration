"""Gemma subject client via HuggingFace transformers (local weights).

Handles both instruct (chat-templated) and base/pretrained (raw-continuation)
checkpoints, and supports loading a PEFT/LoRA adapter on top of the instruct
model so the DPO/SFT-finetuned variants from Section 4 can be evaluated with
the same code path.

Gemma has no native function calling in the open weights, so the welfare
opt-out is offered via a sentinel string the model is told it may emit; we
parse it out of the generated text.
"""

from __future__ import annotations

import threading

from .base import (
    Conversation,
    SubjectClient,
    SubjectResponse,
    detect_sentinel_optout,
)

_LOAD_LOCK = threading.Lock()


def _import_hf():
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "transformers + torch are required for Gemma subjects: "
            "pip install torch transformers accelerate"
        ) from e
    return AutoModelForCausalLM, AutoTokenizer


class GemmaClient(SubjectClient):
    def __init__(
        self,
        spec,
        *,
        use_base_checkpoint: bool = False,
        adapter_path: str | None = None,
        load_in_4bit: bool = False,
        device_map: str = "auto",
    ):
        import torch

        self.spec = spec
        self.use_base_checkpoint = use_base_checkpoint
        AutoModelForCausalLM, AutoTokenizer = _import_hf()

        model_id = spec.base_model_id if use_base_checkpoint else spec.model_id
        if use_base_checkpoint and model_id is None:
            raise ValueError(f"No base checkpoint configured for {spec.key}")

        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        with _LOAD_LOCK:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                **quant_kwargs,
            )
            if adapter_path:
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()

    # --- prompt construction ---------------------------------------------- #
    def _render_chat(self, conversation: Conversation, prefill: str | None = None) -> str:
        """Render a conversation to a prompt string.

        Instruct models use the chat template. Base models have no chat
        template, so we fall back to a plain transcript — but in practice base
        models are only ever called through ``continue_from_prefill`` with an
        explicit prefill, matching the paper's prefilling protocol.
        """
        if not self.use_base_checkpoint and self.tokenizer.chat_template:
            messages = []
            if conversation.system:
                # Gemma chat template has no system role; fold it into the
                # first user turn (standard Gemma practice).
                messages.append({"role": "user", "content": conversation.system})
                messages.append({"role": "assistant", "content": "Understood."})
            for t in conversation.turns:
                messages.append({"role": t.role, "content": t.content})
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if prefill:
                text = text + prefill
            return text

        # Base checkpoint: plain transcript continuation.
        parts = []
        if conversation.system:
            parts.append(conversation.system)
        for t in conversation.turns:
            speaker = "User" if t.role == "user" else "Assistant"
            parts.append(f"{speaker}: {t.content}")
        parts.append("Assistant:" + (f" {prefill}" if prefill else ""))
        return "\n".join(parts)

    def _generate_raw(self, prompt: str, *, max_tokens: int, temperature: float):
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip(), int(gen_ids.shape[0])

    # --- API --------------------------------------------------------------- #
    def generate(
        self,
        conversation: Conversation,
        *,
        max_tokens: int,
        temperature: float,
        optout_tool: bool = False,        # ignored; Gemma uses the sentinel path
        optout_sentinel: str | None = None,
    ) -> SubjectResponse:
        prompt = self._render_chat(conversation)
        text, n_tokens = self._generate_raw(
            prompt, max_tokens=max_tokens, temperature=temperature
        )
        opted_out, text = detect_sentinel_optout(text, optout_sentinel)
        return SubjectResponse(text=text, opted_out=opted_out, n_tokens=n_tokens)

    def continue_from_prefill(
        self,
        conversation: Conversation,
        prefill: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> SubjectResponse:
        prompt = self._render_chat(conversation, prefill=prefill)
        text, n_tokens = self._generate_raw(
            prompt, max_tokens=max_tokens, temperature=temperature
        )
        return SubjectResponse(text=text, n_tokens=n_tokens)
