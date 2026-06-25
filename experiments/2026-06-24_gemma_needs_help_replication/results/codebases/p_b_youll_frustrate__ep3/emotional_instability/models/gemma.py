"""Local HuggingFace client for Gemma 3 (instruct ``-it`` and base ``-pt``).

Responsibilities beyond plain chat:

* **Prefilling** (Section 3): for instruct models we build the chat template and
  append ``prefill`` as the start of the assistant turn; for base models, which
  have no chat template, we render the conversation as plain prefixed text and
  let the model continue. In both cases we return only the newly generated text.
* **Hidden states** (Appendix I): :meth:`forward_hidden_states` returns
  per-layer residual-stream activations for the internal-emotion logit probe.

Heavy deps (torch, transformers) are imported lazily so the rest of the package
is usable without a GPU/model present.
"""

from __future__ import annotations

from typing import List, Optional

from .base import ChatMessage, GenerationConfig, ModelClient

# Base (pretrained) checkpoints have no instruction/chat formatting.
BASE_SUFFIXES = ("-pt", "-base")


def is_base_checkpoint(model_id: str) -> bool:
    return any(model_id.endswith(s) for s in BASE_SUFFIXES)


class GemmaClient(ModelClient):
    def __init__(
        self,
        model: str,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model
        self.model_id = model
        self.is_base = is_base_checkpoint(model)

        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            **quant_kwargs,
        )
        if adapter_path:
            # Attach a trained LoRA adapter (DPO/SFT finetune).
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self._torch = torch

    # ---- prompt construction -------------------------------------------- #

    def _render_base(self, messages: List[ChatMessage]) -> str:
        """Plain-text rendering for base models (no chat template)."""
        parts = []
        for m in messages:
            tag = {"system": "Instructions", "user": "User", "assistant": "Assistant"}[m.role]
            parts.append(f"{tag}: {m.content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    @staticmethod
    def _merge_system(messages: List[ChatMessage]) -> List[ChatMessage]:
        """Gemma's chat template has no separate system role; fold any system
        message into the first user turn (the Gemma convention)."""
        sys_text = " ".join(m.content for m in messages if m.role == "system").strip()
        rest = [m for m in messages if m.role != "system"]
        if not sys_text:
            return rest
        merged: List[ChatMessage] = []
        injected = False
        for m in rest:
            if not injected and m.role == "user":
                merged.append(ChatMessage("user", f"{sys_text}\n\n{m.content}"))
                injected = True
            else:
                merged.append(m)
        if not injected:  # no user turn yet
            merged.insert(0, ChatMessage("user", sys_text))
        return merged

    def _build_inputs(self, messages: List[ChatMessage], prefill: str = ""):
        if self.is_base:
            text = self._render_base(messages)
            if prefill:
                text = text + " " + prefill
            return self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # Instruct: use the official chat template (system folded into user).
        chat = self._merge_system(messages)
        rendered = self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in chat],
            tokenize=False,
            add_generation_prompt=True,
        )
        if prefill:
            rendered = rendered + prefill  # continue inside the assistant turn
        return self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )

    # ---- generation ------------------------------------------------------ #

    def _generate(self, inputs, cfg: GenerationConfig) -> str:
        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            temperature=cfg.temperature if cfg.temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        with self._torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def chat(self, messages: List[ChatMessage], cfg: GenerationConfig) -> str:
        return self._generate(self._build_inputs(messages), cfg)

    def chat_prefill(
        self, messages: List[ChatMessage], prefill: str, cfg: GenerationConfig
    ) -> str:
        # Returns only the continuation (prefill is in the prompt, not the output).
        return self._generate(self._build_inputs(messages, prefill=prefill), cfg)

    # ---- interpretability (Appendix I) ---------------------------------- #

    def forward_hidden_states(self, text: str):
        """Run a forward pass over rendered ``text`` and return
        ``(hidden_states, input_ids)`` where ``hidden_states`` is a tuple of
        per-layer residual-stream tensors. Used by the logit-based emotion
        detector to read internal states at each layer/token."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states, inputs["input_ids"][0]

    def render_conversation(self, messages: List[ChatMessage]) -> str:
        if self.is_base:
            return self._render_base(messages)
        return self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in self._merge_system(messages)],
            tokenize=False,
            add_generation_prompt=False,
        )
