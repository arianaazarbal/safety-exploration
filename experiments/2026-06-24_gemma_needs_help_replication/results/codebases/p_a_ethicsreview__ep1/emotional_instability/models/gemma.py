"""Local Gemma backend (HuggingFace transformers).

Handles both instruct (``-it``) and base (``-pt``) checkpoints. Instruct models
use the chat template; base models, which are not chat-tuned, are driven purely
by prefilled text continuation (the mechanism the paper uses for the
base-vs-instruct comparison in Section 3).

This module is import-light at module scope: heavy imports (torch,
transformers) happen inside ``__init__`` so the rest of the package can be
imported on machines without a GPU stack.
"""

from __future__ import annotations

from typing import Any

from .base import ChatClient, Message


class GemmaClient(ChatClient):
    def __init__(
        self,
        hf_id: str,
        *,
        role: str = "instruct",
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        device_map: str = "auto",
        adapter_path: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = hf_id if adapter_path is None else f"{hf_id}+{adapter_path}"
        self.hf_id = hf_id
        self.role = role
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)

        model_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "torch_dtype": getattr(torch, dtype),
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)

        # Optionally attach a trained LoRA adapter (the DPO/SFT models).
        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    # ---- core sampling -----------------------------------------------------
    def _generate(self, input_ids, *, temperature: float, max_new_tokens: int):
        torch = self._torch
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = self.model.generate(input_ids, **gen_kwargs)
        # Return only the newly generated tokens.
        new_tokens = out[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _encode_chat(self, messages: list[Message], *, add_generation_prompt: bool):
        """Render messages with the chat template and return input_ids on device."""
        ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
        return ids.to(self.model.device)

    def chat(self, messages: list[Message], *, temperature: float, max_new_tokens: int) -> str:
        if self.role == "base":
            # Base checkpoints have no chat template. We linearise the
            # conversation into plain text and let the model continue. In
            # practice the rollout engine drives base models through
            # `continue_text` (prefilling); this path is a sensible fallback.
            text = _linearise_plain(messages)
            ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
            return self._generate(ids, temperature=temperature, max_new_tokens=max_new_tokens)
        ids = self._encode_chat(messages, add_generation_prompt=True)
        return self._generate(ids, temperature=temperature, max_new_tokens=max_new_tokens)

    # ---- prefilled continuation (Section 3) --------------------------------
    def continue_text(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        if self.role == "base":
            # Base model: feed conversation-as-text + the prefilled assistant
            # start, then continue from there.
            prompt = _linearise_plain(messages) + prefill
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        else:
            # Instruct model: open the assistant turn via the chat template,
            # then append the prefill tokens so the model continues mid-turn.
            chat_ids = self._encode_chat(messages, add_generation_prompt=True)
            prefill_ids = self.tokenizer(
                prefill, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.model.device)
            ids = self._torch.cat([chat_ids, prefill_ids], dim=1)
        return self._generate(ids, temperature=temperature, max_new_tokens=max_new_tokens)

    # ---- tokenisation helpers ---------------------------------------------
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids[:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    # ---- introspection for internal-emotion probing (Section 4.2) ----------
    def residual_at_layer(self, text: str, layer: int):
        """Return the last-token hidden state at ``layer`` for ``text``.

        Used by the logit-lens internal-emotion measurement. Returns a 1-D
        tensor of size hidden_dim.
        """
        torch = self._torch
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True)
        # hidden_states[0] is the embedding output; layer L is index L+1.
        return out.hidden_states[layer + 1][0, -1, :]


def _linearise_plain(messages: list[Message]) -> str:
    """Render a conversation as plain prefixed text for base-model continuation.

    Base checkpoints were never trained on a chat format, so we use a simple,
    explicit transcript format. The exact framing is a documented choice
    (see DESIGN.md); what matters for Section 3 is that base and instruct
    models continue from the *same paraphrased prefill text*, which the prefill
    pipeline guarantees.
    """
    parts = []
    for m in messages:
        prefix = {"system": "Instructions", "user": "User", "assistant": "Assistant"}.get(
            m["role"], m["role"].capitalize()
        )
        parts.append(f"{prefix}: {m['content']}")
    parts.append("Assistant:")
    return "\n".join(parts) + " "
