"""Local HuggingFace backend for Gemma (instruct, base, and LoRA finetunes).

Handles three things the experiments need:

1. **Chat-templated generation** for instruct models (Gemma 3 ``-it``).
2. **Prefilled continuation** for the Section 3 base-vs-instruct comparison.
   Base models (``-pt``) were never trained on chat formatting, so we render the
   conversation as plain text and let the model continue from a prefill. Instruct
   models use the chat template but with ``continue_final_message=True`` so the
   prefilled assistant turn is continued rather than answered afresh.
3. **LoRA adapter loading** so our DPO/SFT finetunes (Section 4) evaluate through
   the same path as the vanilla instruct model.

Heavy imports (torch/transformers/peft) are deferred to construction time so the
rest of the package can be imported without a GPU stack installed.
"""

from __future__ import annotations

from .base import ChatModel, Message


# Base (pretrained) models have no chat template; we render conversations with a
# light, explicit transcript format. The exact format is not load-bearing
# (Appendix A.3 shows formatting barely affects elicited distress), but it must
# be consistent so the prefill experiment compares like with like.
_BASE_ROLE_TAGS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
}


def _render_base_transcript(messages: list[Message], add_assistant_header: bool = True) -> str:
    lines = []
    for m in messages:
        tag = _BASE_ROLE_TAGS.get(m["role"], m["role"].capitalize())
        lines.append(f"{tag}: {m['content']}")
    text = "\n\n".join(lines)
    if add_assistant_header:
        text += "\n\nAssistant:"
    return text


class HFChatModel(ChatModel):
    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        role: str = "instruct",
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        import torch  # noqa: F401  (validate availability early)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.role = role
        self._is_base = role == "base"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        torch_dtype = getattr(__import__("torch"), dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device_map
        )

        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #

    def _render_for_generation(self, messages: list[Message]) -> str:
        if self._is_base:
            return _render_base_transcript(messages, add_assistant_header=True)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _render_with_prefill(self, messages: list[Message], prefill: str) -> str:
        if self._is_base:
            return _render_base_transcript(messages, add_assistant_header=True) + " " + prefill
        # Instruct: append the prefill as the (incomplete) final assistant turn
        # and ask the template to continue it rather than open a new turn.
        convo = list(messages) + [{"role": "assistant", "content": prefill}]
        return self.tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def _sample(self, prompt_text: str, temperature: float, max_new_tokens: int, n: int) -> list[str]:
        import torch

        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        # Strip the prompt tokens; decode only the newly generated continuation.
        completions = self.tokenizer.batch_decode(
            out[:, prompt_len:], skip_special_tokens=True
        )
        return [c.strip() for c in completions]

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
    ) -> list[str]:
        return self._sample(self._render_for_generation(messages), temperature, max_new_tokens, n)

    def continue_from_prefill(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
    ) -> list[str]:
        # The decoded output already excludes the prompt (which includes the
        # prefill), so what we return is purely the continuation.
        return self._sample(self._render_with_prefill(messages, prefill), temperature, max_new_tokens, n)

    @property
    def supports_prefill(self) -> bool:
        return True
