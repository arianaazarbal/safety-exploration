"""Local Gemma-3 inference via HuggingFace transformers.

Handles both instruct (`-it`, chat template) and base/pretrained (`-pt`, raw
prefill) variants, optional LoRA adapters (for evaluating finetuned models), and
exposes the underlying model+tokenizer for the internal-emotion-detection code
(Appendix I).

Prefilling (PAPER 3.1): for instruct models we apply the chat template with
`continue_final_message=True` so the model continues an assistant-turn prefix;
for base models there is no chat template, so we render the conversation as
plain text and let the model continue. In both cases the returned string
excludes the prefill — only newly generated text is scored.
"""

from __future__ import annotations

from typing import Optional

from .base import ChatModel, Message


class GemmaModel(ChatModel):
    def __init__(
        self,
        hf_id: str,
        name: str,
        *,
        is_instruct: bool = True,
        adapter_path: Optional[str] = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        load_kwargs: Optional[dict] = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.hf_id = hf_id
        self.is_instruct = is_instruct
        self.adapter_path = adapter_path

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch_dtype, device_map=device_map,
            **(load_kwargs or {}),
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------
    @staticmethod
    def _fold_system(messages: list[Message]) -> list[Message]:
        """Gemma-3's chat template has no dedicated system role; fold any system
        message into the start of the first user turn (separated by a blank
        line). This matches how the paper adds a "reassuring prefix … to the
        initial prompt" (PAPER 4.1) rather than via a system turn, and keeps the
        teacher-SFT system prompt usable on Gemma. See DESIGN.md."""
        system_chunks = [m["content"] for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        if not system_chunks:
            return rest
        prefix = "\n\n".join(system_chunks)
        for i, m in enumerate(rest):
            if m["role"] == "user":
                folded = list(rest)
                folded[i] = {"role": "user", "content": f"{prefix}\n\n{m['content']}"}
                return folded
        # No user turn (shouldn't happen); prepend a user turn carrying the system text.
        return [{"role": "user", "content": prefix}] + rest

    def _render(self, messages: list[Message], prefill: Optional[str]) -> str:
        """Return the full prompt string to tokenize."""
        if self.is_instruct:
            msgs = self._fold_system(list(messages))
            if prefill is not None:
                msgs = msgs + [{"role": "assistant", "content": prefill}]
                text = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, continue_final_message=True,
                )
            else:
                text = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
            return text
        # Base model: no chat template. Render a simple labelled transcript.
        return self._render_plain(messages, prefill)

    @staticmethod
    def _render_plain(messages: list[Message], prefill: Optional[str]) -> str:
        lines = []
        for m in messages:
            if m["role"] == "system":
                lines.append(m["content"])
            elif m["role"] == "user":
                lines.append(f"User: {m['content']}")
            else:
                lines.append(f"Assistant: {m['content']}")
        prompt = "\n".join(lines) + "\nAssistant:"
        if prefill is not None:
            prompt += " " + prefill
        return prompt

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
        prefill: Optional[str] = None,
    ) -> list[str]:
        import torch

        prompt = self._render(messages, prefill)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
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

        # Strip the prompt (incl. any prefill) so only the continuation remains.
        completions = []
        for seq in out:
            new_tokens = seq[prompt_len:]
            completions.append(
                self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            )
        return completions
