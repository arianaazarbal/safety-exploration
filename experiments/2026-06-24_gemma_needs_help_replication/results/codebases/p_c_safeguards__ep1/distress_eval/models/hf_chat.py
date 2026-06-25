"""Local HuggingFace backend for Gemma 3 (instruct + base/pretrained), with
optional LoRA adapters for the Section 4 finetunes.

Handles three needs from the paper:
  * Standard multi-turn instruct rollouts (Section 2).
  * Base-model continuation from a prefill (Section 3): base models lack a chat
    template, so the conversation is rendered as plain text and the model
    continues from the prefilled assistant text.
  * Returning *new-token* continuations only, so prefilled text is excluded from
    what gets scored.

Heavy deps (torch/transformers/peft) are imported lazily so the API-only parts
of the harness import this module without paying for them.
"""
from __future__ import annotations

from typing import Optional

from .. import config
from .base import ChatModel, GenerationConfig, Message


def _select_dtype():
    import torch
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# Plain-text rendering used for *base* models (no chat template). Mirrors the
# Gemma turn structure loosely but in free text so a pretrained model will
# continue rather than expect special tokens. See DESIGN.md "Base-model prompts".
def _render_plain(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        role = {"user": "User", "assistant": "Assistant", "system": "System"}.get(m["role"], m["role"])
        lines.append(f"{role}: {m['content']}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def _fold_system(messages: list[Message]) -> list[Message]:
    """Merge a leading system message into the first user turn (for chat
    templates that don't accept a standalone system role)."""
    if not messages or messages[0]["role"] != "system":
        return messages
    sys = messages[0]["content"]
    rest = messages[1:]
    out = []
    folded = False
    for m in rest:
        if not folded and m["role"] == "user":
            out.append({"role": "user", "content": f"{sys}\n\n{m['content']}"})
            folded = True
        else:
            out.append(m)
    if not folded:  # no user turn yet
        out.insert(0, {"role": "user", "content": sys})
    return out


class HFChatModel(ChatModel):
    def __init__(
        self,
        key: str,
        model_id: str,
        *,
        is_base: bool = False,
        adapter_path: Optional[str] = None,
        device_map: str = "auto",
        attn_implementation: str | None = None,
    ):
        import torch
        from transformers import AutoTokenizer

        self.key = key
        self.model_id = model_id
        self.is_base = is_base
        self.adapter_path = adapter_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding is required for correct batched generation.
        self.tokenizer.padding_side = "left"

        self.model = self._load_model(model_id, device_map, attn_implementation)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()  # fold LoRA in for fast inference
        self.model.eval()
        self._torch = torch

    @staticmethod
    def _load_model(model_id, device_map, attn_implementation):
        """Gemma 3 ships as a conditional-generation (multimodal) class; the
        text-only causal class also exists for some sizes. Try the auto classes
        in order of preference and keep whichever loads."""
        import torch
        kwargs = dict(torch_dtype=_select_dtype(), device_map=device_map)
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        errors = []
        from transformers import AutoModelForCausalLM
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except Exception as e:  # pragma: no cover - depends on transformers version
            errors.append(f"AutoModelForCausalLM: {e}")
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        except Exception as e:  # pragma: no cover
            errors.append(f"AutoModelForImageTextToText: {e}")
        raise RuntimeError(f"Could not load {model_id}:\n" + "\n".join(errors))

    # ------------------------------------------------------------------ #
    def _render(self, messages: list[Message], prefill: str | None) -> str:
        if self.is_base:
            text = _render_plain(messages)
            if prefill:
                text = text + " " + prefill if not text.endswith("\n") else text + prefill
            return text
        # Instruct: use the model's own chat template. Some Gemma chat templates
        # reject a standalone system role; fall back to folding it into the
        # first user turn.
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = self.tokenizer.apply_chat_template(
                _fold_system(messages), tokenize=False, add_generation_prompt=True
            )
        if prefill:
            text = text + prefill
        return text

    def _complete_batch(self, prompts: list[str], gen: GenerationConfig) -> list[str]:
        torch = self._torch
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        do_sample = gen.temperature and gen.temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=gen.max_new_tokens,
                do_sample=do_sample,
                temperature=gen.temperature if do_sample else None,
                top_p=gen.top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_tokens = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------ #
    def generate(self, messages, *, prefill=None, gen=None):
        gen = gen or GenerationConfig()
        if gen.seed is not None:
            self._torch.manual_seed(gen.seed)
        prompt = self._render(messages, prefill)
        return self._complete_batch([prompt], gen)[0].strip()

    def generate_batch(self, batch, *, prefills=None, gen=None):
        gen = gen or GenerationConfig()
        if gen.seed is not None:
            self._torch.manual_seed(gen.seed)
        prefills = prefills or [None] * len(batch)
        prompts = [self._render(m, p) for m, p in zip(batch, prefills)]
        return [t.strip() for t in self._complete_batch(prompts, gen)]

    def close(self):
        try:
            del self.model
            self._torch.cuda.empty_cache()
        except Exception:
            pass
