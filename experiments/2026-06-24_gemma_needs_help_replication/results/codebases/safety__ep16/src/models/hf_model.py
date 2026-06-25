"""Local HuggingFace inference for Gemma (instruct, base, and LoRA-adapted).

Used for every Gemma target in the paper's scope. The 27B models comfortably fit
on one 80GB GPU in bf16; ``load_in_4bit=True`` is exposed for smaller cards.

Two roles:
  * ``HFChatModel`` — instruct / DPO / SFT checkpoints. Uses the tokenizer chat
    template and (optionally) loads a LoRA adapter on top.
  * ``HFCompletionModel`` — base / pretrained checkpoints (``-pt``). Exposes a
    raw ``complete`` for the Section 3 prefill experiment, *and* a ``generate``
    that wraps prefilling in a hand-built chat-like string so base models can be
    pushed through the same rollout harness when desired.

We deliberately keep model loading lazy and cached so a single process can hold
one large model at a time without re-loading between conditions.
"""

from __future__ import annotations

import functools

import torch

from config import API
from src.models.base import ChatModel, CompletionModel, Message


@functools.lru_cache(maxsize=2)
def _load(model_id: str, load_in_4bit: bool, adapter: str | None):
    """Load (and cache) a tokenizer + model, optionally with a LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, token=API.hf_token)

    kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": "auto", "token": API.hf_token}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()  # fold LoRA in for faster inference

    model.eval()
    return tok, model


def _set_seed(seed: int | None) -> None:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class HFChatModel(ChatModel):
    def __init__(self, name: str, model_id: str, *, adapter: str | None = None, load_in_4bit: bool = False):
        self.name = name
        self.model_id = model_id
        self.adapter = adapter
        self.load_in_4bit = load_in_4bit

    @property
    def _mt(self):
        return _load(self.model_id, self.load_in_4bit, self.adapter)

    def _format(self, messages: list[Message]) -> str:
        tok, _ = self._mt
        # Gemma-3 chat template has no system role; fold any system message into
        # the first user turn (a documented design choice — see DESIGN.md).
        msgs = _fold_system(messages)
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate(self, messages, *, temperature=1.0, top_p=1.0, max_new_tokens=2048, seed=None) -> str:
        tok, model = self._mt
        _set_seed(seed)
        prompt = self._format(messages)
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        out = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(gen, skip_special_tokens=True).strip()

    @torch.no_grad()
    def continue_assistant(self, messages, assistant_prefix, *, temperature=1.0, top_p=1.0,
                           max_new_tokens=512, seed=None) -> str:
        """Continue an assistant turn that begins with ``assistant_prefix``.

        Used by the Section 3 prefill experiment: returns ONLY the newly
        generated continuation (excluding the prefix).
        """
        tok, model = self._mt
        _set_seed(seed)
        prompt = self._format(messages) + assistant_prefix
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        out = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(gen, skip_special_tokens=True)

    def render_plain(self, messages, assistant_prefix="") -> str:
        """Render a conversation as plain text (for prefilling BASE models, which
        have no chat template). Mirrors App. C: prior turns shown as labelled
        text, then the assistant turn continues from ``assistant_prefix``."""
        lines = []
        for m in _fold_system(messages):
            tag = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{tag}: {m['content']}")
        lines.append(f"Assistant: {assistant_prefix}")
        return "\n\n".join(lines)

    @torch.no_grad()
    def generate_batch(self, batch, *, temperature=1.0, top_p=1.0, max_new_tokens=2048, seeds=None) -> list[str]:
        tok, model = self._mt
        if seeds:
            _set_seed(seeds[0])  # batched sampling shares one RNG state
        tok.padding_side = "left"
        prompts = [self._format(m) for m in batch]
        inputs = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        gens = out[:, inputs["input_ids"].shape[1]:]
        return [tok.decode(g, skip_special_tokens=True).strip() for g in gens]


class HFCompletionModel(CompletionModel):
    """Raw completion wrapper for base / pretrained Gemma checkpoints."""

    def __init__(self, name: str, model_id: str, *, load_in_4bit: bool = False):
        self.name = name
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit

    @property
    def _mt(self):
        return _load(self.model_id, self.load_in_4bit, None)

    @torch.no_grad()
    def complete(self, prefix, *, temperature=1.0, top_p=1.0, max_new_tokens=512, seed=None) -> str:
        tok, model = self._mt
        _set_seed(seed)
        inputs = tok(prefix, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(gen, skip_special_tokens=True)


def _fold_system(messages: list[Message]) -> list[Message]:
    """Gemma chat template rejects a 'system' role; prepend it to the first user."""
    if not messages or messages[0]["role"] != "system":
        return messages
    sys = messages[0]["content"]
    rest = messages[1:]
    for i, m in enumerate(rest):
        if m["role"] == "user":
            new = list(rest)
            new[i] = {"role": "user", "content": f"{sys}\n\n{m['content']}"}
            return new
    return rest
