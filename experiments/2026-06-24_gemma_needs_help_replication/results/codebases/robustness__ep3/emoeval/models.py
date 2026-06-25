"""Model backends.

Two interchangeable backends behind a common interface:

* ``APIModel``  - OpenAI-compatible chat completions (OpenRouter by default).
                  Used for Gemini targets and the Claude judge/auditor.
* ``LocalHFModel`` - HuggingFace transformers, used for Gemma (generation,
                  prefill continuation, hidden-state access, LoRA adapters).

Both expose:
    chat(messages, n, temperature, max_new_tokens)            -> list[str]
    continue_(messages, prefill, n, temperature, max_new_tokens) -> list[str]

``continue_`` implements the response *prefilling* used in Section 3 and the
recovery experiment: the assistant turn is started with ``prefill`` text and the
model continues from there. Only the continuation (excluding the prefill) is
returned.

Heavy ML imports (torch/transformers) are deferred so that API-only and
analysis workflows do not require a GPU stack.
"""
from __future__ import annotations

import os
from typing import Optional, Protocol

from .config import (
    API_BASE_URL,
    API_KEY_ENV,
    ModelSpec,
    THINKING,
)
from .utils import Message, to_openai_messages, with_retries


class ChatModel(Protocol):
    name: str

    def chat(self, messages: list[Message], n: int = 1, temperature: float = 1.0,
             max_new_tokens: int = 2048) -> list[str]: ...

    def continue_(self, messages: list[Message], prefill: str, n: int = 1,
                  temperature: float = 1.0, max_new_tokens: int = 2048) -> list[str]: ...


# --------------------------------------------------------------------------- #
# API backend (Gemini + Claude judge), OpenAI-compatible
# --------------------------------------------------------------------------- #
class APIModel:
    """Chat model served over an OpenAI-compatible endpoint (OpenRouter)."""

    def __init__(self, spec: ModelSpec, base_url: str = API_BASE_URL,
                 api_key_env: str = API_KEY_ENV):
        from openai import OpenAI  # deferred import

        self.spec = spec
        self.name = spec.name
        api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"No API key found in ${api_key_env} (or $OPENAI_API_KEY). "
                "Set it to call API models."
            )
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def _extra_body(self) -> dict:
        # Disable provider-side reasoning where supported (Appendix B.1).
        if THINKING:
            return {}
        return {"reasoning": {"enabled": False}}

    def chat(self, messages: list[Message], n: int = 1, temperature: float = 1.0,
             max_new_tokens: int = 2048) -> list[str]:
        def _call() -> list[str]:
            resp = self._client.chat.completions.create(
                model=self.spec.model_id,
                messages=to_openai_messages(messages),
                n=n,
                temperature=temperature,
                max_tokens=max_new_tokens,
                extra_body=self._extra_body(),
            )
            return [c.message.content or "" for c in resp.choices]

        return with_retries(_call)

    def continue_(self, messages: list[Message], prefill: str, n: int = 1,
                  temperature: float = 1.0, max_new_tokens: int = 2048) -> list[str]:
        """Prefill an assistant turn. Implemented by appending an assistant
        message holding the prefill; OpenRouter passes this through as an
        assistant-prefix for providers that support continuation. The returned
        text is the continuation only.

        Note: not all API providers honour assistant prefixing. For Gemini this
        is best-effort; the local backend gives exact control and is what the
        paper's Section 3 (Gemma base/instruct) relies on.
        """
        msgs = list(messages) + [Message("assistant", prefill)]

        def _call() -> list[str]:
            resp = self._client.chat.completions.create(
                model=self.spec.model_id,
                messages=to_openai_messages(msgs),
                n=n,
                temperature=temperature,
                max_tokens=max_new_tokens,
                extra_body=self._extra_body(),
            )
            outs = []
            for c in resp.choices:
                text = c.message.content or ""
                # Strip any echoed prefill so we return continuation only.
                outs.append(text[len(prefill):] if text.startswith(prefill) else text)
            return outs

        return with_retries(_call)


# --------------------------------------------------------------------------- #
# Local HuggingFace backend (Gemma)
# --------------------------------------------------------------------------- #
class LocalHFModel:
    """HuggingFace transformers backend with prefill + hidden-state support."""

    def __init__(self, spec: ModelSpec, adapter_path: Optional[str] = None,
                 dtype: str = "bfloat16", device_map: str = "auto",
                 attn_implementation: str = "eager"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.name = spec.name if adapter_path is None else f"{spec.name}+{os.path.basename(adapter_path)}"
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        if adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # -- prompt formatting --------------------------------------------------- #
    def _render(self, messages: list[Message], prefill: Optional[str] = None) -> str:
        """Turn messages into the model's input string.

        Instruct models: use the chat template (with generation prompt, or
        continuing the final assistant message when prefilling).

        Base/pretrained models: no chat template exists, so we use a minimal
        role-tagged transcript. This matches the paper's approach of prefilling
        responses so base models "consistently continue" (Section 3.1).
        """
        if self.spec.is_base:
            return self._render_base(messages, prefill)

        if prefill is None:
            return self.tokenizer.apply_chat_template(
                to_openai_messages(messages), tokenize=False, add_generation_prompt=True,
            )
        # Continue an assistant turn started with `prefill`.
        msgs = to_openai_messages(messages) + [{"role": "assistant", "content": prefill}]
        try:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
                continue_final_message=True,
            )
        except TypeError:
            # Older transformers without continue_final_message: fall back to
            # template-without-prefill + manual append.
            base = self.tokenizer.apply_chat_template(
                to_openai_messages(messages), tokenize=False, add_generation_prompt=True,
            )
            return base + prefill

    def _render_base(self, messages: list[Message], prefill: Optional[str]) -> str:
        lines = []
        for m in messages:
            tag = {"system": "System", "user": "User", "assistant": "Assistant"}.get(m.role, m.role)
            lines.append(f"{tag}: {m.content}")
        lines.append("Assistant:" + (f" {prefill}" if prefill else ""))
        return "\n".join(lines)

    # -- generation ---------------------------------------------------------- #
    def _generate(self, prompt: str, n: int, temperature: float, max_new_tokens: int) -> list[str]:
        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0 if do_sample else None,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    def chat(self, messages: list[Message], n: int = 1, temperature: float = 1.0,
             max_new_tokens: int = 2048) -> list[str]:
        return self._generate(self._render(messages), n, temperature, max_new_tokens)

    def continue_(self, messages: list[Message], prefill: str, n: int = 1,
                  temperature: float = 1.0, max_new_tokens: int = 2048) -> list[str]:
        return self._generate(self._render(messages, prefill=prefill), n, temperature, max_new_tokens)

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def detokenize(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def load_model(spec: ModelSpec, adapter_path: Optional[str] = None, **kwargs) -> ChatModel:
    if spec.backend == "api":
        if adapter_path is not None:
            raise ValueError("Adapters are only supported on the local backend.")
        return APIModel(spec, **kwargs)
    return LocalHFModel(spec, adapter_path=adapter_path, **kwargs)


def load_judge() -> APIModel:
    """Load the Claude-Sonnet-4 judge as an API model."""
    from .config import JUDGE_MODEL_ID
    spec = ModelSpec(name="judge", backend="api", model_id=JUDGE_MODEL_ID, family="judge")
    return APIModel(spec)
