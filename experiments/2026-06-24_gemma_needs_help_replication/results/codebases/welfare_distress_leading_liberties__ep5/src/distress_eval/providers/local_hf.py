"""Optional local backend for Gemma via vLLM (preferred) or transformers.

This is the closest match to the paper's open-weights setup: you control the
exact checkpoint and sampling. It is only imported when a model is configured
with `provider: local`, so the heavy deps (torch/vllm/transformers) stay
optional. Install them with `pip install -r requirements-local.txt`.

Generation runs in a worker thread (`asyncio.to_thread`) so it cooperates with
the async runner, but a single local model is effectively serialized -- set
`concurrency: 1` per local model and rely on the engine's own batching.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from ..messages import Message
from .base import ChatModel


class LocalHFChatModel(ChatModel):
    def __init__(
        self,
        model_id: str,
        *,
        engine: str = "vllm",
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
    ):
        super().__init__(model_id)
        self.engine = engine
        self._lock = asyncio.Lock()
        if engine == "vllm":
            self._init_vllm(dtype, tensor_parallel_size, max_model_len)
        elif engine == "transformers":
            self._init_transformers(dtype)
        else:
            raise ValueError(f"Unknown local engine {engine!r}")

    def _init_vllm(self, dtype, tp, max_len):
        from vllm import LLM  # type: ignore

        self._llm = LLM(
            model=self.model_id,
            dtype=dtype,
            tensor_parallel_size=tp,
            max_model_len=max_len,
        )

    def _init_transformers(self, dtype):
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=getattr(torch, dtype),
            device_map="auto",
        )

    def _format(self, messages: Sequence[Message]) -> list[dict]:
        # Gemma's chat template merges any system content into the first user
        # turn (handled here so callers stay provider-agnostic).
        sys = "\n\n".join(m.content for m in messages if m.role == "system")
        out: list[dict] = []
        for m in messages:
            if m.role == "system":
                continue
            content = m.content
            if sys and not out and m.role == "user":
                content = f"{sys}\n\n{content}"
                sys = ""
            out.append({"role": m.role, "content": content})
        return out

    def _generate_sync(self, messages, temperature, max_tokens) -> str:
        chat = self._format(messages)
        if self.engine == "vllm":
            from vllm import SamplingParams  # type: ignore

            params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            outputs = self._llm.chat(chat, params)
            return outputs[0].outputs[0].text.strip()
        # transformers
        import torch  # type: ignore

        inputs = self._tok.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
            )
        gen = out[0][inputs.shape[-1] :]
        return self._tok.decode(gen, skip_special_tokens=True).strip()

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        async with self._lock:
            return await asyncio.to_thread(
                self._generate_sync, list(messages), temperature, max_tokens
            )
