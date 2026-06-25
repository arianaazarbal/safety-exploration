"""Local HuggingFace inference for Gemma-3 instruct models.

Matches the paper's local-inference setup (Appendix B.1: google/gemma-3-27b-it,
google/gemma-3-12b-it). Heavyweight: requires a GPU and `torch`/`transformers`/
`accelerate`. Imports are deferred so the rest of the pipeline (Gemini via API,
judges) works without these installed.

Gemma-3 instruct uses a chat template with no system role; a leading system
message is folded into the first user turn by the tokenizer's chat template, so
we pass messages through `apply_chat_template` directly.

NOTE on throughput: transformers generation is slow for 4000 rollouts. For a
real full run, serving Gemma with vLLM (an OpenAI-compatible server) and pointing
`OpenRouterClient`-style access at it is far faster. This class is the simple,
dependency-light reference implementation. See DESIGN.md.
"""

from __future__ import annotations

from config import GENERATION, env
from models.base import ChatModel, Message


class GemmaHFClient(ChatModel):
    def __init__(self, name: str, model_id: str, **_ignored):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "pip install torch transformers accelerate to use the hf backend"
            ) from e

        self.name = name
        self.model_id = model_id
        self._torch = torch

        token = env("HF_TOKEN")
        dtype = getattr(torch, GENERATION.hf_dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=GENERATION.hf_device_map,
            token=token,
        )
        self._model.eval()
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _render(self, messages: list[Message]) -> str:
        return self._tokenizer.apply_chat_template(
            [m.to_dict() for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        return self.generate_batch(
            [messages], temperature=temperature, max_tokens=max_tokens
        )[0]

    def generate_batch(
        self,
        conversations: list[list[Message]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        torch = self._torch
        prompts = [self._render(c) for c in conversations]
        enc = self._tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self._model.device)
        input_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=1.0,  # pure temperature sampling (no nucleus truncation)
                top_k=0,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        gen = out[:, input_len:]
        return [
            self._tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen
        ]
