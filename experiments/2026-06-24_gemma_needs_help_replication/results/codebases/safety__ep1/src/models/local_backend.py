"""vLLM-backed local inference for Gemma (instruct, base, and our LoRA finetunes).

Why vLLM: the headline eval samples ~4000 multi-turn rollouts per model at
temperature 1, which is impractical with eager `transformers.generate`. vLLM
batches the many independent samples (`n>1`, many prompts) efficiently.

Prefilling: we build the prompt string with the tokenizer's chat template
(`add_generation_prompt=True`) and, for prefill, append the seed assistant text
before calling `generate` on the raw string. This makes the model *continue*
the seeded assistant turn rather than starting fresh — exactly what Section 3
and the recovery experiment need. Base (pt) models skip the chat template and
operate on raw text.

LoRA finetunes are served by enabling `enable_lora` and passing a `LoRARequest`.
"""
from __future__ import annotations

from .base import ChatModel, Message
import config


class LocalModel(ChatModel):
    def __init__(self, model_id: str, name: str, is_base: bool = False,
                 adapter_path: str | None = None,
                 tensor_parallel_size: int | None = None,
                 max_model_len: int = 16384):
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer
        import torch

        self.name = name
        self.is_base = is_base
        self.adapter_path = adapter_path
        self._LoRARequest = LoRARequest

        tp = tensor_parallel_size or max(1, torch.cuda.device_count())
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=config.HF_TOKEN or None)
        self.llm = LLM(
            model=model_id,
            tokenizer=model_id,
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            enable_lora=adapter_path is not None,
            max_lora_rank=64,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        self._lora_req = (
            LoRARequest("adapter", 1, adapter_path) if adapter_path else None
        )

    # -- prompt construction --------------------------------------------------
    def _render_chat(self, messages: list[Message], add_generation_prompt: bool,
                     continue_final: bool = False) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final,
        )

    def _sampling_params(self, n, temperature, max_tokens):
        from vllm import SamplingParams
        t, m = self._defaults(temperature, max_tokens)
        return SamplingParams(n=n, temperature=t, top_p=config.SAMPLING.top_p,
                              max_tokens=m)

    def _generate(self, prompt: str, sp) -> list[str]:
        kwargs = {"lora_request": self._lora_req} if self._lora_req else {}
        out = self.llm.generate([prompt], sp, **kwargs)
        return [o.text for o in out[0].outputs]

    # -- API ------------------------------------------------------------------
    def sample_chat(self, messages, n=1, temperature=None, max_tokens=None):
        if self.is_base:
            raise NotImplementedError(
                f"{self.name} is a base model; use sample_with_prefill/sample_completion"
            )
        prompt = self._render_chat(messages, add_generation_prompt=True)
        return self._generate(prompt, self._sampling_params(n, temperature, max_tokens))

    def sample_with_prefill(self, messages, prefill, n=1, temperature=None,
                            max_tokens=None):
        if self.is_base:
            # Base model: flatten the conversation to text + prefill, no template.
            text = _flatten_to_text(messages) + prefill
            return self.sample_completion(text, n, temperature, max_tokens)
        # Instruct model: seed the assistant turn and continue it.
        seeded = list(messages) + [{"role": "assistant", "content": prefill}]
        prompt = self._render_chat(seeded, add_generation_prompt=False,
                                   continue_final=True)
        return self._generate(prompt, self._sampling_params(n, temperature, max_tokens))

    def sample_completion(self, text, n=1, temperature=None, max_tokens=None):
        return self._generate(text, self._sampling_params(n, temperature, max_tokens))

    def sample_chat_batch(self, batch_messages, temperature=None, max_tokens=None):
        """Native vLLM batching: render every conversation to a prompt and submit
        them in a single generate() call so the engine schedules them together."""
        prompts = [self._render_chat(m, add_generation_prompt=True)
                   for m in batch_messages]
        sp = self._sampling_params(1, temperature, max_tokens)
        kwargs = {"lora_request": self._lora_req} if self._lora_req else {}
        out = self.llm.generate(prompts, sp, **kwargs)
        return [o.outputs[0].text for o in out]


def _flatten_to_text(messages: list[Message]) -> str:
    """Plain-text rendering of a conversation for base (non-chat) models.

    Base models have never seen a chat template, so we present the conversation
    as a simple transcript and let the model continue. See DESIGN.md for why this
    matches the paper's "prefilled response" methodology for base models.
    """
    lines = []
    for m in messages:
        tag = {"user": "User", "assistant": "Assistant", "system": "System"}[m["role"]]
        lines.append(f"{tag}: {m['content']}")
    lines.append("Assistant: ")
    return "\n\n".join(lines)
