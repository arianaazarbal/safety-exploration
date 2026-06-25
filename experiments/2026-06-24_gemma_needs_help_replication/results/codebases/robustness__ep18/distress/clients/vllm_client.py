"""Local Gemma inference via vLLM (offline ``LLM`` engine).

Handles three cases needed by the experiments:
  * instruct chat generation (Gemma chat template);
  * assistant prefill continuation (instruct) -- continue from fixed opening text;
  * base/pretrained continuation -- raw text completion with no chat template
    (used for Section 3 base-model comparisons).

LoRA adapters (the DPO/SFT finetunes) are applied via vLLM's LoRA support when
``ModelConfig.lora_path`` is set.
"""

from __future__ import annotations

from ..config import ModelConfig
from .base import GenConfig, Message


class VLLMClient:
    def __init__(self, model: ModelConfig):
        from vllm import LLM
        from transformers import AutoTokenizer

        self.cfg = model
        self.name = model.name
        self.is_base = model.is_base

        self._lora_request = None
        enable_lora = model.lora_path is not None
        if enable_lora:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("adapter", 1, model.lora_path)

        self.llm = LLM(
            model=model.model_id,
            tensor_parallel_size=model.tensor_parallel_size,
            max_model_len=model.max_model_len,
            dtype=model.dtype,
            enable_lora=enable_lora,
            max_lora_rank=64,
            trust_remote_code=True,
        )
        # Tokenizer used for applying the chat template ourselves so we can do
        # assistant prefill (append text after the generation prompt).
        self.tokenizer = AutoTokenizer.from_pretrained(model.model_id)

    # ----- prompt construction ------------------------------------------- #
    def _render_chat(self, messages: list[Message], prefill: str | None = None) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if prefill:
            text = text + prefill
        return text

    def _render_base(self, messages: list[Message], prefill: str | None = None) -> str:
        """For base models: a lightweight transcript with no special tokens, so
        the model simply continues the assistant's text (Section 3.1)."""
        parts = []
        for m in messages:
            role = m["role"].capitalize()
            parts.append(f"{role}: {m['content']}")
        parts.append("Assistant:" + (f" {prefill}" if prefill else ""))
        return "\n\n".join(parts)

    def _sampling_params(self, cfg: GenConfig, n: int):
        from vllm import SamplingParams

        return SamplingParams(
            n=n,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop,
            seed=cfg.seed,
        )

    def _run(self, prompt: str, cfg: GenConfig, n: int) -> list[str]:
        kwargs = {}
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        outs = self.llm.generate([prompt], self._sampling_params(cfg, n), **kwargs)
        return [o.text for o in outs[0].outputs]

    # ----- public API ---------------------------------------------------- #
    def generate(self, messages: list[Message], cfg: GenConfig, n: int = 1) -> list[str]:
        prompt = self._render_base(messages) if self.is_base else self._render_chat(messages)
        return self._run(prompt, cfg, n)

    def continue_from_prefill(
        self, messages: list[Message], prefill: str, cfg: GenConfig, n: int = 1
    ) -> list[str]:
        render = self._render_base if self.is_base else self._render_chat
        prompt = render(messages, prefill=prefill)
        # vLLM returns only the continuation (prefill is part of the prompt).
        return self._run(prompt, cfg, n)


def batch_generate_chat(
    client: "VLLMClient", prompts: list[list[Message]], cfg: GenConfig, n: int = 1
) -> list[list[str]]:
    """Throughput helper: render many conversations and run one vLLM batch.

    Returns, per input conversation, a list of `n` completions. Used by the
    rollout engine to keep the GPU busy across a whole turn of many rollouts.
    """
    rendered = [
        client._render_base(m) if client.is_base else client._render_chat(m)
        for m in prompts
    ]
    kwargs = {}
    if client._lora_request is not None:
        kwargs["lora_request"] = client._lora_request
    outs = client.llm.generate(rendered, client._sampling_params(cfg, n), **kwargs)
    return [[o.text for o in out.outputs] for out in outs]
