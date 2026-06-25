"""Local high-throughput generation via vLLM.

Used for instruct Gemma targets and finetuned (LoRA) Gemma variants. vLLM gives
us efficient batched sampling at temperature 1, which is the dominant cost of the
4000-rollout-per-model evaluation.

Prefill support
---------------
vLLM does not have a first-class "continue this assistant message" API, so we
build the prompt string ourselves: render the chat template with
``add_generation_prompt=True`` and then append the prefix text. The model then
continues from the prefix; we return only the newly generated tokens. This works
for both instruct (chat-templated) and is also exercised by the prefill
experiment via the HF backend for base models.
"""

from __future__ import annotations

from typing import Sequence

from .base import GenConfig, Message, ModelClient


class VLLMClient(ModelClient):
    supports_prefill = True

    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        adapter_path: str | None = None,
        is_chat: bool = True,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
    ):
        # Imported lazily so the package is importable without a GPU/vLLM present.
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.name = name
        self.is_chat = is_chat
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self._LoRARequest = LoRARequest

        self._lora = None
        if adapter_path is not None:
            self._lora = LoRARequest(adapter_name=name, lora_int_id=1, lora_path=adapter_path)

        self.llm = LLM(
            model=hf_id,
            enable_lora=adapter_path is not None,
            max_lora_rank=64,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
        )

    # -- prompt rendering ------------------------------------------------------
    def _render(self, messages: Sequence[Message], add_generation_prompt: bool = True) -> str:
        if self.is_chat:
            return self.tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
            )
        # Base model: concatenate content as plain text (no chat roles).
        return "".join(m["content"] for m in messages)

    def _sampling_params(self, cfg: GenConfig):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            stop=list(cfg.stop) if cfg.stop else None,
            seed=cfg.seed,
        )

    def _gen_kwargs(self):
        return {"lora_request": self._lora} if self._lora else {}

    def _encode(self, text: str) -> list[int]:
        # The chat template already injects BOS for instruct models, so we must
        # NOT add special tokens again there; base models get the plain text and
        # DO need BOS prepended. Passing token ids (not strings) to vLLM avoids a
        # second tokenisation that would otherwise double the BOS.
        add_special = not self.is_chat
        return self.tokenizer(text, add_special_tokens=add_special)["input_ids"]

    def _run(self, texts: list[str], cfg: GenConfig) -> list[str]:
        prompts = [{"prompt_token_ids": self._encode(t)} for t in texts]
        outs = self.llm.generate(prompts, self._sampling_params(cfg), **self._gen_kwargs())
        return [o.outputs[0].text for o in outs]

    # -- generation ------------------------------------------------------------
    def generate(self, messages: Sequence[Message], cfg: GenConfig) -> str:
        return self.generate_batch([messages], cfg)[0]

    def generate_batch(
        self, batch: Sequence[Sequence[Message]], cfg: GenConfig, max_workers: int = 8
    ) -> list[str]:
        if not batch:
            return []
        return self._run([self._render(m, add_generation_prompt=True) for m in batch], cfg)

    def prefill(self, messages: Sequence[Message], prefix: str, cfg: GenConfig) -> str:
        return self.prefill_batch([(messages, prefix)], cfg)[0]

    def prefill_batch(
        self,
        batch: Sequence[tuple[Sequence[Message], str]],
        cfg: GenConfig,
        max_workers: int = 8,
    ) -> list[str]:
        if not batch:
            return []
        # Render the chat prefix then append the partial assistant text; the model
        # continues from there. vLLM returns only the continuation (prompt not echoed).
        texts = [self._render(m, add_generation_prompt=True) + prefix for m, prefix in batch]
        return self._run(texts, cfg)
