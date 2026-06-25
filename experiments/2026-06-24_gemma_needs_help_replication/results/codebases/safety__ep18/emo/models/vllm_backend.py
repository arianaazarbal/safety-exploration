"""vLLM backend for fast local Gemma generation.

Used for the large elicitation sweeps (Sec 2) and the prefill continuations
(Sec 3), where throughput matters. We render the chat template ourselves and
feed raw prompt strings to vLLM, which lets us also do prefilled continuations
(append the prefill to the prompt; vLLM returns only the generated suffix).

Falls back conceptually to ``HFModel`` if vLLM is unavailable -- the registry
chooses the backend.
"""

from __future__ import annotations

from emo.models.base import ChatModel, GenConfig, Message
from emo.models.hf_local import _build_base_transcript


class VLLMModel(ChatModel):
    supports_prefill = True

    def __init__(
        self,
        name: str,
        model_id: str,
        is_base: bool = False,
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
        dtype: str = "bfloat16",
    ):
        super().__init__(name, is_base=is_base)
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            # Gemma-3 12B/27B are multimodal; restrict to the text path.
            trust_remote_code=True,
        )

    def _sampling_params(self, cfg: GenConfig):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            seed=cfg.seed,
        )

    def _render(self, messages: list[Message]) -> str:
        if self.is_base:
            return _build_base_transcript(messages)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, messages: list[Message], cfg: GenConfig) -> str:
        return self.generate_batch([messages], cfg)[0]

    def generate_batch(self, batch: list[list[Message]], cfg: GenConfig) -> list[str]:
        prompts = [self._render(m) for m in batch]
        outs = self.llm.generate(prompts, self._sampling_params(cfg))
        return [o.outputs[0].text.strip() for o in outs]

    def continue_prefill(
        self, messages: list[Message], prefill: str, cfg: GenConfig
    ) -> str:
        return self.continue_prefill_batch([(messages, prefill)], cfg)[0]

    def continue_prefill_batch(
        self, batch: list[tuple[list[Message], str]], cfg: GenConfig
    ) -> list[str]:
        prompts = [self._render(m) + p for m, p in batch]
        outs = self.llm.generate(prompts, self._sampling_params(cfg))
        return [o.outputs[0].text.strip() for o in outs]
