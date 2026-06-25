"""vLLM backend for fast temperature-1 sampling of open-weight Gemma.

Recommended for Section 2 / Section 4 where we need thousands of rollouts. vLLM
supports assistant-prefill via the ``continue_final_message`` chat option, which
we use for the Section 3 / Section 4.2 prefill experiments.
"""
from __future__ import annotations

from typing import Sequence

from emoinstab.config import ModelSpec
from emoinstab.models.base import Conversation, ModelClient, SamplingParams


class VLLMClient(ModelClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        from vllm import LLM
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        llm_kwargs = dict(
            model=spec.model_id,
            dtype=spec.extra.get("dtype", "bfloat16"),
            tensor_parallel_size=int(spec.extra.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(spec.extra.get("gpu_memory_utilization", 0.9)),
            max_model_len=int(spec.extra.get("max_model_len", 8192)),
        )
        # LoRA adapter support (DPO/SFT checkpoints).
        adapter_dir = spec.extra.get("adapter_dir")
        self._lora_request = None
        if adapter_dir:
            from vllm.lora.request import LoRARequest

            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = 64
            self._lora_request = LoRARequest("adapter", 1, adapter_dir)
        self.llm = LLM(**llm_kwargs)
        self._supports_system = "system" in (getattr(self.tokenizer, "chat_template", "") or "")

    def _sp(self, params: SamplingParams):
        from vllm import SamplingParams as VSP

        return VSP(
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            n=params.n,
            stop=list(params.stop) or None,
            seed=params.seed,
        )

    def _render(self, messages: Conversation, continue_final: bool = False) -> str:
        msgs = [m.as_dict() for m in messages]
        if not self._supports_system and msgs and msgs[0]["role"] == "system":
            sys = msgs.pop(0)["content"]
            for m in msgs:
                if m["role"] == "user":
                    m["content"] = f"{sys}\n\n{m['content']}"
                    break
        return self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=not continue_final,
            continue_final_message=continue_final,
        )

    def _run(self, prompts: list[str], params: SamplingParams) -> list[list[str]]:
        outs = self.llm.generate(
            prompts, self._sp(params), lora_request=self._lora_request
        )
        return [[o.text for o in r.outputs] for r in outs]

    def chat(self, messages: Conversation, params: SamplingParams | None = None) -> list[str]:
        params = params or self.default_params()
        return self._run([self._render(messages)], params)[0]

    def chat_batch(
        self, conversations: Sequence[Conversation], params: SamplingParams | None = None
    ) -> list[list[str]]:
        params = params or self.default_params()
        return self._run([self._render(c) for c in conversations], params)

    def continue_prefill(
        self, messages: Conversation, prefill: str, params: SamplingParams | None = None
    ) -> list[str]:
        params = params or self.default_params()
        convo = list(messages) + [type(messages[0])(role="assistant", content=prefill)]
        prompt = self._render(convo, continue_final=True)
        return self._run([prompt], params)[0]
