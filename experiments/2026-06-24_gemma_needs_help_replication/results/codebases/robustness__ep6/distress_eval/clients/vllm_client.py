"""Optional high-throughput sampling backend for the large elicitation sweeps.

vLLM is ~10-50x faster than the transformers loop for sampling thousands of
multi-turn rollouts at temperature 1, but does not expose hidden states and its
prefill support is limited. Use it for Section 2 / Section 4 *evaluation* runs and
fall back to LocalHFClient for prefill (Section 3) and probing (Appendix I).

This client is only constructed when `--backend vllm` is requested; vLLM is left
commented out in requirements.txt so the rest of the pipeline installs without a
GPU.
"""

from __future__ import annotations

from .base import ChatMessage, GenerationResult, ModelClient


class VLLMClient(ModelClient):
    def __init__(self, name: str, model_id: str, *, adapter_path: str | None = None,
                 max_model_len: int = 16384, tensor_parallel_size: int = 1):
        from vllm import LLM
        from vllm.lora.request import LoRARequest  # noqa: F401  (used in chat)

        self.name = name
        self.model_id = model_id
        self._llm = LLM(
            model=model_id,
            enable_lora=adapter_path is not None,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
        )
        self._adapter_path = adapter_path
        self._tokenizer = self._llm.get_tokenizer()

    def chat(self, messages, *, n=1, temperature=1.0, max_new_tokens=2048):
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        params = SamplingParams(n=n, temperature=temperature, top_p=1.0,
                                max_tokens=max_new_tokens)
        lora = (LoRARequest("adapter", 1, self._adapter_path)
                if self._adapter_path else None)
        outs = self._llm.generate([prompt], params, lora_request=lora)
        return [GenerationResult(text=o.text) for o in outs[0].outputs]
