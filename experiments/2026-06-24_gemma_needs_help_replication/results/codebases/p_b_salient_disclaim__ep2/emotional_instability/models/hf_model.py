"""Local HuggingFace / vLLM inference for Gemma (instruct, base, and finetunes).

Section 2.1 samples 4000 responses per model at temperature 1, so throughput
matters: we use vLLM when available and fall back to plain `transformers`
generation otherwise. The Gemma-3 chat template is applied via the tokenizer.

Prefill (Section 3) is implemented by rendering the chat template with
`add_generation_prompt=True`, then appending the prefill text to the prompt
string before generation, and stripping the prefill from the decoded output.
Base/pretrained Gemma checkpoints are handled the same way -- they simply
continue from the rendered prompt, which is exactly the paper's prefill setup.
"""

from __future__ import annotations

from typing import Optional

from .base import ChatMessage, GenerationResult


def _messages_to_dicts(messages: list[ChatMessage]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


class HFModelClient:
    """vLLM-backed (or transformers-backed) Gemma client."""

    supports_prefill = True

    def __init__(
        self,
        key: str,
        model_id: str,
        *,
        is_instruct: bool = True,
        default_temperature: float = 1.0,
        default_max_new_tokens: int = 2048,
        use_vllm: bool = True,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        lora_path: Optional[str] = None,
    ):
        self.key = key
        self.model_id = model_id
        self.is_instruct = is_instruct
        self.default_temperature = default_temperature
        self.default_max_new_tokens = default_max_new_tokens
        self.lora_path = lora_path
        self._use_vllm = use_vllm
        self._dtype = dtype
        self._tp = tensor_parallel_size
        self._gpu_util = gpu_memory_utilization

        self._llm = None          # vLLM engine
        self._hf_model = None      # transformers model
        self._tokenizer = None

    # --------------------------------------------------------------------- #
    # Lazy backend init
    # --------------------------------------------------------------------- #
    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self._tokenizer

    def _ensure_backend(self):
        if self._use_vllm and self._llm is None:
            from vllm import LLM

            kwargs = dict(
                model=self.model_id,
                dtype=self._dtype,
                tensor_parallel_size=self._tp,
                gpu_memory_utilization=self._gpu_util,
            )
            if self.lora_path:
                kwargs["enable_lora"] = True
            self._llm = LLM(**kwargs)
        elif not self._use_vllm and self._hf_model is None:
            import torch
            from transformers import AutoModelForCausalLM

            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=getattr(torch, self._dtype),
                device_map="auto",
            )
            if self.lora_path:
                from peft import PeftModel

                self._hf_model = PeftModel.from_pretrained(
                    self._hf_model, self.lora_path
                )

    # --------------------------------------------------------------------- #
    # Prompt rendering
    # --------------------------------------------------------------------- #
    def _render_prompt(self, messages: list[ChatMessage], prefill: str = "") -> str:
        tok = self._ensure_tokenizer()
        text = tok.apply_chat_template(
            _messages_to_dicts(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        if prefill:
            text = text + prefill
        return text

    # --------------------------------------------------------------------- #
    # Generation
    # --------------------------------------------------------------------- #
    def _sample(
        self,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        n: int,
    ) -> list[GenerationResult]:
        self._ensure_backend()
        if self._use_vllm:
            from vllm import SamplingParams

            sp = SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
                n=n,
            )
            lora_req = None
            if self.lora_path:
                from vllm.lora.request import LoRARequest

                lora_req = LoRARequest("adapter", 1, self.lora_path)
            outs = self._llm.generate([prompt], sp, lora_request=lora_req)
            results = []
            for completion in outs[0].outputs:
                results.append(
                    GenerationResult(
                        text=completion.text,
                        token_ids=list(completion.token_ids),
                    )
                )
            return results

        # transformers fallback
        import torch

        tok = self._ensure_tokenizer()
        inputs = tok(prompt, return_tensors="pt").to(self._hf_model.device)
        results = []
        for _ in range(n):
            out = self._hf_model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                max_new_tokens=max_new_tokens,
            )
            gen_ids = out[0][inputs["input_ids"].shape[1] :]
            results.append(
                GenerationResult(
                    text=tok.decode(gen_ids, skip_special_tokens=True),
                    token_ids=gen_ids.tolist(),
                )
            )
        return results

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        n: int = 1,
    ) -> list[GenerationResult]:
        prompt = self._render_prompt(messages)
        return self._sample(
            prompt,
            temperature if temperature is not None else self.default_temperature,
            max_new_tokens or self.default_max_new_tokens,
            n,
        )

    def generate_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        *,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        n: int = 1,
    ) -> list[GenerationResult]:
        prompt = self._render_prompt(messages, prefill=prefill)
        results = self._sample(
            prompt,
            temperature if temperature is not None else self.default_temperature,
            max_new_tokens or self.default_max_new_tokens,
            n,
        )
        for r in results:
            r.prefill = prefill
        return results
