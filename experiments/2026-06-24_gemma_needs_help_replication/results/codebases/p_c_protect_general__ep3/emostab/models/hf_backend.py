"""Local HuggingFace backend for Gemma models.

Used for all open-weight generation (Section 2 elicitation on Gemma, Section 3
prefilling, calm-data generation, and post-finetuning evaluation). Supports:

* chat generation at temperature 1 (the paper's setting),
* response *prefilling* / continuation for the Section 3 experiment, and
* optional LoRA adapter loading for the DPO/SFT finetuned variants.

By default we use plain ``transformers``. For the paper-scale sample counts
(4000 rollouts/model) batched generation via vLLM is far faster; set
``backend="vllm"`` to use it if installed. See DESIGN.md.
"""
from __future__ import annotations

from functools import cached_property

from ..config import SamplingConfig
from .base import Message, ModelBackend


class HFBackend(ModelBackend):
    supports_prefill = True

    def __init__(
        self,
        model_id: str,
        *,
        key: str | None = None,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        engine: str = "transformers",  # "transformers" | "vllm"
    ) -> None:
        self.model_id = model_id
        self.key = key or model_id
        self.adapter_path = adapter_path
        self.dtype = dtype
        self.device_map = device_map
        self.engine = engine

    # ------------------------------------------------------------------ #
    # Lazy model loading (kept out of __init__ so configs are cheap to build)
    # ------------------------------------------------------------------ #
    @cached_property
    def _tokenizer(self):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_id)
        return tok

    @cached_property
    def _model(self):
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=getattr(torch, self.dtype),
            device_map=self.device_map,
        )
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        return model

    # ------------------------------------------------------------------ #
    # Prompt formatting
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[Message], add_generation_prompt: bool = True) -> str:
        """Render a chat to a prompt string using Gemma's chat template.

        Pretrained ('-pt') checkpoints have no chat template; for those we fall
        back to a minimal turn format. The Section 3 prefill experiment always
        appends a forced prefix to this, so base models still continue coherently.
        """
        tok = self._tokenizer
        if tok.chat_template is not None:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        # Base model fallback: plain alternating turns.
        parts = []
        for m in messages:
            parts.append(f"{m['role'].capitalize()}: {m['content']}")
        if add_generation_prompt:
            parts.append("Assistant:")
        return "\n".join(parts)

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self, messages: list[Message], sampling: SamplingConfig) -> str:
        return self.generate_batch([messages], sampling)[0]

    def generate_batch(
        self, batch: list[list[Message]], sampling: SamplingConfig
    ) -> list[str]:
        prompts = [self._render(m) for m in batch]
        return self._raw_generate(prompts, sampling)

    def continue_prefill(
        self,
        messages: list[Message],
        prefix: str,
        sampling: SamplingConfig,
        n_samples: int = 1,
    ) -> list[str]:
        # Render the chat, then append the forced assistant prefix verbatim so
        # the model continues from it. We strip the prefix from each completion.
        base_prompt = self._render(messages, add_generation_prompt=True) + prefix
        prompts = [base_prompt] * n_samples
        return self._raw_generate(prompts, sampling, strip_prefixes=[prefix] * n_samples)

    # ------------------------------------------------------------------ #
    def _raw_generate(
        self,
        prompts: list[str],
        sampling: SamplingConfig,
        strip_prefixes: list[str] | None = None,
    ) -> list[str]:
        if self.engine == "vllm":
            return self._vllm_generate(prompts, sampling, strip_prefixes)
        return self._transformers_generate(prompts, sampling, strip_prefixes)

    def _transformers_generate(self, prompts, sampling, strip_prefixes):
        import torch

        tok = self._tokenizer
        model = self._model
        outputs: list[str] = []
        for i, prompt in enumerate(prompts):
            if sampling.seed is not None:
                torch.manual_seed(sampling.seed + i)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    do_sample=sampling.temperature > 0,
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    max_new_tokens=sampling.max_new_tokens,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            text = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            outputs.append(text)
        return outputs

    def _vllm_generate(self, prompts, sampling, strip_prefixes):
        from vllm import SamplingParams  # noqa: F401

        llm = self._vllm_engine
        params = SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_new_tokens,
            seed=sampling.seed,
        )
        results = llm.generate(prompts, params)
        return [r.outputs[0].text for r in results]

    @cached_property
    def _vllm_engine(self):
        from vllm import LLM

        return LLM(model=self.model_id, dtype=self.dtype, enable_lora=bool(self.adapter_path))
