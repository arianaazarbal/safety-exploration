"""Local Gemma client (vLLM or transformers backend).

* ``backend="vllm"`` — fast batched generation for the 4000-rollout eval sweep.
* ``backend="hf"`` — transformers; required for (a) base/pretrained ("-pt")
  models that have no chat template, (b) prefill continuations
  (``continue_assistant``), and (c) loading LoRA adapters produced by the
  finetuning code.

For instruct models the chat template is applied via the tokenizer. For base
models (``spec.is_chat=False``) we render the conversation as plain text and
continue it, matching the prefill methodology of Section 3.
"""

from __future__ import annotations

from .base import Message, ModelClient
from ..config import ModelSpec


def _render_base_text(messages: list[Message], assistant_prefix: str = "") -> str:
    """Plain-text rendering for base models (no chat template).

    Uses a simple, explicit transcript format. This is intentionally neutral so
    the base model continues from the same starting point as the instruct model
    (the prefill experiment paraphrases responses to control for style anyway).
    """
    parts: list[str] = []
    for m in messages:
        if m["role"] == "system":
            parts.append(m["content"])
        elif m["role"] == "user":
            parts.append(f"User: {m['content']}")
        elif m["role"] == "assistant":
            parts.append(f"Assistant: {m['content']}")
    text = "\n\n".join(parts)
    # Open the assistant turn the continuation should extend.
    text += "\n\nAssistant: " + assistant_prefix
    return text


class LocalChat(ModelClient):
    def __init__(
        self,
        spec: ModelSpec,
        *,
        backend: str | None = None,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
    ):
        self.spec = spec
        self.backend = backend or spec.backend
        self.adapter_path = adapter_path
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self._llm = None  # vLLM LLM
        self._hf_model = None
        self._tokenizer = None

    # --------------------------------------------------------------------- #
    # Lazy backend init.
    # --------------------------------------------------------------------- #
    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.spec.identifier)
        return self._tokenizer

    def _ensure_vllm(self):
        if self._llm is None:
            from vllm import LLM

            kwargs: dict = {
                "model": self.spec.identifier,
                "dtype": self.dtype,
                "tensor_parallel_size": self.tensor_parallel_size,
            }
            if self.max_model_len:
                kwargs["max_model_len"] = self.max_model_len
            if self.adapter_path:
                kwargs["enable_lora"] = True
            self._llm = LLM(**kwargs)
        return self._llm

    def _ensure_hf(self):
        if self._hf_model is None:
            import torch
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                self.spec.identifier,
                torch_dtype=getattr(torch, self.dtype),
                device_map="auto",
            )
            if self.adapter_path:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, self.adapter_path)
            model.eval()
            self._hf_model = model
        return self._hf_model

    # --------------------------------------------------------------------- #
    # Prompt rendering.
    # --------------------------------------------------------------------- #
    def _render_chat(self, messages: list[Message], assistant_prefix: str | None) -> str:
        tok = self._ensure_tokenizer()
        if not self.spec.is_chat:
            return _render_base_text(messages, assistant_prefix or "")
        if assistant_prefix is not None:
            # Continue a partial final assistant turn.
            msgs = list(messages) + [{"role": "assistant", "content": assistant_prefix}]
            return tok.apply_chat_template(
                msgs,
                tokenize=False,
                continue_final_message=True,
            )
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # --------------------------------------------------------------------- #
    # Generation.
    # --------------------------------------------------------------------- #
    def _generate(
        self,
        prompt_text: str,
        *,
        temperature: float,
        max_tokens: int,
        top_p: float,
        n: int,
    ) -> list[str]:
        if self.backend == "vllm":
            from vllm import SamplingParams

            sp = SamplingParams(
                temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n
            )
            lora_req = None
            if self.adapter_path:
                from vllm.lora.request import LoRARequest

                lora_req = LoRARequest("adapter", 1, self.adapter_path)
            outs = self._ensure_vllm().generate(
                [prompt_text], sp, lora_request=lora_req
            )
            return [o.text for o in outs[0].outputs]

        # transformers backend
        import torch

        tok = self._ensure_tokenizer()
        model = self._ensure_hf()
        inputs = tok(prompt_text, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            num_return_sequences=n,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        return [tok.decode(g[prompt_len:], skip_special_tokens=True) for g in gen]

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        n: int = 1,
    ) -> list[str]:
        prompt_text = self._render_chat(messages, assistant_prefix=None)
        return self._generate(
            prompt_text, temperature=temperature, max_tokens=max_tokens, top_p=top_p, n=n
        )

    def continue_assistant(
        self,
        messages: list[Message],
        assistant_prefix: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        n: int = 1,
    ) -> list[str]:
        prompt_text = self._render_chat(messages, assistant_prefix=assistant_prefix)
        return self._generate(
            prompt_text, temperature=temperature, max_tokens=max_tokens, top_p=top_p, n=n
        )

    def chat_batch(
        self,
        batch: list[list[Message]],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
    ) -> list[str]:
        if self.backend != "vllm":
            return super().chat_batch(
                batch, temperature=temperature, max_tokens=max_tokens, top_p=top_p
            )
        from vllm import SamplingParams

        prompts = [self._render_chat(m, assistant_prefix=None) for m in batch]
        sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=1)
        lora_req = None
        if self.adapter_path:
            from vllm.lora.request import LoRARequest

            lora_req = LoRARequest("adapter", 1, self.adapter_path)
        outs = self._ensure_vllm().generate(prompts, sp, lora_request=lora_req)
        return [o.outputs[0].text for o in outs]
