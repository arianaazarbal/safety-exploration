"""Local Gemma inference via HuggingFace transformers (with optional vLLM).

Two code paths:
  * vLLM (preferred): high throughput, needed for the ~4000 rollouts/model in
    Section 2 and the 50-continuation prefill sweeps in Section 3.
  * transformers fallback: simpler, also exposes the residual stream which the
    internal-emotion probing in Appendix I requires (vLLM does not give us
    hidden states), so the probing module always uses this path.

Gemma chat formatting is delegated to the tokenizer's chat template. Base
(`-pt`) checkpoints have no chat template, so chat-format generation on a base
model raises; use `continue_text` for those (the prefill experiments).
"""
from __future__ import annotations

import os
from typing import Optional

from ..config import ModelSpec
from .base import ChatClient, GenResult, Message


def _messages_to_dicts(messages: list[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _fold_system(dicts: list[dict]) -> list[dict]:
    """Merge a leading system message into the first user turn.

    Gemma's chat template historically rejects the `system` role; folding the
    system text into the first user message preserves the instruction while
    staying template-compatible. Only used as a fallback when the template
    raises on a system message.
    """
    if not dicts or dicts[0]["role"] != "system":
        return dicts
    system = dicts[0]["content"]
    rest = dicts[1:]
    for i, m in enumerate(rest):
        if m["role"] == "user":
            rest[i] = {"role": "user",
                       "content": f"{system}\n\n{m['content']}"}
            return rest
    return [{"role": "user", "content": system}] + rest


def _adapter_tag(path: str) -> str:
    return os.path.basename(os.path.normpath(path))


class HFLocalClient(ChatClient):
    """Backed by `transformers`. Loads the model lazily on first use."""

    def __init__(self, spec: ModelSpec, *, dtype: str = "bfloat16",
                 device_map: str = "auto", load_in_4bit: bool = False,
                 adapter_path: Optional[str] = None):
        self.spec = spec
        # Distinguish finetuned variants in result filenames.
        self.key = spec.key if adapter_path is None else f"{spec.key}+{_adapter_tag(adapter_path)}"
        self.dtype = dtype
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self.adapter_path = adapter_path
        self._model = None
        self._tok = None

    # ------------------------------------------------------------------ #
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant_kwargs = {}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self._tok = AutoTokenizer.from_pretrained(self.spec.identifier)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.spec.identifier,
            torch_dtype=getattr(torch, self.dtype),
            device_map=self.device_map,
            **quant_kwargs,
        )
        if self.adapter_path is not None:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tok

    # ------------------------------------------------------------------ #
    @property
    def supports_prefill(self) -> bool:
        return True

    def _render_chat(self, messages: list[Message], add_generation_prompt: bool,
                     continue_final: bool = False) -> str:
        tok = self.tokenizer
        if tok.chat_template is None:
            raise RuntimeError(
                f"{self.key} has no chat template (base model). Use "
                "continue_text() for raw prefill instead.")
        dicts = _messages_to_dicts(messages)
        try:
            return tok.apply_chat_template(
                dicts, tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final)
        except Exception:  # noqa: BLE001 - Gemma rejects system role -> fold it
            return tok.apply_chat_template(
                _fold_system(dicts), tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final)

    # ------------------------------------------------------------------ #
    def generate(self, messages, *, temperature=1.0, max_new_tokens=2048,
                 n=1, seed=None) -> list[GenResult]:
        self._ensure_loaded()
        # If the final message is an assistant turn, we are *prefilling* that
        # turn and want the model to continue it rather than open a new one.
        continue_final = messages[-1].role == "assistant"
        prompt = self._render_chat(
            messages,
            add_generation_prompt=not continue_final,
            continue_final=continue_final,
        )
        return self._sample_from_text(
            prompt, temperature=temperature, max_new_tokens=max_new_tokens,
            n=n, seed=seed)

    def continue_text(self, prompt_text, *, temperature=1.0, max_new_tokens=512,
                      n=1, seed=None) -> list[GenResult]:
        self._ensure_loaded()
        return self._sample_from_text(
            prompt_text, temperature=temperature, max_new_tokens=max_new_tokens,
            n=n, seed=seed)

    # ------------------------------------------------------------------ #
    def _sample_from_text(self, prompt_text, *, temperature, max_new_tokens,
                          n, seed) -> list[GenResult]:
        import torch
        tok = self._tok
        if seed is not None:
            torch.manual_seed(seed)
        enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self._model.device) for k, v in enc.items()}
        do_sample = temperature > 0
        out = self._model.generate(
            **enc,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=1.0 if do_sample else None,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        results = []
        for seq in out:
            gen_ids = seq[prompt_len:]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            results.append(GenResult(
                text=text,
                prompt_tokens=prompt_len,
                completion_tokens=int(gen_ids.shape[0]),
            ))
        return results

    def close(self):
        self._model = None
        self._tok = None


class VLLMClient(ChatClient):
    """High-throughput local inference via vLLM.

    Preferred for the large sampling runs. Does not expose hidden states, so the
    Appendix I probing falls back to `HFLocalClient`.
    """

    def __init__(self, spec: ModelSpec, *, tensor_parallel_size: Optional[int] = None,
                 gpu_memory_utilization: float = 0.90, max_model_len: int = 16384,
                 adapter_path: Optional[str] = None):
        self.spec = spec
        self.adapter_path = adapter_path
        self.key = spec.key if adapter_path is None else f"{spec.key}+{_adapter_tag(adapter_path)}"
        self.tp = tensor_parallel_size or int(os.environ.get("DISTRESS_TP", "1"))
        self.gpu_mem = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._llm = None
        self._tok = None
        self._lora_request = None

    def _ensure_loaded(self):
        if self._llm is not None:
            return
        from vllm import LLM
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.spec.identifier)
        self._llm = LLM(
            model=self.spec.identifier,
            tensor_parallel_size=self.tp,
            gpu_memory_utilization=self.gpu_mem,
            max_model_len=self.max_model_len,
            dtype="bfloat16",
            enable_lora=self.adapter_path is not None,
            max_lora_rank=64,
        )
        if self.adapter_path is not None:
            from vllm.lora.request import LoRARequest
            self._lora_request = LoRARequest("adapter", 1, self.adapter_path)

    @property
    def supports_prefill(self) -> bool:
        return True

    def _sampling_params(self, temperature, max_new_tokens, n, seed):
        from vllm import SamplingParams
        return SamplingParams(
            n=n,
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_new_tokens,
            seed=seed,
        )

    def generate(self, messages, *, temperature=1.0, max_new_tokens=2048,
                 n=1, seed=None) -> list[GenResult]:
        self._ensure_loaded()
        continue_final = messages[-1].role == "assistant"
        if self._tok.chat_template is None:
            raise RuntimeError(
                f"{self.key} has no chat template (base model). Use continue_text().")
        dicts = _messages_to_dicts(messages)
        try:
            prompt = self._tok.apply_chat_template(
                dicts, tokenize=False,
                add_generation_prompt=not continue_final,
                continue_final_message=continue_final)
        except Exception:  # noqa: BLE001 - Gemma rejects system role -> fold it
            prompt = self._tok.apply_chat_template(
                _fold_system(dicts), tokenize=False,
                add_generation_prompt=not continue_final,
                continue_final_message=continue_final)
        return self._gen_text(prompt, temperature, max_new_tokens, n, seed)

    def continue_text(self, prompt_text, *, temperature=1.0, max_new_tokens=512,
                      n=1, seed=None) -> list[GenResult]:
        self._ensure_loaded()
        return self._gen_text(prompt_text, temperature, max_new_tokens, n, seed)

    def _gen_text(self, prompt, temperature, max_new_tokens, n, seed):
        params = self._sampling_params(temperature, max_new_tokens, n, seed)
        outputs = self._llm.generate([prompt], params, use_tqdm=False,
                                     lora_request=self._lora_request)
        results = []
        for comp in outputs[0].outputs:
            results.append(GenResult(
                text=comp.text,
                completion_tokens=len(comp.token_ids),
                finish_reason=comp.finish_reason,
            ))
        return results

    def close(self):
        self._llm = None
