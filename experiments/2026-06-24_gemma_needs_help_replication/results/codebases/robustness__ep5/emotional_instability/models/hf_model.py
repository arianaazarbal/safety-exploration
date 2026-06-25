"""Local Gemma inference.

Uses vLLM when available (much faster for the 4000-sample eval), and falls back
to plain HuggingFace `transformers` otherwise. Both paths support:

* chat()     — applies the Gemma chat template to the message list.
* complete() — raw continuation, used for prefilling and for *base* models
               (which have no chat template).

A loaded LoRA adapter (the DPO/SFT mitigation) can be supplied via
`adapter_path`; vLLM serves it with LoRARequest, transformers via PEFT.
"""
from __future__ import annotations

from typing import Optional

from .base import ChatMessage, ModelClient


class HFModelClient(ModelClient):
    def __init__(
        self,
        spec,
        adapter_path: Optional[str] = None,
        backend: str = "auto",          # "vllm" | "transformers" | "auto"
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
    ):
        self.spec = spec
        self.adapter_path = adapter_path
        self.dtype = dtype
        self.max_model_len = max_model_len
        self._gpu_util = gpu_memory_utilization
        self.backend = self._resolve_backend(backend)
        self._engine = None
        self._tokenizer = None
        self._hf_model = None
        self._lora_request = None
        self._load()

    # ---- loading -------------------------------------------------------- #
    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend != "auto":
            return backend
        try:
            import vllm  # noqa: F401
            return "vllm"
        except ImportError:
            return "transformers"

    def _load(self) -> None:
        if self.backend == "vllm":
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_vllm(self) -> None:
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)
        self._engine = LLM(
            model=self.spec.model_id,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self._gpu_util,
            enable_lora=self.adapter_path is not None,
            max_lora_rank=64,
        )
        if self.adapter_path:
            self._lora_request = LoRARequest("mitigation", 1, self.adapter_path)

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id,
            torch_dtype=getattr(torch, self.dtype),
            device_map="auto",
        )
        if self.adapter_path:
            from peft import PeftModel
            self._hf_model = PeftModel.from_pretrained(self._hf_model, self.adapter_path)
        self._hf_model.eval()

    # ---- prompt formatting --------------------------------------------- #
    def _render_chat(self, messages: list[ChatMessage], add_generation_prompt: bool = True,
                     continue_final: bool = False) -> str:
        """Apply the Gemma chat template. If `continue_final` is True the final
        assistant message is treated as a prefill to be continued (no EOT)."""
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        # Gemma has no system role; fold any system content into the first user
        # turn (mirrors transformers' Gemma chat template behaviour).
        if msgs and msgs[0]["role"] == "system":
            sys_msg = msgs.pop(0)
            if msgs:
                msgs[0]["content"] = sys_msg["content"] + "\n\n" + msgs[0]["content"]
            else:
                msgs = [{"role": "user", "content": sys_msg["content"]}]
        return self._tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt and not continue_final,
            continue_final_message=continue_final,
        )

    # ---- generation ----------------------------------------------------- #
    def chat(self, messages, n=1, temperature=1.0, max_new_tokens=2048):
        prompt = self._render_chat(messages, add_generation_prompt=True)
        return self._generate(prompt, n, temperature, max_new_tokens)

    def complete(self, prompt, n=1, temperature=1.0, max_new_tokens=2048):
        return self._generate(prompt, n, temperature, max_new_tokens)

    def chat_with_prefill(self, messages, prefill, n=1, temperature=1.0, max_new_tokens=2048):
        """Continue an assistant turn that starts with `prefill` (Section 3).

        Instruct models continue inside the chat template. Base/pretrained
        models have no chat template, so we render the conversation as plain
        text and continue from there (matching the paper's base-model prefill).
        """
        if getattr(self.spec, "is_base", False):
            prompt = self._render_plain(messages, prefill)
        else:
            msgs = list(messages) + [ChatMessage("assistant", prefill)]
            prompt = self._render_chat(msgs, continue_final=True)
        outs = self._generate(prompt, n, temperature, max_new_tokens)
        # Strip the prefill so only the *continuation* is returned/scored.
        return [o[len(prefill):] if o.startswith(prefill) else o for o in outs]

    @staticmethod
    def _render_plain(messages, prefill: str) -> str:
        """Plain-text transcript rendering for base models (no chat template)."""
        lines = []
        for m in messages:
            tag = {"user": "User", "assistant": "Assistant", "system": "System"}[m.role]
            lines.append(f"{tag}: {m.content}")
        lines.append(f"Assistant: {prefill}")
        return "\n\n".join(lines)

    def _generate(self, prompt: str, n: int, temperature: float, max_new_tokens: int):
        if self.backend == "vllm":
            return self._generate_vllm(prompt, n, temperature, max_new_tokens)
        return self._generate_transformers(prompt, n, temperature, max_new_tokens)

    def _generate_vllm(self, prompt, n, temperature, max_new_tokens):
        from vllm import SamplingParams

        params = SamplingParams(
            n=n, temperature=temperature, top_p=1.0, max_tokens=max_new_tokens,
        )
        out = self._engine.generate(
            [prompt], params, lora_request=self._lora_request, use_tqdm=False,
        )
        return [c.text for c in out[0].outputs]

    def _generate_transformers(self, prompt, n, temperature, max_new_tokens):
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._hf_model.device)
        with torch.no_grad():
            gen = self._hf_model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = gen[:, inputs["input_ids"].shape[1]:]
        return self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def close(self):
        self._engine = None
        self._hf_model = None
