"""Local HuggingFace backend for the Gemma family.

This is the only backend that supports the operations Sections 3-4 require:
prefilled continuations, hidden-state / logit access (Appendix I), and being a
fine-tuning target.  By default it uses plain ``transformers`` generation; if
vLLM is installed and enabled it is used for the high-throughput rollout and
continuation sampling (4000 responses/model, 50 continuations/prefill).

The class is written so it imports lazily — none of torch/transformers is
touched until a backend is actually constructed, so the rest of the package
(config, scoring, data) can be imported and unit-tested without a GPU.
"""

from __future__ import annotations

from typing import Optional

from ..config import ModelSpec, RuntimeConfig, SamplingConfig
from .base import ChatBackend, GenerationResult, Message


class HFBackend(ChatBackend):
    def __init__(
        self,
        spec: ModelSpec,
        runtime: RuntimeConfig,
        adapter_path: Optional[str] = None,
    ):
        super().__init__(spec)
        self.runtime = runtime
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None
        self._vllm = None
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)

        # vLLM path (generation only; logits still go through transformers).
        if self.runtime.use_vllm and self.adapter_path is None:
            try:
                from vllm import LLM
                self._vllm = LLM(
                    model=self.spec.model_id,
                    dtype=self.runtime.hf_dtype,
                    enable_lora=False,
                )
                return
            except Exception:
                # vLLM not available / failed to init -> fall back to HF.
                self._vllm = None

        import torch
        dtype = getattr(torch, self.runtime.hf_dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id,
            torch_dtype=dtype,
            device_map=self.runtime.hf_device_map,
            output_hidden_states=self.spec.supports_logits,
        )
        if self.adapter_path is not None:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    # ------------------------------------------------------------------
    def _render_prompt(self, messages: list[Message], prefill: str | None) -> str:
        """Turn a chat conversation into a single prompt string.

        Instruct models use the chat template; base/pretrained models (which
        were never trained on chat formatting) get a plain concatenation so the
        prefill experiment can force a consistent continuation point (Sec 3.1).
        """
        tok = self._tokenizer
        if self.spec.supports_chat_template:
            if prefill is not None:
                msgs = list(messages) + [{"role": "assistant", "content": prefill}]
                return tok.apply_chat_template(
                    msgs, tokenize=False,
                    add_generation_prompt=False, continue_final_message=True,
                )
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        # Base model: concatenate turns plainly, then the prefill (if any).
        parts = []
        for m in messages:
            tag = {"user": "User", "assistant": "Assistant",
                   "system": "System"}.get(m["role"], m["role"])
            parts.append(f"{tag}: {m['content']}")
        parts.append("Assistant: " + (prefill or ""))
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    def generate(
        self,
        messages: list[Message],
        sampling: SamplingConfig,
        n: int = 1,
        prefill: str | None = None,
    ) -> list[GenerationResult]:
        prompt = self._render_prompt(messages, prefill)
        prefill = prefill or ""

        if self._vllm is not None:
            from vllm import SamplingParams
            params = SamplingParams(
                n=n, temperature=sampling.temperature, top_p=sampling.top_p,
                max_tokens=sampling.max_new_tokens,
            )
            out = self._vllm.generate([prompt], params)[0]
            return [
                GenerationResult(text=o.text, prefill=prefill,
                                 finish_reason=o.finish_reason)
                for o in out.outputs
            ]

        return self._generate_transformers(prompt, sampling, n, prefill)

    def _generate_transformers(
        self, prompt: str, sampling: SamplingConfig, n: int, prefill: str,
    ) -> list[GenerationResult]:
        import torch
        tok, model = self._tokenizer, self._model
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=sampling.temperature > 0,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_new_tokens=sampling.max_new_tokens,
                num_return_sequences=n,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        results = []
        for seq in out:
            text = tok.decode(seq[prompt_len:], skip_special_tokens=True)
            results.append(GenerationResult(text=text, prefill=prefill,
                                            finish_reason="stop"))
        return results

    # ------------------------------------------------------------------
    def hidden_states(self, text: str):
        """Return ``(tokens, per-layer hidden states)`` for ``text``.

        Used by the Appendix-I internal-emotion detector.  ``hidden_states`` is
        a list of length ``n_layers+1``; element ``L`` has shape
        ``[seq_len, d_model]`` (batch dim squeezed).
        """
        import torch
        if self._model is None:
            raise RuntimeError(
                "hidden_states requires the transformers model; disable vLLM "
                "(runtime.use_vllm=False) for internal-emotion experiments."
            )
        tok, model = self._tokenizer, self._model
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        token_ids = inputs["input_ids"][0].tolist()
        tokens = tok.convert_ids_to_tokens(token_ids)
        hs = [h[0] for h in out.hidden_states]  # squeeze batch
        return tokens, hs

    def unembed(self, hidden: "torch.Tensor"):  # type: ignore[name-defined]
        """Project a residual-stream vector through the LM head to vocab logits."""
        import torch
        model = self._model
        lm_head = model.get_output_embeddings()
        with torch.no_grad():
            # Gemma applies a final norm before the head; approximate the
            # standard "logit lens" by norm-then-unembed when available.
            norm = getattr(getattr(model, "model", model), "norm", None)
            h = norm(hidden) if norm is not None else hidden
            return lm_head(h)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model
