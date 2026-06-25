"""Local Gemma client via HuggingFace ``transformers``.

Handles both the instruct (``-it``) and pretrained/base (``-pt``) checkpoints,
prefill continuation (needed for Section 3), optional LoRA adapters (to evaluate
DPO/SFT-finetuned models from Section 4), and a hook to read hidden states for
the internal-emotion probe (Appendix I).

Imports of ``torch``/``transformers``/``peft`` are deferred to construction so
that the rest of the package (config, analysis of cached results) can be used in
an environment without a GPU stack installed.
"""
from __future__ import annotations

from typing import Any

from .base import GenerationResult, Message, ModelClient


class GemmaClient(ModelClient):
    """Open-weight Gemma 3 inference.

    ``options`` keys:
        is_base (bool)      -- pretrained checkpoint; skip chat template.
        dtype (str)         -- "bfloat16" (default), "float16", "float32".
        device_map (str)    -- passed to ``from_pretrained`` (default "auto").
        adapter_path (str)  -- optional LoRA adapter dir to load on top.
        attn_implementation -- e.g. "eager" (needed for hidden-state hooks).
    """

    def __init__(self, model_id: str, **kw):
        super().__init__(model_id, **kw)
        self._model = None
        self._tokenizer = None
        self.is_base = bool(self.options.get("is_base", model_id.endswith("-pt")))

    # ------------------------------------------------------------------ load
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.options.get("dtype", "bfloat16")]

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.options.get("device_map", "auto"),
            attn_implementation=self.options.get("attn_implementation", "sdpa"),
        )
        adapter = self.options.get("adapter_path")
        if adapter:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, adapter)
        self._model.eval()

    # -------------------------------------------------------------- prompts
    def _render_prompt(self, messages: list[Message], prefill: str | None = None) -> str:
        """Turn messages into the model's input string.

        Instruct: apply the chat template, add the generation prompt, then append
        any ``prefill`` so generation continues the assistant turn.

        Base: there is no chat template. We render a lightweight transcript
        ("User: ... / Assistant: ...") and append the prefill, matching the
        paper's approach of prefilling base models so they "consistently continue
        the response".
        """
        if not self.is_base:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if prefill:
                text = text + prefill
            return text

        # base model: plain transcript
        parts = []
        for m in messages:
            tag = {"user": "User", "assistant": "Assistant", "system": "System"}.get(m["role"], m["role"])
            parts.append(f"{tag}: {m['content']}")
        parts.append("Assistant:" + (f" {prefill}" if prefill else ""))
        return "\n".join(parts)

    # ------------------------------------------------------------- generate
    def _generate_raw(self, prompt: str, *, temperature: float, max_tokens: int,
                      n: int, seed: int | None) -> list[str]:
        import torch

        self._ensure_loaded()
        tok = self._tokenizer
        inputs = tok(prompt, return_tensors="pt").to(self._model.device)
        if seed is not None:
            torch.manual_seed(seed)

        do_sample = temperature > 0
        gen = self._model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            max_new_tokens=max_tokens,
            num_return_sequences=n,
            pad_token_id=tok.pad_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        out = []
        for seq in gen:
            cont = seq[prompt_len:]
            out.append(tok.decode(cont, skip_special_tokens=True))
        return out

    def generate(self, messages, *, temperature=None, max_tokens=None, n=1,
                 stop=None, seed=None) -> list[GenerationResult]:
        prompt = self._render_prompt(messages)
        texts = self._generate_raw(
            prompt,
            temperature=self.default_temperature if temperature is None else temperature,
            max_tokens=self.default_max_tokens if max_tokens is None else max_tokens,
            n=n,
            seed=seed,
        )
        return [GenerationResult(text=t.strip(), meta={"prompt_chars": len(prompt)}) for t in texts]

    def continue_from(self, messages, prefill, *, temperature=None, max_tokens=None,
                      n=1, seed=None) -> list[GenerationResult]:
        prompt = self._render_prompt(messages, prefill=prefill)
        texts = self._generate_raw(
            prompt,
            temperature=self.default_temperature if temperature is None else temperature,
            max_tokens=self.default_max_tokens if max_tokens is None else max_tokens,
            n=n,
            seed=seed,
        )
        # texts already exclude the prompt (and thus the prefill)
        return [GenerationResult(text=t, meta={"prefill": prefill}) for t in texts]

    # ----------------------------------------------- hidden states (App. I)
    def hidden_states_for(self, messages: list[Message], prefill: str | None = None):
        """Return per-layer hidden states for the last token of the input.

        Used by the logit-based internal-emotion probe. Returns a tensor of
        shape ``(num_layers + 1, hidden_dim)`` on CPU.
        """
        import torch

        self._ensure_loaded()
        prompt = self._render_prompt(messages, prefill=prefill)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        # hidden_states: tuple(len = n_layers+1) of (batch, seq, hidden)
        last = torch.stack([h[0, -1, :].float().cpu() for h in out.hidden_states])
        return last

    def logit_lens_token_probs(self, hidden_per_layer, token_ids: list[int]):
        """Project central-layer hidden states through the unembedding (logit
        lens) and return softmax probability mass on ``token_ids`` per layer.

        This is the mechanism behind the §4.2 "internal emotions" measurement:
        how strongly emotional tokens are promoted in central layers.
        """
        import torch

        self._ensure_loaded()
        model = self._model.get_base_model() if hasattr(self._model, "get_base_model") else self._model
        lm_head = model.get_output_embeddings()
        norm = getattr(model.model, "norm", None)
        probs_per_layer = []
        with torch.no_grad():
            for h in hidden_per_layer:
                h = h.to(lm_head.weight.device, lm_head.weight.dtype)
                if norm is not None:
                    h = norm(h)
                logits = lm_head(h)
                p = torch.softmax(logits, dim=-1)
                probs_per_layer.append(float(p[token_ids].sum()))
        return probs_per_layer
