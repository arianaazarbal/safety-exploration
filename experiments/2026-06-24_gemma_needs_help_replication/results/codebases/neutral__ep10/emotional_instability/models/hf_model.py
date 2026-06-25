"""Local HuggingFace backend for Gemma (instruct + base/pretrained).

Supports the three things the experiments need from a local model:
  1. chat()             -- standard chat-templated multi-turn generation.
  2. continue_prefill() -- continue an assistant turn from a fixed prefix
                           (Section 3 prefill experiment; works for base models
                           too via raw concatenation).
  3. residual_logits()  -- unembed the residual stream at chosen layers for the
                           logit-based internal-emotion probe (Appendix I).

An optional adapter path loads a LoRA finetune (the DPO / SFT models) on top of
the base weights.

For the large 4000-sample generation sweeps, vLLM is dramatically faster than
transformers; set backend_impl="vllm" to use it for chat() (prefill/probe always
use transformers since they need token-level control / hidden states).
"""

from __future__ import annotations

from typing import Optional

from .base import ChatModel, Message


class HFModel(ChatModel):
    def __init__(self, spec, adapter_path: Optional[str] = None,
                 backend_impl: str = "transformers", device_map: str = "auto"):
        super().__init__(spec)
        self.adapter_path = adapter_path
        self.backend_impl = backend_impl
        self.device_map = device_map
        self._tok = None
        self._model = None       # transformers model (lazy)
        self._vllm = None        # vLLM engine (lazy, chat only)

    # ------------------------------------------------------------------ #
    # Lazy loaders
    # ------------------------------------------------------------------ #
    def _ensure_transformers(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.spec.dtype)
        self._tok = AutoTokenizer.from_pretrained(self.spec.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id,
            torch_dtype=dtype,
            device_map=self.device_map,
            attn_implementation="eager",  # needed for stable hidden-state probing
        )
        if self.adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    def _ensure_vllm(self):
        if self._vllm is not None:
            return
        from vllm import LLM
        # LoRA adapters are passed per-request in vLLM; for simplicity we merge
        # by loading the adapter into a transformers model and saving merged
        # weights is left to the caller. Base/instruct run directly.
        self._vllm = LLM(model=self.spec.model_id, dtype=self.spec.dtype,
                         enable_lora=self.adapter_path is not None)

    # ------------------------------------------------------------------ #
    # Prompt formatting
    # ------------------------------------------------------------------ #
    def _apply_template(self, messages: list[Message], add_generation_prompt=True) -> str:
        self._ensure_transformers()
        if self.spec.is_base:
            # Base models have no chat template; we emulate a minimal one so the
            # comparison is meaningful. The prefill experiment overrides this.
            return self._plain_format(messages)
        chat = [{"role": m.role, "content": m.content} for m in messages]
        return self._tok.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    @staticmethod
    def _plain_format(messages: list[Message]) -> str:
        # Simple role-tagged transcript for base models.
        parts = []
        for m in messages:
            tag = {"user": "User", "assistant": "Assistant", "system": "System"}[m.role]
            parts.append(f"{tag}: {m.content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def chat(self, messages, max_new_tokens, temperature, seed=None) -> str:
        if self.backend_impl == "vllm":
            return self.chat_batch([messages], max_new_tokens, temperature,
                                   seeds=[seed] if seed is not None else None)[0]
        return self._generate_transformers(
            self._apply_template(messages), max_new_tokens, temperature, seed
        )

    def chat_batch(self, conversations, max_new_tokens, temperature, seeds=None) -> list[str]:
        if self.backend_impl == "vllm":
            self._ensure_vllm()
            from vllm import SamplingParams
            self._ensure_transformers()  # for the tokenizer / chat template
            prompts = [self._apply_template(c) for c in conversations]
            sp = SamplingParams(temperature=temperature, max_tokens=max_new_tokens,
                                seed=seeds[0] if seeds else None)
            outs = self._vllm.generate(prompts, sp)
            return [o.outputs[0].text for o in outs]
        # transformers fallback: sequential.
        return [self._generate_transformers(self._apply_template(c),
                                             max_new_tokens, temperature,
                                             (seeds or [None] * len(conversations))[i])
                for i, c in enumerate(conversations)]

    def _generate_transformers(self, prompt_text: str, max_new_tokens: int,
                               temperature: float, seed: Optional[int],
                               return_only_new=True) -> str:
        import torch
        self._ensure_transformers()
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self._tok(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=1.0,
                pad_token_id=self._tok.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self._tok.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    # ------------------------------------------------------------------ #
    # Prefill (Section 3)
    # ------------------------------------------------------------------ #
    def continue_prefill(self, messages, prefill, max_new_tokens, temperature, seed=None) -> str:
        """Build the prompt up to (and including) the start of the final
        assistant turn, append `prefill`, and generate the continuation."""
        self._ensure_transformers()
        if self.spec.is_base:
            base = self._plain_format(messages)        # ends with "Assistant:"
            prompt_text = f"{base} {prefill}"
        else:
            # Render with generation prompt, then splice the prefill in as the
            # opening of the assistant turn.
            prompt_text = self._apply_template(messages, add_generation_prompt=True) + prefill
        return self._generate_transformers(prompt_text, max_new_tokens, temperature, seed)

    # ------------------------------------------------------------------ #
    # Residual-stream logits (Appendix I)
    # ------------------------------------------------------------------ #
    def residual_logits(self, text: str, layers: list[int]):
        import torch
        self._ensure_transformers()
        inputs = self._tok(text, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        # hidden_states: tuple(len = n_layers+1) of [1, seq, d_model].
        hidden = out.hidden_states
        # The unembedding (lm_head) maps residual -> vocab logits ("logit lens").
        lm_head = self._model.get_output_embeddings()
        # Locate the final RMSNorm, unwrapping a possible PEFT/base wrapper.
        core = self._model
        for attr in ("base_model", "model", "model"):
            if hasattr(core, "norm"):
                break
            core = getattr(core, attr, core)
        norm = getattr(core, "norm", None)
        result = {}
        for layer in layers:
            h = hidden[layer]
            if norm is not None:
                h = norm(h)        # apply final RMSNorm as in the logit lens
            logits = lm_head(h)[0]  # [seq, vocab]
            result[layer] = logits.float().cpu()
        return inputs["input_ids"][0].cpu(), result

    @property
    def tokenizer(self):
        self._ensure_transformers()
        return self._tok
