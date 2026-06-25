"""Local HuggingFace inference for Gemma models (instruct + base/pretrained),
with optional LoRA adapters for the finetuned variants.

Key behaviours required by the paper:
  * temperature-1 sampling (Section 2.1);
  * prefilling assistant turns so *base* models continue a response, and for the
    recovery experiment (Sections 3.1, 4.2);
  * exposing per-layer residual-stream logits for internal-emotion detection
    (Appendix I) — see ``unembed_residual``.

Gemma's chat template has no dedicated system role, so any system message is
prepended to the first user turn (see ``_render``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import ADAPTER_DIRS, hf_token
from .base import ChatModel, Message


class HFLocalModel(ChatModel):
    def __init__(self, spec, device_map: str = "auto", dtype: str = "bfloat16",
                 load_in_4bit: bool = False, lora_layers: Optional[list[int]] = None):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        token = hf_token() or None
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, token=token)

        load_kwargs = dict(
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            token=token,
        )
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4")

        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **load_kwargs)

        # Attach a LoRA adapter for finetuned variants.
        adapter_dir = ADAPTER_DIRS.get(spec.key)
        if adapter_dir is not None and Path(adapter_dir).exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter_dir))
            self.model = self.model.merge_and_unload()  # fold adapter for fast inference
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[Message], add_generation_prompt: bool = True,
                prefill: str = "") -> str:
        """Render chat messages to a prompt string.

        Instruct models: apply the chat template. Base models: there is no chat
        template, so we use a lightweight plain-text transcript — the prefill
        experiment relies on the prefill text to anchor continuations anyway.
        A leading system message is folded into the first user turn for Gemma.
        """
        msgs = [m for m in messages]
        if not self.spec.supports_system and msgs and msgs[0].role == "system":
            sys = msgs.pop(0)
            if msgs and msgs[0].role == "user":
                msgs[0] = Message("user", f"{sys.content}\n\n{msgs[0].content}")
            else:
                msgs.insert(0, Message("user", sys.content))

        if self.spec.is_base:
            # Plain transcript for pretrained checkpoints.
            lines = [f"{m.role.capitalize()}: {m.content}" for m in msgs]
            text = "\n".join(lines) + "\nAssistant:"
            return text + (" " + prefill if prefill else "")

        text = self.tokenizer.apply_chat_template(
            [m.as_dict() for m in msgs],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return text + prefill  # prefill continues the open assistant turn

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _sample(self, prompt: str, *, temperature, top_p, max_new_tokens, n, seed):
        torch = self.torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        if seed is not None:
            torch.manual_seed(seed)
        gen = self.model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        outs = []
        for seq in gen:
            outs.append(self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True))
        return outs

    def generate(self, messages, *, temperature=1.0, top_p=1.0, max_new_tokens=2048,
                 n=1, seed=None):
        prompt = self._render(messages, add_generation_prompt=True)
        return self._sample(prompt, temperature=temperature, top_p=top_p,
                            max_new_tokens=max_new_tokens, n=n, seed=seed)

    def prefill_continue(self, messages, prefill, *, temperature=1.0,
                         max_new_tokens=2048, n=1, seed=None):
        prompt = self._render(messages, add_generation_prompt=True, prefill=prefill)
        # Returns continuation only (the prompt, incl. prefill, is stripped).
        return self._sample(prompt, temperature=temperature, top_p=1.0,
                            max_new_tokens=max_new_tokens, n=n, seed=seed)

    # ------------------------------------------------------------------ #
    # Interpretability hook (Appendix I)
    # ------------------------------------------------------------------ #
    def unembed_residual(self, text: str):
        """Return per-layer, per-token logits over the vocabulary obtained by
        applying the model's final norm + unembedding to each layer's residual
        stream (``output_hidden_states=True``).

        Returns a tensor of shape [num_layers+1, seq_len, vocab]. Used by
        ``src/internal/emotion_logits.py`` to z-score emotion-token logits.
        """
        torch = self.torch
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        norm = self.model.get_decoder().norm          # final RMSNorm
        lm_head = self.model.get_output_embeddings()    # tied unembedding
        logits_per_layer = []
        for h in out.hidden_states:                     # tuple len = n_layers+1
            logits_per_layer.append(lm_head(norm(h)).squeeze(0))
        return torch.stack(logits_per_layer), inputs["input_ids"].squeeze(0)

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()
