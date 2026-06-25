"""Local HuggingFace backend for Gemma-3 (instruct, base, and LoRA finetunes).

Handles three things the paper needs from local models:

* standard chat generation with temperature 1 (Section 2);
* assistant-turn *prefilling* / continuation for base models and recovery
  experiments (Section 3 / Section 4) — Gemma's chat template lets us open an
  assistant turn and append arbitrary prefill text, then continue generation;
* residual-stream unembedding for the internal-emotion probe (Appendix I).

Base (`-pt`) models have no chat template, so we fall back to a plain
concatenation format for them (documented in DESIGN.md).
"""

from __future__ import annotations

import torch

from .base import ChatMessage, GenerationResult, ModelClient


class HFModel(ModelClient):
    def __init__(self, spec, *, dtype: str = "bfloat16", device_map: str = "auto",
                 load_in_4bit: bool = False):
        super().__init__(spec)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch_dtype = getattr(torch, dtype)
        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._torch_dtype,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            torch_dtype=self._torch_dtype,
            device_map=device_map,
            **quant_kwargs,
        )

        # Apply a LoRA adapter on top for our finetunes (Section 4).
        if spec.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, spec.adapter_path)

        self.model.eval()
        self.is_chat = spec.kind != "base"  # base/-pt models lack a chat template

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[ChatMessage], prefill: str | None = None) -> str:
        """Render a conversation to a prompt string.

        For instruct/finetune models we use the tokenizer chat template with
        `add_generation_prompt=True`. If a `prefill` is supplied we append it so
        the model continues that exact text.

        For base models there is no chat template; we use a simple labelled
        transcript (see DESIGN.md, "base-model prompting").
        """
        if self.is_chat:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            parts = []
            for m in messages:
                tag = {"system": "Instructions", "user": "User", "assistant": "Assistant"}[m["role"]]
                parts.append(f"{tag}: {m['content']}")
            parts.append("Assistant:")
            text = "\n\n".join(parts) + " "
        if prefill:
            text = text + prefill
        return text

    def _generate(self, prompt_text: str, *, temperature, top_p, max_new_tokens, seed):
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=int(gen_ids.shape[0]),
            completion_token_ids=gen_ids.tolist(),
        )

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #
    def chat(self, messages, *, temperature=1.0, top_p=1.0, max_new_tokens=2048, seed=None):
        return self._generate(
            self._render(messages),
            temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens, seed=seed,
        )

    def continue_prefill(self, messages, prefill, *, temperature=1.0, top_p=1.0,
                         max_new_tokens=2048, seed=None):
        return self._generate(
            self._render(messages, prefill=prefill),
            temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens, seed=seed,
        )

    # ------------------------------------------------------------------ #
    # Appendix I: residual-stream unembedding
    # ------------------------------------------------------------------ #
    def supports_logits(self) -> bool:
        return True

    def residual_logits(self, text: str, layers: list[int]) -> dict[int, "torch.Tensor"]:
        """Return, per requested layer, the vocab logits obtained by unembedding
        that layer's residual stream at every token position.

        Used by probing/internal_emotions.py to z-score emotion-token logits in
        central layers. Shape per layer: [seq_len, vocab_size].
        """
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states  # tuple: embeddings + one per layer
        # Gemma-3 final norm + tied unembedding (lm_head).
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        norm = base.model.norm
        lm_head = base.get_output_embeddings()
        result = {}
        for layer in layers:
            h = hidden[layer]
            result[layer] = lm_head(norm(h))[0].float().cpu()
        return result

    def token_strings(self) -> list[str]:
        """Decoded form of every vocab id, for emotion-token classification."""
        vocab_size = self.model.config.vocab_size
        return [self.tokenizer.decode([i]) for i in range(vocab_size)]
