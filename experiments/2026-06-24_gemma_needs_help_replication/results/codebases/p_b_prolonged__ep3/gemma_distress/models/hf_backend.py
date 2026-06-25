"""Local Gemma backend via HuggingFace ``transformers``.

Handles both instruct (``-it``) and pretrained (``-pt``) Gemma 3 checkpoints,
optionally with a PEFT/LoRA adapter (the DPO/SFT finetunes). This backend is the
one used for probing, because it can expose hidden states and logits.

Why transformers and not vLLM as the default: the probing experiments
(Appendix I) need the residual stream and per-token logits, which vLLM does not
expose. To keep a single code path that is *correct* for every experiment we use
transformers everywhere; a vLLM fast path for bulk generation is provided
separately (``vllm_backend``) and is interchangeable for the pure-generation
experiments. See DESIGN.md §"Inference backend".
"""
from __future__ import annotations

from typing import Optional

import torch

from .base import GenerationConfig, ModelCapabilities, ModelInterface, Turn


class HFGemmaModel(ModelInterface):
    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        is_base_model: bool = False,
        adapter_path: Optional[str] = None,
        device: str = "auto",
        dtype: str = "bfloat16",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.hf_id = hf_id
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device,
            output_hidden_states=False,
        )
        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.capabilities = ModelCapabilities(
            supports_internal_states=True,
            supports_prefill=True,
            is_base_model=is_base_model,
        )

    # --------------------------------------------------------------------- #
    # Prompt construction
    # --------------------------------------------------------------------- #
    def _render_chat(self, messages: list[Turn], add_generation_prompt: bool) -> str:
        """Render a conversation to a prompt string.

        Instruct models use the Gemma chat template. Base models have no chat
        template; we fall back to a plain ``Role: content`` transcript so that
        prefilling still works (Section 3.1 deliberately uses prefills precisely
        because base models are not chat-tuned).
        """
        if self.capabilities.is_base_model:
            # Plain transcript; base models simply continue text.
            lines = [f"{t.role}: {t.content}" for t in messages]
            if add_generation_prompt:
                lines.append("assistant:")
            return "\n".join(lines)

        chat = [{"role": t.role, "content": t.content} for t in messages]
        return self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    # --------------------------------------------------------------------- #
    # Generation
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def _generate(self, prompt: str, cfg: GenerationConfig) -> list[str]:
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        out = self.model.generate(
            **inputs,
            do_sample=cfg.temperature > 0,
            temperature=cfg.temperature,
            top_p=1.0,                      # CHOICE: paper sets only temperature
            max_new_tokens=cfg.max_new_tokens,
            num_return_sequences=cfg.n,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        gen = out[:, prompt_len:]
        return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in gen]

    def chat(self, messages: list[Turn], cfg: GenerationConfig) -> list[str]:
        prompt = self._render_chat(messages, add_generation_prompt=True)
        return self._generate(prompt, cfg)

    def continue_from(
        self, messages: list[Turn], prefill: str, cfg: GenerationConfig
    ) -> list[str]:
        # Build the conversation up to (and including the start of) the final
        # assistant turn, then append the prefill text so generation continues it.
        prompt = self._render_chat(messages, add_generation_prompt=True) + prefill
        return self._generate(prompt, cfg)

    # --------------------------------------------------------------------- #
    # Probing hooks (Appendix I) — see probing/logit_detector.py
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def forward_with_hidden_states(self, text: str):
        """Run a forward pass returning per-layer residual-stream activations.

        Returns ``(input_ids, hidden_states)`` where ``hidden_states`` is a tuple
        of ``(1, seq, d_model)`` tensors, one per layer (index 0 = embeddings).
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model(**inputs, output_hidden_states=True)
        return inputs["input_ids"][0], out.hidden_states

    def unembed(self, residual: "torch.Tensor") -> "torch.Tensor":
        """Project a residual-stream vector to vocab logits (logit lens).

        Applies the model's final norm then the LM head, matching how the model
        would read out at that layer.
        """
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        normed = base.model.norm(residual)
        return base.lm_head(normed)
