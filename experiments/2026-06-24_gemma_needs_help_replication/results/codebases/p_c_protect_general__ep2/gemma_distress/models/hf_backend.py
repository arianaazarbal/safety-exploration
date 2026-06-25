"""HuggingFace transformers backend for local Gemma inference.

Supports the three things the paper needs from a *white-box* model:
  1. Multi-turn chat generation (Section 2).
  2. Prefill continuation - generate from a partially written assistant turn
     (Sections 3 & 4 recovery). Base (`-pt`) models are handled by rendering the
     conversation through the same chat template and prefilling; this is exactly
     the mechanism the paper uses to make base models "consistently continue".
  3. Residual-stream extraction + unembedding for the logit-lens emotion probe
     (Appendix I).

A finetuned LoRA adapter can be attached via `adapter_path`.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import GenConfig, ModelBackend, Turn


class HFBackend(ModelBackend):
    supports_prefill = True
    supports_hidden_states = True

    def __init__(
        self,
        name: str,
        hf_id: str,
        family: str = "gemma",
        kind: str = "instruct",
        load_in_4bit: bool = False,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        adapter_path: Optional[str] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.family = family
        self.kind = kind
        self.hf_id = hf_id

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model_kwargs: dict = {"device_map": device_map, "torch_dtype": getattr(torch, dtype)}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)
        self.model.eval()

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()

        # Base models lack a chat template; fall back to a simple, consistent render.
        self._has_chat_template = self.tokenizer.chat_template is not None

    # ----- prompt rendering ------------------------------------------------ #
    def _render_prompt(self, messages: list[Turn], add_generation_prompt: bool = True) -> str:
        if self._has_chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        # Plaintext fallback for base models without a chat template.
        lines = []
        for m in messages:
            role = {"user": "User", "assistant": "Assistant", "system": "System"}[m["role"]]
            lines.append(f"{role}: {m['content']}")
        if add_generation_prompt:
            lines.append("Assistant:")
        return "\n".join(lines)

    # ----- generation ------------------------------------------------------ #
    @torch.no_grad()
    def _generate(self, prompt_text: str, gen: GenConfig) -> str:
        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model.generate(
            **inputs,
            do_sample=gen.temperature > 0,
            temperature=gen.temperature if gen.temperature > 0 else None,
            top_p=gen.top_p,
            max_new_tokens=gen.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def chat(self, messages: list[Turn], gen: GenConfig | None = None) -> str:
        gen = gen or GenConfig()
        prompt = self._render_prompt(messages, add_generation_prompt=True)
        return self._generate(prompt, gen).strip()

    def prefill_continue(
        self, messages: list[Turn], prefill: str, gen: GenConfig | None = None
    ) -> str:
        gen = gen or GenConfig()
        # Open the assistant turn, then inject the prefill so generation continues it.
        prompt = self._render_prompt(messages, add_generation_prompt=True) + prefill
        return self._generate(prompt, gen)

    # ----- tokenisation ---------------------------------------------------- #
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    # ----- interpretability ------------------------------------------------ #
    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    def vocab_strings(self) -> list[str]:
        vocab = self.tokenizer.get_vocab()  # token -> id
        ordered = [None] * (max(vocab.values()) + 1)
        for tok, idx in vocab.items():
            ordered[idx] = self.tokenizer.convert_tokens_to_string([tok])
        return ordered

    @torch.no_grad()
    def hidden_states_and_tokens(self, messages: list[Turn], prefill: str = ""):
        """Return (hidden_states, token_strings, assistant_mask).

        hidden_states: tensor [n_layers+1, seq, d_model] (embeddings + each block).
        assistant_mask: bool tensor [seq], True for positions in the final assistant
        turn (the part we want to probe).
        """
        prompt = self._render_prompt(messages, add_generation_prompt=True)
        full = prompt + prefill
        enc = self.tokenizer(full, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))

        out = self.model(**enc, output_hidden_states=True)
        hidden = torch.stack(out.hidden_states, dim=0)[:, 0]  # [L+1, seq, d]

        seq = enc["input_ids"].shape[1]
        mask = torch.zeros(seq, dtype=torch.bool)
        mask[prompt_len:] = True  # assistant continuation positions
        tok_strings = [
            self.tokenizer.convert_tokens_to_string([t])
            for t in self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
        ]
        return hidden.float().cpu(), tok_strings, mask

    @torch.no_grad()
    def apply_final_norm(self, hidden: "torch.Tensor") -> "torch.Tensor":
        """Apply the model's final RMSNorm so a logit lens is calibrated."""
        norm = self.model.get_decoder().norm if hasattr(self.model, "get_decoder") else None
        if norm is None:  # fall back to common attribute paths
            norm = getattr(self.model.model, "norm", None)
        if norm is None:
            return hidden
        param = next(norm.parameters())
        out = norm(hidden.to(param.device, dtype=param.dtype))
        return out.float().cpu()

    def unembed_matrix(self) -> "torch.Tensor":
        return self.model.get_output_embeddings().weight.detach()

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
