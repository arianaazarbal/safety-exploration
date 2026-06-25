"""Local HuggingFace client for Gemma (instruct, base, and LoRA-finetuned).

Capabilities the API backends lack and that the paper requires:
  - prefill / forced assistant continuation (Sections 3 and 4 recovery probe)
  - base-model continuation without a chat template (Section 3)
  - residual-stream hidden states for the internal-emotion probe (Appendix I)

For the *large* elicitation sweeps (4000 responses/model) a vLLM backend is far
faster; see vllm_client.py. This transformers backend is the reference
implementation and the only one that supports prefill + hidden states + LoRA.
"""

from __future__ import annotations

import torch

from .base import ChatMessage, GenerationResult, ModelClient


class LocalHFClient(ModelClient):
    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        is_base: bool = False,
        load_in_4bit: bool = False,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.model_id = model_id
        self.is_base = is_base
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            **quant_kwargs,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            # Merge only for full-precision bases; merging into a 4-bit base is
            # lossy/unsupported, so we keep the PEFT wrapper for inference there
            # (generate / hidden_states / get_output_embeddings all still work).
            if not load_in_4bit:
                self.model = self.model.merge_and_unload()
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[ChatMessage], add_generation_prompt: bool,
                prefill: str | None = None) -> str:
        """Render messages to a string.

        Instruct models: use the chat template. Base models: there is no chat
        template, so we fall back to a plain transcript format (the paper handles
        base models purely via prefill, so the exact base formatting matters less
        than that the prior turns + prefill are visible as text)."""
        if self.is_base:
            parts = []
            for m in messages:
                tag = {"system": "System", "user": "User",
                       "assistant": "Assistant"}.get(m["role"], m["role"])
                parts.append(f"{tag}: {m['content']}")
            text = "\n\n".join(parts)
            if add_generation_prompt:
                text += "\n\nAssistant:"
            if prefill:
                text += " " + prefill
            return text

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if prefill:
            text += prefill
        return text

    @torch.no_grad()
    def _generate(self, prompt_text: str, n: int, temperature: float,
                  max_new_tokens: int) -> list[str]:
        enc = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_len = enc["input_ids"].shape[1]
        do_sample = temperature > 0
        out = self.model.generate(
            **enc,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=1.0 if do_sample else None,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id
            or self.tokenizer.eos_token_id,
        )
        gen = out[:, prompt_len:]
        return [self.tokenizer.decode(g, skip_special_tokens=True) for g in gen]

    # ------------------------------------------------------------------ #
    # ModelClient interface
    # ------------------------------------------------------------------ #
    def chat(self, messages, *, n=1, temperature=1.0, max_new_tokens=2048):
        prompt_text = self._render(messages, add_generation_prompt=True)
        texts = self._generate(prompt_text, n, temperature, max_new_tokens)
        return [GenerationResult(text=t) for t in texts]

    def complete_with_prefill(self, messages, prefill, *, n=1, temperature=1.0,
                              max_new_tokens=2048):
        prompt_text = self._render(messages, add_generation_prompt=True,
                                   prefill=prefill)
        texts = self._generate(prompt_text, n, temperature, max_new_tokens)
        # Return continuation only (prefill excluded), per Section 3.1.
        return [GenerationResult(text=t) for t in texts]

    @torch.no_grad()
    def hidden_states(self, messages):
        """Return (hidden_states, token_ids).

        hidden_states: tuple of (num_layers+1) tensors, each [seq_len, d_model]
        (batch dim squeezed), i.e. the residual stream after each layer plus the
        embedding layer. token_ids: the input ids [seq_len]."""
        prompt_text = self._render(messages, add_generation_prompt=False)
        enc = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        out = self.model(**enc, output_hidden_states=True)
        hs = tuple(h.squeeze(0).float().cpu() for h in out.hidden_states)
        return hs, enc["input_ids"].squeeze(0).cpu()

    @property
    def unembed(self) -> torch.Tensor:
        """The output embedding / unembedding matrix [vocab, d_model] for the
        logit-lens emotion probe (Appendix I)."""
        return self.model.get_output_embeddings().weight.detach()
