"""Local HuggingFace transformers backend for Gemma (instruct, base, finetuned).

Supports:
  * chat() / chat_batch() via the model's chat template (instruct models)
  * prefill_continue() for the Section 3 prefill experiment -- works for both
    instruct (prefilled assistant turn) and base (raw continuation) models
  * optional LoRA adapter loading (for evaluating DPO/SFT finetunes)
  * a hook to grab residual-stream activations (used by Appendix I)

Heavy imports (torch/transformers/peft) are done lazily inside __init__ so that
API-only workflows don't need them installed.
"""
from __future__ import annotations

from typing import Optional

from .base import GenerationConfig, Message, ModelClient


class HFModel(ModelClient):
    def __init__(self, spec, device_map: str = "auto", adapter_path: Optional[str] = None):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        hf_id = spec.get("hf_id")
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[spec.get("dtype", "bfloat16")]

        load_kwargs = dict(torch_dtype=dtype, device_map=device_map)
        if spec.get("load_in_4bit"):
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)

        # Resolve adapter: explicit arg wins, else config's adapter_path.
        adapter = adapter_path or spec.get("adapter_path")
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model = self.model.merge_and_unload()

        self.model.eval()
        self.is_base = spec.kind == "base"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding so batched generation aligns at the right edge.
        self.tokenizer.padding_side = "left"

    # ---- prompt construction ---------------------------------------------
    def _render(self, messages: list[Message], add_generation_prompt: bool = True) -> str:
        """Render messages to a single string.

        Instruct models use the chat template. Base models have no chat
        template, so we fall back to a plain transcript -- but in practice base
        models are only ever called through prefill_continue (Section 3), which
        builds its own minimal context. See DESIGN.md.
        """
        if not self.is_base and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        # Base-model fallback transcript.
        parts = []
        for m in messages:
            parts.append(f"{m['role'].capitalize()}: {m['content']}")
        if add_generation_prompt:
            parts.append("Assistant:")
        return "\n".join(parts)

    def _gen(self, prompt_text: str, cfg: GenerationConfig) -> str:
        return self._gen_batch([prompt_text], cfg)[0]

    def _gen_batch(self, prompt_texts: list[str], cfg: GenerationConfig) -> list[str]:
        torch = self.torch
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        enc = self.tokenizer(
            prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)
        do_sample = cfg.temperature and cfg.temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=do_sample,
                temperature=cfg.temperature if do_sample else None,
                top_p=cfg.top_p if do_sample else None,
                max_new_tokens=cfg.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Strip the prompt tokens; decode only the continuation.
        gen = out[:, enc["input_ids"].shape[1]:]
        return [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]

    # ---- public API ------------------------------------------------------
    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        return self._gen(self._render(messages), cfg)

    def chat_batch(self, batch: list[list[Message]], cfg: GenerationConfig) -> list[str]:
        return self._gen_batch([self._render(m) for m in batch], cfg)

    def supports_prefill(self) -> bool:
        return True

    def prefill_continue(
        self, messages: list[Message], prefill: str, cfg: GenerationConfig
    ) -> str:
        """Continue from a prefilled assistant turn.

        For instruct models we render the chat template with a generation prompt,
        then append the prefill text and let the model continue. For base models
        we append the prefill to the plain transcript. In both cases we return
        only the new tokens beyond the prefill.
        """
        base = self._render(messages, add_generation_prompt=True)
        full_prompt = base + prefill
        return self._gen(full_prompt, cfg)

    # ---- activation access (Appendix I) ----------------------------------
    def residual_stream(self, text: str):
        """Return (input_ids, hidden_states) for `text`.

        hidden_states is a tuple of (num_layers+1) tensors of shape
        [seq, hidden]; index 0 is the embedding output. Used by the logit-based
        internal-emotion detector.
        """
        torch = self.torch
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        hs = tuple(h[0] for h in out.hidden_states)  # drop batch dim
        return enc["input_ids"][0], hs

    def unembed(self, hidden):
        """Apply final norm + lm_head to a [.., hidden] tensor -> logits."""
        torch = self.torch
        model = self.model
        with torch.no_grad():
            # Gemma: model.model.norm then lm_head (tied embeddings).
            normed = model.model.norm(hidden)
            logits = model.lm_head(normed)
        return logits
