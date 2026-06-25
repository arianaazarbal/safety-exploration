"""Gemma client backed by HuggingFace transformers.

Handles instruct *and* base ("-pt") checkpoints, optional LoRA adapters
(finetuned variants), true response prefilling (Section 3 / recovery), and
residual-stream unembedding for the logit-based internal-emotion probe
(Appendix I). Heavy deps (torch, transformers, peft) are imported lazily so this
module can be imported in environments without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import GenConfig, Message, ModelClient


@dataclass
class GemmaLoadOptions:
    hf_id: str
    adapter_path: str | None = None      # LoRA adapter dir (finetuned variants)
    dtype: str = "bfloat16"
    device_map: str = "auto"
    load_in_4bit: bool = False           # for fitting 27B on smaller GPUs
    attn_implementation: str = "eager"   # Gemma 3 recommends eager attention


class GemmaClient(ModelClient):
    def __init__(
        self,
        name: str,
        opts: GemmaLoadOptions,
        family: str | None = "gemma",
        is_base: bool = False,
    ):
        super().__init__(name=name, family=family)
        self.opts = opts
        self.is_base = is_base
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------ load
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.opts.dtype)
        quant_kwargs: dict[str, Any] = {}
        if self.opts.load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.opts.hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.opts.hf_id,
            torch_dtype=dtype,
            device_map=self.opts.device_map,
            attn_implementation=self.opts.attn_implementation,
            **quant_kwargs,
        )
        if self.opts.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.opts.adapter_path)
        model.eval()
        self._model = model

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    def supports_prefill(self) -> bool:
        return True

    # ------------------------------------------------------------ rendering
    def render_chat(self, messages: list[Message], add_generation_prompt: bool = True) -> str:
        """Render chat history to a raw prompt string via the Gemma chat template.

        For base ("-pt") models there is no chat template; we fall back to a
        simple turn-delimited rendering that mirrors the instruct format closely
        enough for prefilling continuations (Section 3 prefills the assistant
        turn explicitly, so exact special tokens matter less than consistency)."""
        tok = self.tokenizer
        if not self.is_base and tok.chat_template:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        # Base-model fallback rendering.
        parts = []
        for m in messages:
            parts.append(f"<start_of_turn>{m['role']}\n{m['content']}<end_of_turn>")
        if add_generation_prompt:
            parts.append("<start_of_turn>model\n")
        return "\n".join(parts)

    # ----------------------------------------------------------------- chat
    def chat(self, messages: list[Message], cfg: GenConfig) -> str:
        prompt = self.render_chat(messages, add_generation_prompt=True)
        return self.complete(prompt, cfg, prefix="")

    # --------------------------------------------------------- completion
    def complete(self, prompt: str, cfg: GenConfig, prefix: str = "") -> str:
        """Generate a continuation. ``prefix`` is appended to the prompt and the
        model continues from it; the returned text excludes both prompt and
        prefix (this is how base models are made to continue emotional
        trajectories in Section 3)."""
        import torch

        self._ensure_loaded()
        full = prompt + prefix
        inputs = self._tokenizer(full, return_tensors="pt").to(self._model.device)
        n_prompt = inputs["input_ids"].shape[1]

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            temperature=cfg.temperature if cfg.temperature > 0 else None,
            top_p=cfg.top_p,
            pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = self._model.generate(**inputs, **{k: v for k, v in gen_kwargs.items() if v is not None})
        gen_ids = out[0][n_prompt:]
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    # ----------------------------------------------- internal probing (App. I)
    def residual_logits(self, text: str, layers: list[int] | None = None,
                        vocab_subset: list[int] | None = None):
        """Return per-layer unembedded logits for every token in ``text``.

        For the logit-based emotion probe we run a forward pass with
        ``output_hidden_states=True``, apply the model's final norm + unembedding
        (lm_head) to each layer's residual stream, and return a tensor of shape
        ``[n_layers, seq_len, n_kept]`` plus the token ids of the sequence.

        ``vocab_subset`` restricts the returned columns to those vocab ids (the
        emotion + control tokens), keeping memory tractable — the full vocab is
        ~256k wide, so without this the tensor would be enormous. When given, the
        returned tensor's column j corresponds to ``vocab_subset[j]``.
        """
        import torch

        self._ensure_loaded()
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states  # tuple: embeddings + one per layer
        base = self._model.get_base_model() if hasattr(self._model, "get_base_model") else self._model
        lm_head = base.get_output_embeddings()
        norm = base.model.norm  # Gemma final RMSNorm

        sel = layers if layers is not None else list(range(1, len(hidden)))
        idx = None
        if vocab_subset is not None:
            idx = torch.tensor(vocab_subset, device=self._model.device)

        logits_per_layer = []
        with torch.no_grad():
            for layer_idx in sel:
                h = hidden[layer_idx]
                logits = lm_head(norm(h)).squeeze(0)          # [seq, vocab]
                if idx is not None:
                    logits = logits.index_select(-1, idx)     # [seq, n_kept]
                logits_per_layer.append(logits.float().cpu())
        token_ids = inputs["input_ids"].squeeze(0).cpu().tolist()
        return torch.stack(logits_per_layer), token_ids
