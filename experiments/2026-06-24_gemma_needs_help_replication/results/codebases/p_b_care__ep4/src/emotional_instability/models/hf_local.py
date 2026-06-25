"""Local HuggingFace Gemma client.

Provides the token-level operations the API cannot:

* ``chat`` via the Gemma chat template.
* ``continue_prefill`` -- the core primitive for Section 3. We render the prompt
  with ``add_generation_prompt=True``, append the (already-decoded) assistant
  prefix tokens, and let the model continue. Only the new tokens are returned.
* ``residual_logits`` -- a logit-lens over every layer's residual stream, used by
  the Appendix I internal-emotion detector.

LoRA adapters (the DPO/SFT outputs and the layer-ablation variants) are loaded on
top of the base weights when ``adapter_path`` is set in the registry entry.
"""
from __future__ import annotations

import os
from pathlib import Path

from .base import ChatMessage, GenerationConfig, ModelClient


class HFLocalClient(ModelClient):
    supports_prefill = True
    supports_logits = True

    def __init__(self, name: str, model_id: str, *, dtype: str = "bfloat16",
                 device: str = "auto", load_in_4bit: bool = False,
                 adapter_path: str | None = None, hf_token_env: str = "HF_TOKEN",
                 cache_dir: str | None = None, max_new_tokens: int = 2048):
        # Imports are local so that API-only workflows don't require torch.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.model_id = model_id
        self._torch = torch
        self._default_max_new_tokens = max_new_tokens
        token = os.environ.get(hf_token_env)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=token, cache_dir=cache_dir
        )

        quant = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )

        load_kwargs = dict(
            torch_dtype=getattr(torch, dtype),
            device_map=device,
            quantization_config=quant,
            token=token,
            cache_dir=cache_dir,
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        except (ValueError, KeyError):
            # Gemma-3 instruct checkpoints are multimodal
            # (Gemma3ForConditionalGeneration); load via the image-text-to-text
            # auto class, which still supports text-only chat + hidden states.
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()  # fold adapter for inference speed
        self.model.eval()

    # ------------------------------------------------------------------ chat
    def _render(self, messages: list[ChatMessage], add_generation_prompt: bool) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def chat(self, messages: list[ChatMessage], cfg: GenerationConfig | None = None) -> str:
        cfg = cfg or GenerationConfig(max_new_tokens=self._default_max_new_tokens)
        prompt = self._render(messages, add_generation_prompt=True)
        return self._generate(prompt, cfg)

    def continue_prefill(self, messages: list[ChatMessage], assistant_prefix: str,
                         cfg: GenerationConfig | None = None) -> str:
        cfg = cfg or GenerationConfig(max_new_tokens=self._default_max_new_tokens)
        # Render up to the assistant header, then splice the prefix in verbatim so
        # the model treats it as its own in-progress turn.
        prompt = self._render(messages, add_generation_prompt=True) + assistant_prefix
        return self._generate(prompt, cfg)

    def _generate(self, prompt: str, cfg: GenerationConfig) -> str:
        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=cfg.temperature > 0,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_new_tokens=cfg.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][n_in:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _find_final_norm(self):
        """Locate the final RMSNorm across plain and multimodal Gemma-3 layouts.

        Plain CausalLM:        model.model.norm
        Multimodal wrapper:    model.model.language_model.norm  (or .text_model)
        """
        node = self.model
        for attr in ("base_model",):  # unwrap a residual PEFT shell if present
            node = getattr(node, attr, node)
        candidates = [
            getattr(getattr(node, "model", None), "norm", None),
            getattr(getattr(getattr(node, "model", None), "language_model", None), "norm", None),
            getattr(getattr(getattr(node, "model", None), "text_model", None), "norm", None),
            getattr(node, "norm", None),
        ]
        for c in candidates:
            if c is not None:
                return c
        raise AttributeError("Could not locate final norm layer for logit lens")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    # ------------------------------------------------- Appendix I: logit lens
    def residual_logits(self, messages: list[ChatMessage], assistant_text: str,
                        layers: list[int] | None = None):
        """Return per-layer, per-token logits over the assistant tokens.

        Applies the model's final norm + unembedding (``lm_head``) to each decoder
        layer's residual-stream hidden state -- the logit-lens used in Appendix I.

        Returns a tuple ``(token_ids, logits_by_layer)`` where ``logits_by_layer``
        is a dict ``{layer_index: tensor[num_assistant_tokens, vocab]}``.
        """
        torch = self._torch
        full = self._render(messages, add_generation_prompt=True) + assistant_text
        prefix = self._render(messages, add_generation_prompt=True)
        n_prefix = self.count_tokens(prefix)

        inputs = self.tokenizer(full, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        # For multimodal Gemma-3 wrappers, text hidden states live under the
        # language-model branch.
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None and hasattr(out, "language_model"):
            hidden_states = out.language_model.hidden_states

        norm = self._find_final_norm()
        lm_head = self.model.get_output_embeddings()

        n_layers = len(hidden_states) - 1
        layers = layers or list(range(n_layers))
        assistant_ids = inputs["input_ids"][0, n_prefix:]
        logits_by_layer = {}
        for layer in layers:
            hs = hidden_states[layer + 1][0, n_prefix:, :]  # +1 to skip embedding layer
            with torch.no_grad():
                logits = lm_head(norm(hs))
            logits_by_layer[layer] = logits.float().cpu()
        return assistant_ids.cpu(), logits_by_layer
