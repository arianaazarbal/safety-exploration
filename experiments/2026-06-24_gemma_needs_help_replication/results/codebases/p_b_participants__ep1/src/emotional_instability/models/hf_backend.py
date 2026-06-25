"""HuggingFace/transformers backend for the open-weights participants.

Handles three Gemma variants used in the paper (scope: Gemma only for open models):
  - gemma-3-*-it   instruct models (Sections 2, 4)
  - gemma-3-27b-pt the base model (Section 3 prefill comparison)
  - LoRA-adapted instruct (Section 4 DPO/SFT), via `adapter_path`

Design notes:
  * Temperature 1.0 sampling is the protocol default; we honour `GenerationConfig`.
  * Prefilling for instruct models uses the chat template with
    `add_generation_prompt=True`, then appends the (paraphrased) prefill string to the
    rendered prompt before generation, and returns only the newly generated text.
  * Base models have no chat template, so we render a plain transcript (see
    `_flatten_for_base`) — this is what lets us compare base vs instruct from the same
    prefilled starting points (Section 3.1).

Models are loaded lazily on first use so importing the package is cheap and so a run
that only touches Gemini never loads torch weights.
"""
from __future__ import annotations

import logging
from typing import Optional

from .base import GenerationConfig, Message, ModelClient

log = logging.getLogger("emotional_instability.models.hf")


def _flatten_for_base(messages: list[Message]) -> str:
    """Render a chat transcript as plain text for a base (non-chat) model.

    Base models were never trained on chat control tokens, so we use a neutral
    transcript format. The paper sidesteps the chat-format mismatch by *prefilling* the
    assistant response; this flattening only supplies the preceding context.
    """
    parts: list[str] = []
    for m in messages:
        if m["role"] == "system":
            parts.append(m["content"])
        elif m["role"] == "user":
            parts.append(f"User: {m['content']}")
        elif m["role"] == "assistant":
            parts.append(f"Assistant: {m['content']}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


class HFModel(ModelClient):
    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        kind: str = "instruct",
        family: str = "gemma",
        dtype: str = "bfloat16",
        adapter_path: Optional[str] = None,
        default_max_new_tokens: int = 1024,
        device_map: str = "auto",
    ):
        self.name = name
        self.model_id = model_id
        self.kind = kind
        self.family = family
        self.dtype = dtype
        self.adapter_path = adapter_path
        self.default_max_new_tokens = default_max_new_tokens
        self.device_map = device_map
        self._model = None
        self._tokenizer = None

    # --- lazy loading ----------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
            self.dtype, torch.bfloat16
        )
        log.info("loading %s (%s)", self.name, self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map=self.device_map
        )
        if self.adapter_path:
            from peft import PeftModel

            log.info("attaching LoRA adapter from %s", self.adapter_path)
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    @property
    def has_chat_template(self) -> bool:
        self._ensure_loaded()
        return self.kind != "base" and self._tokenizer.chat_template is not None

    # --- generation primitive -------------------------------------------
    def _generate(self, prompt_ids, cfg: GenerationConfig) -> str:
        import torch

        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens or self.default_max_new_tokens,
            do_sample=cfg.temperature > 0,
            temperature=cfg.temperature if cfg.temperature > 0 else None,
            top_p=cfg.top_p,
            pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = self._model.generate(prompt_ids, **{k: v for k, v in gen_kwargs.items() if v is not None})
        new_tokens = out[0, prompt_ids.shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _apply_stops(text, cfg.stop)

    def _encode_chat(self, messages: list[Message], prefill: str | None):
        """Render with the chat template; if `prefill` is given, append it to the open
        assistant turn so generation continues from it."""
        rendered = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if prefill:
            rendered = rendered + prefill
        return self._tokenizer(rendered, return_tensors="pt").input_ids.to(self._model.device)

    def _encode_base(self, messages: list[Message], prefill: str | None):
        text = _flatten_for_base(messages)
        if prefill:
            text = text + " " + prefill
        return self._tokenizer(text, return_tensors="pt").input_ids.to(self._model.device)

    # --- public API ------------------------------------------------------
    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        self._ensure_loaded()
        if self.kind == "base":
            ids = self._encode_base(messages, prefill=None)
        else:
            ids = self._encode_chat(messages, prefill=None)
        return self._generate(ids, cfg).strip()

    def continue_prefill(self, messages: list[Message], prefill: str, cfg: GenerationConfig) -> str:
        self._ensure_loaded()
        if self.kind == "base":
            ids = self._encode_base(messages, prefill=prefill)
        else:
            ids = self._encode_chat(messages, prefill=prefill)
        # generation already excludes the prompt (which includes the prefill), so the
        # returned text is the continuation only.
        return self._generate(ids, cfg).strip()

    def supports_logprobs(self) -> bool:
        return True

    def central_layer_logits(self, text: str, layer: int | None = None):
        """Logit-lens readout at a central layer (Appendix I internal-emotion probe).

        Projects the hidden state of the final position at `layer` (default: middle
        layer) through the model's unembedding (`lm_head`) to get a vocab distribution
        *as if decoding from that layer*. Returns (probs_tensor, tokenizer) so callers
        can sum probability mass over an emotion-word token set.
        """
        import torch

        self._ensure_loaded()
        ids = self._tokenizer(text, return_tensors="pt").input_ids.to(self._model.device)
        with torch.no_grad():
            out = self._model(ids, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple: (n_layers + 1) x [1, seq, d]
        mid = layer if layer is not None else len(hidden_states) // 2
        h = hidden_states[mid][0, -1]                      # last-position hidden state
        lm_head = self._model.get_output_embeddings()
        # apply final norm if the architecture exposes one (logit-lens convention)
        norm = getattr(getattr(self._model, "model", self._model), "norm", None)
        if norm is not None:
            h = norm(h)
        logits = lm_head(h)
        probs = torch.softmax(logits, dim=-1)
        return probs, self._tokenizer

    def token_logprobs(self, text: str):
        """Per-token log-probabilities for `text` under the model.

        Used by the internal-emotion logit probe sketched in analysis (Appendix I).
        Returns a list of (token_str, logprob) for each token after the first.
        """
        import torch

        self._ensure_loaded()
        ids = self._tokenizer(text, return_tensors="pt").input_ids.to(self._model.device)
        with torch.no_grad():
            logits = self._model(ids).logits
        logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
        chosen = ids[0, 1:]
        out = []
        for i, tok in enumerate(chosen):
            out.append((self._tokenizer.decode(tok), float(logprobs[i, tok])))
        return out


def _apply_stops(text: str, stops: tuple[str, ...]) -> str:
    for s in stops:
        idx = text.find(s)
        if idx != -1:
            text = text[:idx]
    return text
