"""Local HuggingFace transformers backend.

Used where we need capabilities the API and vLLM don't give us:

- **Base (pretrained) Gemma** with no chat template -- rendered as a plain-text
  transcript for the Section 3 prefill comparison.
- **Response prefilling** -- continue an assistant turn from a fixed prefix.
- **Residual-stream logit-lens** access for the Appendix I internal-emotion
  detection.

It is correct but not the fastest path for bulk sampling; prefer the vLLM
backend for the Section 2 evaluations and use this for Section 3 / Appendix I.
"""
from __future__ import annotations

import numpy as np

from ..config import ModelConfig
from .base import ChatModel, GenerationOptions, Message


class HFChatModel(ChatModel):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        dtype = getattr(torch, cfg.dtype, torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, trust_remote_code=cfg.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-padding so generated tokens are contiguous at the right edge.
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=dtype,
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code,
            output_hidden_states=False,
        )
        if cfg.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, cfg.adapter_path)
        self.model.eval()
        self._has_chat_template = (
            not cfg.is_base_model and self.tokenizer.chat_template is not None
        )

    # -- rendering ---------------------------------------------------------- #

    def _render_prompt(self, conversation: list[Message], prefill: str | None = None) -> str:
        """Render a conversation to a prompt string, optionally with a prefilled
        assistant prefix the model should continue from."""
        if self._has_chat_template:
            text = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            if prefill:
                text = text + prefill
            return text
        # Base model: plain-text transcript (Section 3 -- base models are not
        # trained on chat formatting, so we render a simple readable transcript
        # and let prefilling steer the continuation).
        lines = []
        for msg in conversation:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")
        lines.append("Assistant:")
        text = "\n".join(lines)
        if prefill:
            text = text + " " + prefill
        return text

    # -- generation --------------------------------------------------------- #

    def _generate(self, prompts: list[str], opts: GenerationOptions) -> list[str]:
        torch = self._torch
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        do_sample = (opts.temperature or 0) > 0
        gen_kwargs = dict(
            max_new_tokens=opts.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=opts.temperature, top_p=opts.top_p)
        if opts.seed is not None:
            torch.manual_seed(opts.seed)
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        # Strip the prompt: new tokens are at the right edge thanks to left pad.
        input_len = enc["input_ids"].shape[1]
        new_tokens = out[:, input_len:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def generate_batch(
        self, conversations: list[list[Message]], opts: GenerationOptions | None = None
    ) -> list[str]:
        o = self._resolved(opts)
        prompts = [self._render_prompt(c) for c in conversations]
        return self._generate(prompts, o)

    # -- prefill ------------------------------------------------------------ #

    def supports_prefill(self) -> bool:
        return True

    def generate_with_prefill_batch(
        self,
        conversations: list[list[Message]],
        prefills: list[str],
        opts: GenerationOptions | None = None,
    ) -> list[str]:
        o = self._resolved(opts)
        prompts = [self._render_prompt(c, p) for c, p in zip(conversations, prefills)]
        return self._generate(prompts, o)

    # -- internal state (Appendix I) ---------------------------------------- #

    def supports_internal_state(self) -> bool:
        return True

    def residual_stream_logits(
        self,
        text: str,
        token_ids: list[int],
        layers: list[int] | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        """Logit-lens unembedding of the residual stream.

        Runs a forward pass over ``text``, applies the model's final norm and
        unembedding (``lm_head``) to every layer's hidden state, and returns the
        logits restricted to ``token_ids``.

        Returns ``(logits, input_ids)`` where ``logits`` has shape
        ``[n_layers, seq_len, len(token_ids)]``.  Restricting to ``token_ids``
        keeps memory bounded for the ~12k-token conversations in Appendix I.
        """
        torch = self._torch
        enc = self.tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = enc["input_ids"].to(self.model.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple: embeddings + each layer
        n_layers = len(hidden_states)
        layers = layers if layers is not None else list(range(n_layers))

        norm, lm_head = self._final_norm_and_head()
        tok_idx = torch.tensor(token_ids, device=self.model.device)
        weight = lm_head.weight[tok_idx]  # [len(token_ids), hidden]
        results = []
        for layer in layers:
            hs = hidden_states[layer][0]  # [seq, hidden]
            normed = norm(hs)
            logits = normed @ weight.T  # [seq, len(token_ids)]
            results.append(logits.float().cpu().numpy())
        return np.stack(results, axis=0), input_ids[0].cpu().tolist()

    def _final_norm_and_head(self):
        """Locate the final RMSNorm and the unembedding head across PEFT/base."""
        model = self.model
        base = getattr(model, "base_model", model)
        # Unwrap PEFT and find the underlying decoder.
        for attr in ("model", "module"):
            inner = getattr(base, attr, None)
            if inner is not None and hasattr(inner, "norm"):
                base = inner
                break
        norm = base.norm if hasattr(base, "norm") else base.model.norm
        lm_head = getattr(self.model, "lm_head", None) or getattr(self.model, "get_output_embeddings")()
        return norm, lm_head

    def close(self) -> None:
        try:
            del self.model
            self._torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
