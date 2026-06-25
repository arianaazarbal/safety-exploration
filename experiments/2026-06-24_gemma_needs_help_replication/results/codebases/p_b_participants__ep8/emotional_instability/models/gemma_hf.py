"""Local HuggingFace backend for Gemma (instruct, base, and LoRA-adapted).

Used for:
  * Section 2 sampling of Gemma-3-{12,27}B-it.
  * Section 3 prefilling of Gemma base vs instruct (base models have no chat
    template, so we build the prompt manually and let them continue).
  * Section 4 evaluation of DPO/SFT LoRA adapters (load adapter on top of the
    instruct base).
  * Appendix I internal-emotion probing (needs residual-stream / logit access).

Loads lazily so that importing the package on a machine without a GPU (or
without torch) does not fail -- only instantiating a Gemma client does.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .base import ChatMessage, Generation


class GemmaHFClient:
    """Wraps a transformers CausalLM + tokenizer for one Gemma checkpoint."""

    def __init__(
        self,
        model_id: str,
        spec_name: str,
        *,
        is_instruct: bool = True,
        adapter_path: Optional[str] = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ) -> None:
        import torch  # local import: torch is optional for API-only runs
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec_name = spec_name
        self.model_id = model_id
        self.is_instruct = is_instruct
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["torch_dtype"] = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # -- prompt construction --------------------------------------------------

    def _render_chat(self, messages: Sequence[ChatMessage], add_generation: bool) -> str:
        """Render messages to a prompt string.

        Instruct models use the Gemma chat template. Base models (no template)
        fall back to a plain transcript -- but in practice base models are only
        ever called through ``continue_prefill`` with an explicit prefill, so
        the format there matches the paper's prefilling protocol.
        """
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        if self.is_instruct and self.tokenizer.chat_template:
            # Gemma chat template has no system role; fold any system message
            # into the first user turn.
            msgs = _fold_system_into_user(msgs)
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_generation
            )
        # Base-model fallback: simple labelled transcript.
        parts = [f"{m['role'].capitalize()}: {m['content']}" for m in msgs]
        if add_generation:
            parts.append("Assistant:")
        return "\n".join(parts)

    # -- generation -----------------------------------------------------------

    def _sample(self, prompt: str, temperature: float, max_new_tokens: int,
                seed: Optional[int]) -> str:
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def generate(self, messages, *, temperature=1.0, max_new_tokens=2048,
                 seed=None) -> Generation:
        prompt = self._render_chat(messages, add_generation=True)
        text = self._sample(prompt, temperature, max_new_tokens, seed)
        return Generation(text=text.strip())

    def continue_prefill(self, messages, prefill, *, temperature=1.0,
                         max_new_tokens=2048, seed=None) -> Generation:
        # Build prompt up to (and including the start of) the assistant turn,
        # then append the prefill so the model continues from it.
        prompt = self._render_chat(messages, add_generation=True) + prefill
        cont = self._sample(prompt, temperature, max_new_tokens, seed)
        return Generation(text=cont)  # continuation only (paper scores this)

    # -- internal probing (Appendix I) ---------------------------------------

    def logits_for_text(self, text: str):
        """Return (token_ids, logits[seq, vocab]) for a full text in one pass.

        Used by ``internal.emotion_logit`` to unembed the residual stream and
        aggregate over Ekman emotion tokens. Returns torch tensors on CPU.
        """
        torch = self._torch
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs)
        return inputs["input_ids"][0].cpu(), out.logits[0].float().cpu()

    def hidden_states_for_text(self, text: str):
        """Return per-layer hidden states [n_layers+1, seq, d_model] (CPU).

        Used for the central-layer logit-lens probing in Appendix I.
        """
        torch = self._torch
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)[:, 0]  # [L+1, seq, d]
        return inputs["input_ids"][0].cpu(), hs.float().cpu()

    def unembed(self, hidden: "object"):
        """Project a residual-stream vector through the (tied) unembedding."""
        torch = self._torch
        with torch.no_grad():
            # Gemma applies a final RMSNorm before the LM head; apply it too.
            base = getattr(self.model, "model", self.model)
            normed = base.norm(hidden.to(self.model.device))
            logits = self.model.get_output_embeddings()(normed)
        return logits.float().cpu()


def _fold_system_into_user(msgs: list[dict]) -> list[dict]:
    """Gemma has no system role; prepend any system text to the first user turn."""
    if not msgs or msgs[0]["role"] != "system":
        return msgs
    system = msgs[0]["content"]
    rest = msgs[1:]
    for i, m in enumerate(rest):
        if m["role"] == "user":
            rest = list(rest)
            rest[i] = {"role": "user", "content": f"{system}\n\n{m['content']}"}
            return rest
    return [{"role": "user", "content": system}] + rest
