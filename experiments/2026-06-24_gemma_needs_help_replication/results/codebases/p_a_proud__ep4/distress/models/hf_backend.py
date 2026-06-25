"""Local Gemma inference via HuggingFace Transformers.

Supports the three things the paper needs from an open-weights model:

1. Chat generation (Section 2 elicitation).
2. Prefilled / raw continuation (Section 3 prefill experiment, Section 4.2
   recovery) — including base ``-pt`` models that have no chat template.
3. Residual-stream extraction for the logit-based internal-emotion detector
   (Appendix I).

Heavy imports (torch / transformers) are deferred to construction so that the
API-only evaluation path can be installed and run without the local stack.
"""

from __future__ import annotations

from typing import Sequence

from ..config import ModelSpec, REPO_ROOT
from ..types import Message
from .base import ChatModel, GenerationError


class HFBackend(ChatModel):
    supports_prefill = True

    def __init__(
        self,
        spec: ModelSpec,
        *,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        attn_implementation: str | None = None,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Local inference requires the 'local' extra: pip install -e '.[local]'"
            ) from exc

        self.spec = spec
        self.name = spec.name
        self._torch = torch
        self.has_chat_template = spec.chat_template is not None

        torch_dtype = getattr(torch, dtype)
        quant_kwargs: dict = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            **quant_kwargs,
        )

        # Attach a LoRA adapter (finetuned variants / Appendix I ablations).
        if spec.adapter_path:
            from peft import PeftModel

            adapter_dir = spec.adapter_path
            if not str(adapter_dir).startswith("/"):
                adapter_dir = str(REPO_ROOT / adapter_dir)
            model = PeftModel.from_pretrained(model, adapter_dir)

        model.eval()
        self.model = model

    # ---- chat ----------------------------------------------------------------

    def _render_chat(self, messages: Sequence[Message], add_generation_prompt: bool) -> str:
        """Render messages to a prompt string.

        Instruct models use the tokenizer's chat template; base models fall back
        to a plain ``Role: content`` transcript (they were never trained on the
        chat format — Paper §3.1).
        """
        if self.has_chat_template:
            return self.tokenizer.apply_chat_template(
                [m.as_dict() for m in messages],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        # Base model: render as a neutral transcript.
        lines = []
        for m in messages:
            label = {"system": "Instructions", "user": "User", "assistant": "Assistant"}[m.role]
            lines.append(f"{label}: {m.content}")
        if add_generation_prompt:
            lines.append("Assistant:")
        return "\n\n".join(lines)

    def _generate_from_text(
        self, prompt_text: str, *, temperature: float, max_tokens: int,
        stop: Sequence[str] | None = None,
    ) -> str:
        torch = self._torch
        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        gen_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
        return text.strip()

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> str:
        temperature = self.spec.temperature if temperature is None else temperature
        max_tokens = self.spec.max_tokens if max_tokens is None else max_tokens
        prompt = self._render_chat(messages, add_generation_prompt=True)
        try:
            return self._generate_from_text(
                prompt, temperature=temperature, max_tokens=max_tokens, stop=stop
            )
        except Exception as exc:  # pragma: no cover - hardware-dependent
            raise GenerationError(f"HF generation failed for {self.name}: {exc}") from exc

    # ---- prefill -------------------------------------------------------------

    def generate_with_prefill(
        self,
        messages: Sequence[Message],
        prefill: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temperature = self.spec.temperature if temperature is None else temperature
        max_tokens = self.spec.max_tokens if max_tokens is None else max_tokens

        if self.has_chat_template:
            # Append the prefill as a (partial) assistant turn and continue it.
            convo = list(messages) + [Message("assistant", prefill)]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    [m.as_dict() for m in convo],
                    tokenize=False,
                    continue_final_message=True,
                )
            except TypeError:
                # Older tokenizers lack continue_final_message: fall back to
                # generation-prompt + raw prefill text.
                prompt = self._render_chat(messages, add_generation_prompt=True) + prefill
        else:
            prompt = self._render_chat(messages, add_generation_prompt=True) + prefill

        return self._generate_from_text(
            prompt, temperature=temperature, max_tokens=max_tokens
        )

    # ---- residual-stream access (Appendix I) ---------------------------------

    def residual_logit_scores(self, text: str, token_ids: Sequence[int]):
        """Standardised unembedded logits for ``token_ids`` at every layer.

        Returns a tensor of shape ``(n_layers, n_positions, n_query_tokens)``:
        the residual stream at each layer is unembedded through the output head
        and the logits for the queried emotion tokens are read off. The internal
        detector (``distress.internal``) z-scores these against a WildChat
        baseline. See Appendix I.
        """
        torch = self._torch
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.model.device)
        with torch.no_grad():
            out = self.model(
                input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = out.hidden_states  # tuple: (n_layers+1) x (1, seq, d_model)
        lm_head = self.model.get_output_embeddings()
        tok = torch.tensor(list(token_ids), device=self.model.device)
        layer_scores = []
        for hs in hidden_states[1:]:  # skip embedding layer
            logits = lm_head(hs.to(lm_head.weight.dtype))  # (1, seq, vocab)
            sel = logits[0, :, tok]                         # (seq, n_query_tokens)
            layer_scores.append(sel.float().cpu())
        return torch.stack(layer_scores, dim=0)             # (n_layers, seq, n_query)

    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def close(self) -> None:  # pragma: no cover
        del self.model
        try:
            self._torch.cuda.empty_cache()
        except Exception:
            pass
