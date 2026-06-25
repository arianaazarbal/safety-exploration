"""Local HuggingFace backend for Gemma (instruct, base, and LoRA finetunes).

Supports:
  * standard chat generation via the Gemma chat template
  * prefilled assistant turns (continue_final_message) for the prefill study
  * a plain-text transcript format for *base* (pt) models, which have no chat
    template -- the paper prefills base models so they "consistently continue
    the model response"
  * loading LoRA adapters (our DPO / SFT finetunes)
  * hidden-state / logit access for the internal-emotion probing (Appendix I)
"""

from __future__ import annotations

import torch

from .base import Message, ModelClient


# Gemma-3 chat turn markers (used to build the base-model transcript format).
_GEMMA_USER = "<start_of_turn>user\n{content}<end_of_turn>\n"
_GEMMA_MODEL_OPEN = "<start_of_turn>model\n"


class HFModel(ModelClient):
    def __init__(self, name: str, model_id: str, *, kind: str = "instruct",
                 adapter_path: str | None = None, load_in_4bit: bool = False,
                 dtype: str = "bfloat16", device_map: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.model_id = model_id
        self.kind = kind  # "instruct" | "base" | "finetune"
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
            self.model = self.model.merge_and_unload()  # fold LoRA in for fast inference

        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _build_prompt(self, messages: list[Message], prefill: str | None) -> str:
        """Return the full prompt string the model should continue from."""
        if self.kind == "base":
            # Base/pretrained models have no chat template. We use a light
            # transcript format mirroring Gemma's turn markers so the format is
            # consistent across base/instruct, then open a model turn.
            parts = []
            for m in messages:
                if m["role"] == "user":
                    parts.append(_GEMMA_USER.format(content=m["content"]))
                elif m["role"] == "assistant":
                    parts.append(f"{_GEMMA_MODEL_OPEN}{m['content']}<end_of_turn>\n")
                # system messages are folded into the first user turn upstream
            prompt = "".join(parts) + _GEMMA_MODEL_OPEN
            if prefill:
                prompt += prefill
            return prompt

        # Instruct / finetune: use the tokenizer chat template.
        if prefill is not None:
            # Append an assistant message containing the prefill and ask the
            # template to *continue* it rather than close it.
            msgs = list(messages) + [{"role": "assistant", "content": prefill}]
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, continue_final_message=True)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, messages, *, max_new_tokens=2048, temperature=1.0,
                 prefill=None) -> str:
        return self.generate_batch(
            [messages], max_new_tokens=max_new_tokens,
            temperature=temperature, prefills=[prefill])[0]

    @torch.no_grad()
    def generate_batch(self, batch_messages, *, max_new_tokens=2048,
                       temperature=1.0, prefills=None) -> list[str]:
        if prefills is None:
            prefills = [None] * len(batch_messages)
        prompts = [self._build_prompt(m, p) for m, p in zip(batch_messages, prefills)]

        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                             add_special_tokens=False).to(self.model.device)

        do_sample = temperature and temperature > 0
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=1.0 if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return [t.strip() for t in texts]

    @property
    def supports_prefill(self) -> bool:
        return True

    @property
    def supports_internals(self) -> bool:
        return True

    # ------------------------------------------------------------------ #
    # Internals access for probing (Appendix I)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def residual_stream(self, text: str) -> tuple[torch.Tensor, list[int]]:
        """Return per-layer hidden states for `text`.

        Returns (hidden, token_ids) where hidden has shape
        [n_layers+1, seq_len, d_model] (layer 0 = embeddings).
        """
        enc = self.tokenizer(text, return_tensors="pt",
                             add_special_tokens=False).to(self.model.device)
        out = self.model(**enc, output_hidden_states=True)
        hidden = torch.stack(out.hidden_states, dim=0).squeeze(1)  # [L+1, T, D]
        return hidden, enc["input_ids"][0].tolist()

    @torch.no_grad()
    def unembed(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the final norm + LM head to a [.., d_model] residual tensor,
        returning logits over the vocabulary."""
        model = self.model
        # Gemma applies a final RMSNorm before the LM head.
        norm = getattr(model.model, "norm", None)
        h = norm(hidden) if norm is not None else hidden
        return model.lm_head(h)

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()
