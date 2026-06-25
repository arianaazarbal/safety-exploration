"""Local HuggingFace client for Gemma (instruct and base).

Supports assistant-turn *prefilling*, which Section 3 needs: the assistant
message is seeded with fixed text and the model continues from it. We implement
this by building the prompt token ids up to (and including) the prefill, then
decoding only the newly generated tokens.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from .. import config
from .base import Message

# Imports of torch/transformers are deferred to __init__ so importing this
# module (e.g. for type hints) is cheap and does not hard-require a GPU stack.


# A minimal plain-text rendering for *base* models, which have no chat template.
# Section 3 only ever uses these with a prefill, so the exact template matters
# little; we mirror a generic instruct layout to keep base/instruct comparable.
_BASE_TEMPLATE_TURN = "{role}: {content}\n"
_BASE_ASSISTANT_TAG = "assistant: "


class HFChatModel:
    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        is_base: bool = False,
        adapter_path: Optional[str] = None,
        runtime: Optional[config.RuntimeConfig] = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.hf_id = hf_id
        self.is_base = is_base
        self.adapter_path = adapter_path
        self.runtime = runtime or config.RUNTIME

        dtype = getattr(torch, self.runtime.dtype)
        load_kwargs = dict(torch_dtype=dtype, device_map=self.runtime.device_map)
        if self.runtime.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )

        token = config.get_key(config.HF_TOKEN)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, token=token, **load_kwargs)
        if adapter_path:
            # Load a finetuned LoRA adapter (DPO/SFT) on top of the base weights.
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _render_prompt_ids(self, messages: List[Message], prefill: Optional[str]):
        """Return input_ids (1, T) for the prompt, ending right where the model
        should start generating (after any prefill)."""
        import torch

        if self.is_base:
            text = "".join(
                _BASE_TEMPLATE_TURN.format(role=m["role"], content=m["content"])
                for m in messages
            )
            text += _BASE_ASSISTANT_TAG
            if prefill:
                text += prefill
            ids = self.tokenizer(text, return_tensors="pt").input_ids
            return ids.to(self.model.device)

        # Instruct: use the model's chat template.
        ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        if prefill:
            # Append the prefill WITHOUT special tokens so it is part of the
            # assistant turn the model continues.
            prefill_ids = self.tokenizer(
                prefill, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.model.device)
            ids = torch.cat([ids, prefill_ids], dim=-1)
        return ids

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = config.TEMPERATURE,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        prefill: Optional[str] = None,
    ) -> str:
        return self.generate_batch(
            [messages],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            prefill=prefill,
        )[0]

    def generate_batch(
        self,
        batch: Iterable[List[Message]],
        *,
        temperature: float = config.TEMPERATURE,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        prefill: Optional[str] = None,
    ) -> List[str]:
        import torch

        batch = list(batch)
        outputs: List[str] = []
        # We tokenize per-sample (variable-length prompts) and left-pad into a
        # batch. For simplicity and determinism we loop; callers that want true
        # batched throughput can raise the inner batch size here.
        self.tokenizer.padding_side = "left"
        for messages in batch:
            ids = self._render_prompt_ids(messages, prefill)
            with torch.no_grad():
                gen = self.model.generate(
                    ids,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=1.0,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_tokens = gen[0, ids.shape[-1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(text.strip())
        return outputs

    # ------------------------------------------------------------------
    # Helpers used by Section 3 (token-level truncation).
    # ------------------------------------------------------------------
    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        """Return the prefix of ``text`` containing the first ``n_tokens`` tokens
        under this model's tokenizer (used for the 'early' truncation)."""
        ids = self.tokenizer(text, add_special_tokens=False).input_ids[:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)
