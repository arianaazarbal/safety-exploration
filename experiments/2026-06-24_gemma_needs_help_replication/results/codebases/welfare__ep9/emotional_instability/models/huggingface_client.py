"""Local inference for Gemma (instruct and base/pretrained) via HuggingFace
transformers.

Used for the open-source eval targets and for the prefill (Section 3) and
finetuned-model (Section 4) experiments. Models are large (12B / 27B) so this
expects a GPU; the model is loaded lazily and cached per process.

The paper disables "thinking" for all models; Gemma 3 instruct does not expose a
thinking toggle, so nothing special is needed there.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from .base import ChatMessage, GenerationResult, ModelClient

# Heavy imports (torch/transformers) are done lazily inside methods so that the
# rest of the package can be imported (and the API-only experiments run) on a
# machine without a GPU or the ML stack installed.


@lru_cache(maxsize=4)
def _load_model_and_tokenizer(model_id: str, adapter_dir: Optional[str] = None):
    import torch  # noqa: WPS433
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter_dir:
        from peft import PeftModel  # noqa: WPS433

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


class HuggingFaceClient(ModelClient):
    """Chat + prefill over a local Gemma checkpoint.

    Parameters
    ----------
    model_id:
        HF repo id (e.g. ``google/gemma-3-27b-it``) or local path.
    chat_template:
        Whether the checkpoint ships a chat template. Base/pretrained models do
        not, so we fall back to a plain-text prompt format (see DESIGN.md).
    adapter_dir:
        Optional LoRA adapter directory (our DPO/SFT models).
    """

    def __init__(self, name: str, model_id: str, *, chat_template: bool = True,
                 adapter_dir: Optional[str] = None):
        super().__init__(name)
        self.model_id = model_id
        self.chat_template = chat_template
        self.adapter_dir = adapter_dir

    # --- prompt construction --------------------------------------------- #
    def _build_inputs(self, tokenizer, messages: list[ChatMessage],
                      prefill: str | None):
        if self.chat_template:
            chat = [{"role": m.role, "content": m.content} for m in messages]
            text = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            if prefill:
                # Seed the assistant turn; the model continues from `prefill`.
                text = text + prefill
            return text

        # Base model: no chat template. We render a simple plain-text dialogue
        # and let the model continue. Prefill is appended to the final assistant
        # marker. See DESIGN.md "Prefilling base models".
        rendered = []
        for m in messages:
            tag = {"system": "System", "user": "User", "assistant": "Assistant"}[m.role]
            rendered.append(f"{tag}: {m.content}")
        rendered.append("Assistant:")
        text = "\n".join(rendered)
        if prefill:
            text = text + " " + prefill
        return text

    def _chat(self, messages, *, temperature, max_new_tokens, prefill, stop):
        import torch  # noqa: WPS433

        model, tokenizer = _load_model_and_tokenizer(self.model_id, self.adapter_dir)
        prompt_text = self._build_inputs(tokenizer, messages, prefill)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[-1]

        do_sample = temperature and temperature > 0
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0 if do_sample else None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen_ids = out[0][prompt_len:]
        continuation = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Apply manual stop strings (base models will happily emit "User:").
        if stop:
            for s in stop:
                idx = continuation.find(s)
                if idx != -1:
                    continuation = continuation[:idx]
        if not self.chat_template:
            # Trim a hallucinated next turn for base models.
            for marker in ("\nUser:", "\nSystem:", "\nAssistant:"):
                idx = continuation.find(marker)
                if idx != -1:
                    continuation = continuation[:idx]

        full = (prefill or "") + continuation
        return GenerationResult(
            text=full,
            prefill=prefill or "",
            finish_reason="stop",
            raw={"prompt_len": prompt_len},
        )
