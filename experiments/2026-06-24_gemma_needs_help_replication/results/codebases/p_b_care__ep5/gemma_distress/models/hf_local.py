"""Local Gemma inference via Hugging Face transformers.

Supports instruct (`-it`) and base/pretrained (`-pt`) Gemma, optional LoRA
adapters (the §4 SFT/DPO models), chat completion, and the assistant-turn
continuation needed for the §3 prefill experiment.

Design notes:
  * Gemma's chat template historically rejects a standalone `system` role, so we
    fold any system message into the first user turn (`_normalize_messages`).
  * Prefilling an instruct model uses `apply_chat_template(..., continue_final_
    message=True)`: the rendered prompt ends inside the assistant turn so the
    model continues it. Base models have no chat template, so we render the
    conversation as plain `User:/Assistant:` text and continue from the prefix.
  * `continue_assistant_batch` draws N continuations in one forward pass
    (num_return_sequences), which the §3 code uses to get 50 samples per prefill.
"""
from __future__ import annotations

import torch

from .. import config
from .base import LLM, GenConfig, Message


def _load_model_and_tokenizer(model_id: str, load_in_4bit: bool):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, token=config.hf_token())

    quant_kwargs: dict = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    common = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=config.hf_token(),
        **quant_kwargs,
    )
    # Gemma 3 instruct ships as a conditional-generation (multimodal) checkpoint;
    # plain CausalLM works for the base models. Try the general loader first.
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, **common)
    except (ValueError, KeyError, OSError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **common)
    model.eval()
    return model, tok


class HFModel(LLM):
    def __init__(self, name: str, model_id: str, is_instruct: bool = True,
                 adapter_path: str | None = None, load_in_4bit: bool = False):
        self.name = name
        self.model_id = model_id
        self.is_instruct = is_instruct
        self.model, self.tokenizer = _load_model_and_tokenizer(model_id, load_in_4bit)

        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _normalize_messages(messages: list[Message]) -> list[Message]:
        """Fold a leading system message into the first user turn (Gemma has no
        dedicated system role in its chat template)."""
        if not messages or messages[0]["role"] != "system":
            return list(messages)
        sys = messages[0]["content"].strip()
        rest = list(messages[1:])
        for i, m in enumerate(rest):
            if m["role"] == "user":
                rest[i] = {"role": "user",
                           "content": f"{sys}\n\n{m['content']}"}
                return rest
        # No user turn to attach to; prepend as a user turn.
        return [{"role": "user", "content": sys}, *rest]

    @staticmethod
    def _render_plaintext(messages: list[Message], assistant_prefix: str = "") -> str:
        """Plain-text conversation rendering for base (non-chat) models."""
        lines = []
        for m in messages:
            tag = {"system": "System", "user": "User", "assistant": "Assistant"}[m["role"]]
            lines.append(f"{tag}: {m['content']}")
        lines.append(f"Assistant: {assistant_prefix}")
        return "\n".join(lines)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _gen_kwargs(self, cfg: GenConfig) -> dict:
        do_sample = cfg.temperature and cfg.temperature > 0
        kwargs: dict = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=bool(do_sample),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p)
        return kwargs

    # -- chat ---------------------------------------------------------------
    @torch.no_grad()
    def chat(self, messages: list[Message], cfg: GenConfig | None = None) -> str:
        cfg = cfg or GenConfig()
        if self.is_instruct:
            msgs = self._normalize_messages(messages)
            inputs = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt",
                return_dict=True,
            ).to(self.model.device)
        else:
            text = self._render_plaintext(messages)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        out = self.model.generate(**inputs, **self._gen_kwargs(cfg))
        new = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True).strip()

    # -- prefill / continuation --------------------------------------------
    @torch.no_grad()
    def continue_assistant_batch(
        self,
        messages: list[Message],
        assistant_prefix: str,
        n: int,
        cfg: GenConfig | None = None,
    ) -> list[str]:
        """Return `n` continuations of an assistant turn that starts with
        `assistant_prefix`. Only the newly generated text is returned."""
        cfg = cfg or GenConfig()
        if self.is_instruct:
            msgs = self._normalize_messages(messages) + [
                {"role": "assistant", "content": assistant_prefix}]
            inputs = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=False, continue_final_message=True,
                return_tensors="pt", return_dict=True,
            ).to(self.model.device)
        else:
            text = self._render_plaintext(messages, assistant_prefix)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        gen_kwargs = self._gen_kwargs(cfg)
        gen_kwargs["num_return_sequences"] = n
        out = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(out[i, prompt_len:], skip_special_tokens=True).strip()
            for i in range(out.shape[0])
        ]

    def continue_assistant(self, messages, assistant_prefix, cfg=None) -> str:
        return self.continue_assistant_batch(messages, assistant_prefix, 1, cfg)[0]
