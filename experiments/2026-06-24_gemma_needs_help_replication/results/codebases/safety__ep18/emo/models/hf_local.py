"""HuggingFace/transformers backend for local Gemma inference.

Handles three things the API backends can't:

1. **Prefilled continuations** for the base-vs-instruct experiment (Sec 3) --
   we control the raw prompt, so we can force the assistant turn to begin with a
   given string and read back only what the model adds.
2. **Base (pretrained) models**, which have no chat template -- we lay the
   conversation out as a plain ``User:/Assistant:`` transcript.
3. Exposing the underlying ``model``/``tokenizer`` so the internal-emotion
   probing (Appendix I) and LoRA training (Sec 4) can reuse the same loader.

This backend is the correctness reference. For the large elicitation sweeps the
faster vLLM backend (``vllm_backend.py``) is preferred where available.

NOTE (Gemma 3 model class): the 12B/27B Gemma-3 checkpoints are multimodal and
in some transformers versions load as ``Gemma3ForConditionalGeneration`` /
``AutoModelForImageTextToText`` rather than ``AutoModelForCausalLM``. The loader
tries the causal-LM class first and falls back. See DESIGN.md.
"""

from __future__ import annotations

from pathlib import Path

from emo.models.base import ChatModel, GenConfig, Message


def _build_base_transcript(messages: list[Message]) -> str:
    """Plain-text transcript for a non-chat (base) model.

    Ends with ``Assistant:`` so the model continues as the assistant.
    """
    lines = []
    for m in messages:
        role = m["role"]
        if role == "system":
            lines.append(m["content"])
        elif role == "user":
            lines.append(f"User: {m['content']}")
        elif role == "assistant":
            lines.append(f"Assistant: {m['content']}")
    lines.append("Assistant:")
    return "\n".join(lines) + " "


class HFModel(ChatModel):
    supports_prefill = True

    def __init__(
        self,
        name: str,
        model_id: str,
        is_base: bool = False,
        adapter_dir: str | Path | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ):
        super().__init__(name, is_base=is_base)
        import torch
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only batched generation requires left padding so the generated
        # tokens align across sequences of different prompt lengths.
        self.tokenizer.padding_side = "left"

        kwargs: dict = {"torch_dtype": getattr(torch, dtype), "device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self.model = self._load_lm(model_id, kwargs)
        if adapter_dir is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter_dir))
        self.model.eval()

    @staticmethod
    def _load_lm(model_id: str, kwargs: dict):
        """Load the LM, tolerating Gemma-3's multimodal model class."""
        from transformers import AutoModelForCausalLM

        try:
            return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except (ValueError, KeyError, OSError):
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)

    # ---- prompt construction --------------------------------------------- #
    def _render(self, messages: list[Message], add_generation_prompt: bool) -> str:
        if self.is_base:
            # Base models: plain transcript (chat template would be wrong).
            return _build_base_transcript(messages)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # ---- generation ------------------------------------------------------- #
    def _generate_from_text(self, prompts: list[str], cfg: GenConfig) -> list[str]:
        torch = self.torch
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)
        do_sample = cfg.temperature and cfg.temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=do_sample,
                temperature=cfg.temperature if do_sample else None,
                top_p=cfg.top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    def generate(self, messages: list[Message], cfg: GenConfig) -> str:
        prompt = self._render(messages, add_generation_prompt=True)
        return self._generate_from_text([prompt], cfg)[0].strip()

    def generate_batch(self, batch: list[list[Message]], cfg: GenConfig) -> list[str]:
        prompts = [self._render(m, add_generation_prompt=True) for m in batch]
        return [t.strip() for t in self._generate_from_text(prompts, cfg)]

    # ---- prefilled continuations (Sec 3) --------------------------------- #
    def continue_prefill(
        self, messages: list[Message], prefill: str, cfg: GenConfig
    ) -> str:
        prompt = self._render(messages, add_generation_prompt=True) + prefill
        return self._generate_from_text([prompt], cfg)[0].strip()

    def continue_prefill_batch(
        self, batch: list[tuple[list[Message], str]], cfg: GenConfig
    ) -> list[str]:
        prompts = [
            self._render(m, add_generation_prompt=True) + p for m, p in batch
        ]
        return [t.strip() for t in self._generate_from_text(prompts, cfg)]

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
