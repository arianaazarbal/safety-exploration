"""Local Gemma backend via HuggingFace transformers.

Supports the three things the API backends cannot:

  * batched sampling at temperature 1 (throughput for the 4,000-response sweep);
  * prefilling an assistant turn and returning only the continuation
    (Section 3.1 base-vs-instruct, Section 4.2 recovery);
  * exposing per-layer residual streams for the logit-lens probe (Appendix I).

Base ("-pt") and instruct ("-it") checkpoints are both supported. For base
models there is no chat template, so we fall back to a plain concatenation with
role markers — this only matters for prefilling experiments, which is the one
place base models are used (Section 3).

A finetuned LoRA adapter (Section 4) can be layered on top via ``adapter_path``.
"""

from __future__ import annotations

from dataclasses import dataclass

from emotional_stability.config import GEMMA_LOCAL_MODELS, Settings
from emotional_stability.models.base import ChatModel, GenerationConfig
from emotional_stability.models.registry import _fold_system
from emotional_stability.records import Message


@dataclass
class GemmaLoadConfig:
    dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: str = "eager"  # required for stable hidden-state hooks
    load_in_4bit: bool = False  # set True to fit 27B on a single 48GB GPU


class GemmaLocalModel(ChatModel):
    supports_prefill = True
    supports_hidden_states = True

    def __init__(
        self,
        name: str,
        *,
        adapter_path: str | None = None,
        load_cfg: GemmaLoadConfig | None = None,
        settings: Settings | None = None,
    ):
        if name not in GEMMA_LOCAL_MODELS and adapter_path is None:
            raise ValueError(f"unknown Gemma model key: {name}")
        self.name = name
        self.hf_id = GEMMA_LOCAL_MODELS.get(name, name)
        self.is_base = name.endswith("-pt")
        self.load_cfg = load_cfg or GemmaLoadConfig()
        self.settings = settings or Settings.load()
        self._adapter_path = adapter_path
        self._model = None
        self._tokenizer = None

    # ----------------------------------------------------------------- load --
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = self.settings.hf_token
        quant_kwargs: dict = {}
        if self.load_cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id, token=token)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            torch_dtype=getattr(torch, self.load_cfg.dtype),
            device_map=self.load_cfg.device_map,
            attn_implementation=self.load_cfg.attn_implementation,
            token=token,
            **quant_kwargs,
        )
        if self._adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self._adapter_path)
        self._model.eval()

    # ------------------------------------------------------------- prompts --
    def _render(self, messages: list[Message], add_generation_prompt: bool) -> str:
        """Render messages to a prompt string.

        Instruct models use the official Gemma chat template. Base models have no
        template; we use a minimal role-tagged concatenation, which is only ever
        used under prefilling where the assistant turn is provided anyway.
        """
        self._ensure_loaded()
        if not self.is_base and self._tokenizer.chat_template:
            # Gemma chat template has no system role; fold any system message
            # into the first user turn (standard Gemma practice).
            msgs = _fold_system(messages)
            return self._tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in msgs],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        # Base model fallback.
        parts = [f"{m.role.upper()}: {m.content}" for m in messages]
        if add_generation_prompt:
            parts.append("ASSISTANT:")
        return "\n\n".join(parts)

    # --------------------------------------------------------------- chat --
    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        return self.chat_batch([messages], cfg)[0]

    def chat_batch(
        self, batch: list[list[Message]], cfg: GenerationConfig
    ) -> list[str]:
        self._ensure_loaded()
        import torch

        prompts = [self._render(m, add_generation_prompt=True) for m in batch]
        return self._generate(prompts, cfg, torch)

    def generate_with_prefill(
        self, messages: list[Message], prefill: str, cfg: GenerationConfig
    ) -> str:
        self._ensure_loaded()
        import torch

        base_prompt = self._render(messages, add_generation_prompt=True)
        # Append the prefill so generation continues from inside the assistant
        # turn. We return only the continuation (text generated after prefill).
        prompt = base_prompt + prefill
        out = self._generate([prompt], cfg, torch)[0]
        return out

    def _generate(self, prompts: list[str], cfg: GenerationConfig, torch) -> list[str]:
        tok = self._tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(self._model.device) for k, v in enc.items()}
        do_sample = cfg.temperature > 0
        with torch.no_grad():
            gen = self._model.generate(
                **enc,
                do_sample=do_sample,
                temperature=cfg.temperature if do_sample else None,
                top_p=cfg.top_p if do_sample else None,
                max_new_tokens=cfg.max_tokens,
                pad_token_id=tok.pad_token_id,
            )
        # Strip the prompt tokens; decode only the continuation.
        input_len = enc["input_ids"].shape[1]
        completions = gen[:, input_len:]
        texts = tok.batch_decode(completions, skip_special_tokens=True)
        return [t.strip() for t in texts]

    # ------------------------------------------------ hidden states (App. I) --
    def residual_stream_logits(self, text: str):
        """Return (hidden_states, lm_head) for the logit-lens probe.

        ``hidden_states`` is a tuple of per-layer tensors [seq, d_model] for a
        single forward pass over ``text`` (the full rendered conversation). The
        probe (internal/emotion_logits.py) unembeds each layer with ``lm_head``
        and aggregates over emotion-token columns.
        """
        self._ensure_loaded()
        import torch

        enc = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self._model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc, output_hidden_states=True)
        lm_head = self._model.get_output_embeddings()
        # input_ids returned so the probe can align tokens to positions.
        return out.hidden_states, lm_head, enc["input_ids"][0]

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer
