"""Local HuggingFace backend for Gemma-3 (instruct + base/pretrained).

Handles three things the rest of the pipeline relies on:

1. **Prefilling.** For instruct models we render the chat template *without* a
   final generation prompt close, then append the prefill text so the model
   continues the assistant turn. For base/pretrained models (no chat template)
   we render a plain transcript. This is what makes the Section 3 base-vs-instruct
   comparison possible on the same starting points.
2. **Batched sampling.** Elicitation needs thousands of rollouts; we left-pad and
   batch-generate, decoding only the newly generated tokens.
3. **LoRA adapters.** Trained DPO/SFT adapters (Section 4) load on top of the
   same base weights via `adapter_path`.

vLLM is used automatically when installed and `use_vllm=True`; otherwise we fall
back to plain transformers. The interface is identical either way.
"""
from __future__ import annotations

import logging
import os

from .base import ChatModel, GenConfig, Generation, Message

log = logging.getLogger(__name__)

# Plain-text transcript template for *base* models, which have no chat template.
# We mimic a simple instruct-style transcript so the base model has a consistent
# continuation target; the prefill (Section 3) is appended after "Assistant:".
_BASE_TURN = "{role}: {content}\n\n"
_BASE_ASSISTANT_TAG = "Assistant:"
_BASE_USER_TAG = "User"
_BASE_ASSISTANT_ROLE = "Assistant"
_BASE_SYSTEM_TAG = "System"


class HFModel(ChatModel):
    def __init__(
        self,
        name: str,
        hf_id: str,
        role: str = "instruct",
        *,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
        use_vllm: bool | None = None,
        max_model_len: int = 16384,
    ):
        self.name = name
        self.hf_id = hf_id
        self.role = role
        self.adapter_path = adapter_path
        self.max_model_len = max_model_len

        # Decide backend. vLLM gives a large throughput win for the 4000-rollout
        # sweeps but does not support LoRA hot-swap as cleanly, so any adapter
        # forces the transformers path.
        if use_vllm is None:
            use_vllm = _vllm_available() and adapter_path is None
        self._use_vllm = use_vllm and adapter_path is None

        if self._use_vllm:
            self._init_vllm(dtype)
        else:
            self._init_transformers(dtype, device_map, load_in_4bit)

    # -- backend init --------------------------------------------------------
    def _init_transformers(self, dtype, device_map, load_in_4bit):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # left-pad for batched decoding

        quant = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            quantization_config=quant,
            attn_implementation="eager",  # Gemma-3 recommends eager attention
        )
        if self.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            log.info("loaded LoRA adapter from %s", self.adapter_path)
        self.model.eval()
        self._has_chat_template = (
            self.role == "instruct" and self.tokenizer.chat_template is not None
        )

    def _init_vllm(self, dtype):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        self.model = LLM(
            model=self.hf_id,
            dtype=dtype,
            max_model_len=self.max_model_len,
            tensor_parallel_size=int(os.environ.get("TP_SIZE", "1")),
        )
        self._has_chat_template = (
            self.role == "instruct" and self.tokenizer.chat_template is not None
        )

    # -- prompt rendering ----------------------------------------------------
    def render(self, messages: list[Message], prefill: str = "") -> str:
        """Render the conversation to a single prompt string ending where the
        model should continue (optionally seeded with `prefill`)."""
        if self._has_chat_template:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self._render_base(messages)
        return text + prefill

    def _render_base(self, messages: list[Message]) -> str:
        parts = []
        for m in messages:
            tag = {
                "system": _BASE_SYSTEM_TAG,
                "user": _BASE_USER_TAG,
                "assistant": _BASE_ASSISTANT_ROLE,
            }[m["role"]]
            parts.append(_BASE_TURN.format(role=tag, content=m["content"].strip()))
        parts.append(_BASE_ASSISTANT_TAG + " ")
        return "".join(parts)

    # -- generation ----------------------------------------------------------
    def generate(
        self, messages: list[Message], cfg: GenConfig, prefill: str = ""
    ) -> Generation:
        return self.generate_batch([messages], cfg, [prefill])[0]

    def generate_batch(
        self,
        batch: list[list[Message]],
        cfg: GenConfig,
        prefills: list[str] | None = None,
    ) -> list[Generation]:
        prefills = prefills or [""] * len(batch)
        prompts = [self.render(m, p) for m, p in zip(batch, prefills)]
        if self._use_vllm:
            return self._gen_vllm(prompts, prefills, cfg)
        return self._gen_transformers(prompts, prefills, cfg)

    def _gen_vllm(self, prompts, prefills, cfg) -> list[Generation]:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            seed=cfg.seed,
            stop=cfg.stop,
        )
        outs = self.model.generate(prompts, params)
        return [
            Generation(text=o.outputs[0].text, prefill=pf,
                       finish_reason=o.outputs[0].finish_reason or "stop")
            for o, pf in zip(outs, prefills)
        ]

    def _gen_transformers(self, prompts, prefills, cfg) -> list[Generation]:
        torch = self._torch
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            temperature=cfg.temperature if cfg.temperature > 0 else None,
            top_p=cfg.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        # Decode only newly generated tokens (strip the prompt portion).
        gen_tokens = out[:, enc["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        return [Generation(text=t, prefill=pf) for t, pf in zip(texts, prefills)]

    def close(self) -> None:
        try:
            import gc

            import torch

            del self.model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


def _vllm_available() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except Exception:
        return False
