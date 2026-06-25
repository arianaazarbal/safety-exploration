"""Local Gemma inference via HuggingFace `transformers`.

Handles three things the paper relies on:

1. Instruct models (`*-it`) driven through Gemma's chat template.
2. Base models (`*-pt`) driven purely by prefill continuation (Section 3). We
   render the *same* chat-template prefix for base and instruct models so they
   continue "from the same starting points" (Section 3.1); see DESIGN.md for
   why we template the base model rather than using a bespoke plain-text format.
3. Assistant prefill: force the reply to start with a fixed string and return
   only the continuation, used in Sections 3 and 4.2.

A vLLM fast path is used automatically when vLLM is installed and the model is
not being loaded with a LoRA adapter; otherwise we fall back to `transformers`
`.generate`. Sampling throughput matters here — the main evaluation samples
thousands of multi-turn rollouts per model.

This backend also exposes `.model` / `.tokenizer` so Appendix I's logit-lens
emotion detection can read hidden states directly.
"""

from __future__ import annotations

from typing import Optional

from ..config import MAX_NEW_TOKENS, ModelSpec
from .base import ChatMessage, GenerationResult, ModelBackend, SamplingParams


def _fold_system(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Fold a leading system message into the first user turn (Gemma has no
    system role). Non-leading system messages are unexpected and left as-is.
    """
    if not messages or messages[0].role != "system":
        return messages
    system = messages[0].content
    rest = messages[1:]
    for i, m in enumerate(rest):
        if m.role == "user":
            merged = ChatMessage("user", f"{system}\n\n{m.content}")
            return rest[:i] + [merged] + rest[i + 1:]
    # No user turn (degenerate): emit system as a user preamble.
    return [ChatMessage("user", system)] + rest


class HuggingFaceBackend(ModelBackend):
    supports_prefill = True

    def __init__(
        self,
        spec: ModelSpec,
        adapter_path: Optional[str] = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        use_vllm: Optional[bool] = None,
    ):
        self.spec = spec
        self.is_base = spec.is_base
        self.adapter_path = adapter_path
        self._dtype = dtype
        self._device_map = device_map
        self._model = None
        self._tokenizer = None
        # vLLM cannot hot-swap arbitrary LoRA adapters as conveniently and does
        # not expose hidden states, so disable it whenever an adapter is set or
        # internal probing is needed.
        self._use_vllm = (adapter_path is None) if use_vllm is None else use_vllm
        self._vllm = None

    # ------------------------------------------------------------------ #
    # Lazy loading                                                       #
    # ------------------------------------------------------------------ #
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.spec.model_id)
            # Base (-pt) tokenizers usually ship without a chat template. Section 3
            # requires base and instruct to be rendered identically, so borrow the
            # matching instruct tokenizer's template when one is missing.
            if getattr(tok, "chat_template", None) is None:
                it_id = self.spec.model_id.replace("-pt", "-it")
                if it_id != self.spec.model_id:
                    it_tok = AutoTokenizer.from_pretrained(it_id)
                    tok.chat_template = it_tok.chat_template
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"  # required for batched decoder generation
            self._tokenizer = tok
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM

            self._model = AutoModelForCausalLM.from_pretrained(
                self.spec.model_id,
                torch_dtype=getattr(torch, self._dtype),
                device_map=self._device_map,
            )
            if self.adapter_path:
                from peft import PeftModel

                self._model = PeftModel.from_pretrained(
                    self._model, self.adapter_path
                )
            self._model.eval()
        return self._model

    def _ensure_vllm(self):
        if self._vllm is None:
            from vllm import LLM

            self._vllm = LLM(
                model=self.spec.model_id,
                dtype=self._dtype,
                enable_prefix_caching=True,
            )
        return self._vllm

    # ------------------------------------------------------------------ #
    # Prompt rendering                                                   #
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[ChatMessage], prefill: str = "") -> str:
        """Render a conversation to a single prompt string.

        We always apply Gemma's chat template (including for base models) with a
        trailing generation prompt, then append `prefill` so generation
        continues from it. Identical rendering for base and instruct keeps the
        Section 3 comparison fair.

        Gemma's chat template has no dedicated system role, so a leading system
        message is folded into the first user turn (the conventional Gemma
        usage) rather than passed through as `role="system"`.
        """
        msgs = [m.as_dict() for m in _fold_system(messages)]
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        return text + prefill

    # ------------------------------------------------------------------ #
    # Generation                                                         #
    # ------------------------------------------------------------------ #
    def _sampling_kwargs(self, params: SamplingParams) -> dict:
        return dict(
            temperature=params.temperature,
            top_p=params.top_p,
            max_new_tokens=params.max_new_tokens or MAX_NEW_TOKENS,
        )

    def generate(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> GenerationResult:
        return self.generate_batch([messages], params)[0]

    def generate_with_prefill(
        self, messages: list[ChatMessage], prefill: str, params: SamplingParams
    ) -> GenerationResult:
        return self.generate_with_prefill_batch([(messages, prefill)], params)[0]

    def generate_batch(
        self, batch: list[list[ChatMessage]], params: SamplingParams
    ) -> list[GenerationResult]:
        prompts = [self._render(m) for m in batch]
        texts = self._raw_generate(prompts, params)
        return [GenerationResult(text=t.strip()) for t in texts]

    def generate_with_prefill_batch(
        self, batch: list[tuple[list[ChatMessage], str]], params: SamplingParams
    ) -> list[GenerationResult]:
        prompts = [self._render(m, prefill=p) for (m, p) in batch]
        texts = self._raw_generate(prompts, params)
        return [
            GenerationResult(text=t.strip(), prefill=p)
            for t, (_, p) in zip(texts, batch)
        ]

    # ------------------------------------------------------------------ #
    # Low-level: prompt strings -> completion strings                    #
    # ------------------------------------------------------------------ #
    def _raw_generate(self, prompts: list[str], params: SamplingParams) -> list[str]:
        if self._use_vllm:
            try:
                return self._raw_generate_vllm(prompts, params)
            except ImportError:
                self._use_vllm = False  # fall through to transformers
        return self._raw_generate_hf(prompts, params)

    def _raw_generate_vllm(self, prompts: list[str], params: SamplingParams) -> list[str]:
        from vllm import SamplingParams as VSP

        sp = VSP(
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_new_tokens or MAX_NEW_TOKENS,
            seed=params.seed,
            stop=params.stop,
        )
        outs = self._ensure_vllm().generate(prompts, sp)
        return [o.outputs[0].text for o in outs]

    def _raw_generate_hf(self, prompts: list[str], params: SamplingParams) -> list[str]:
        import torch

        tok = self.tokenizer
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=params.temperature > 0,
                pad_token_id=tok.pad_token_id,
                **self._sampling_kwargs(params),
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        return tok.batch_decode(gen, skip_special_tokens=True)
