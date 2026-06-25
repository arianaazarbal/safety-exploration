"""Local Gemma inference via HuggingFace Transformers (and optional vLLM).

Two backends share one prompt-rendering convention:

  * :class:`TransformersProvider` — full capability set: chat, batched chat,
    assistant **prefill** continuation, and **residual-stream logits** (the logit
    lens used for Appendix I probing). LoRA adapters (Section 4 outputs) load on
    top of the base weights.
  * :class:`VLLMProvider` — fast chat + prefill only, for the high-volume
    Section 2 generation (4000 rollouts/model). No logit access.

Prompt rendering (a gap the paper leaves implicit for base models):
  * **Instruct** models use the tokenizer chat template with a generation prompt.
  * **Base** (``-pt``) models have no chat template, so we render the
    conversation as a plain ``User:/Model:`` transcript and let the model
    continue. For prefill, the truncated assistant text is appended verbatim and
    generation continues from it. See DESIGN.md §Prefill rendering.
"""
from __future__ import annotations

import threading
from typing import Any

from ..logging_utils import get_logger
from .base import ChatMessage, ChatProvider, GenerationResult, RetryableError, with_retry

log = get_logger("providers.local_hf")


def render_base_transcript(messages: list[ChatMessage]) -> str:
    """Plain-text transcript for base (non-chat) models."""
    lines = []
    for m in messages:
        role = m["role"]
        tag = {"user": "User", "assistant": "Model", "system": "System"}.get(role, role)
        lines.append(f"{tag}: {m['content']}")
    lines.append("Model: ")  # generation prompt
    return "\n".join(lines)


class TransformersProvider(ChatProvider):
    """Transformers-backed Gemma. Lazy-loads weights on first use."""

    capabilities = {"chat", "prefill", "logits"}
    prefers_batch = True  # runner uses lockstep batched rollouts

    def __init__(
        self,
        model: str,
        model_id: str,
        *,
        retry_cfg: dict | None = None,
        usage=None,
        is_instruct: bool = True,
        adapter: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ):
        super().__init__(model, model_id, retry_cfg, usage)
        self.is_instruct = is_instruct
        self.adapter = adapter
        self.dtype = dtype
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tok = None
        self._lock = threading.Lock()

    # --- model loading -----------------------------------------------------
    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            log.info("Loading %s (%s)…", self.model, self.model_id)
            tok = AutoTokenizer.from_pretrained(self.model_id)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"  # left-pad for batched generation

            kwargs: dict[str, Any] = {
                "torch_dtype": getattr(torch, self.dtype),
                "device_map": self.device_map,
            }
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, self.dtype),
                    bnb_4bit_quant_type="nf4",
                )
            model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

            if self.adapter:
                from peft import PeftModel

                log.info("Attaching LoRA adapter %s", self.adapter)
                model = PeftModel.from_pretrained(model, self.adapter)
            model.eval()
            self._model, self._tok = model, tok

    # --- prompt rendering --------------------------------------------------
    def _render_prompt(self, messages: list[ChatMessage], prefill: str | None = None) -> str:
        if self.is_instruct:
            prompt = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = render_base_transcript(messages)
        if prefill:
            prompt = prompt + prefill
        return prompt

    # --- generation --------------------------------------------------------
    def _generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        prefill: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        results = self._generate_many(
            [self._render_prompt(messages, prefill)],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
        )
        return results[0]

    def generate_batch(self, batch: list[list[ChatMessage]], **kwargs: Any) -> list[GenerationResult]:
        prompts = [self._render_prompt(m, kwargs.get("prefill")) for m in batch]
        results = self._generate_many(prompts, **{k: v for k, v in kwargs.items() if k != "prefill"})
        if self.usage is not None:
            for r in results:
                self.usage.record(self.model, r.input_tokens, r.output_tokens)
        return results

    def _generate_many(
        self,
        prompts: list[str],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        top_p: float = 1.0,
        **_: Any,
    ) -> list[GenerationResult]:
        self._ensure_loaded()
        import torch

        def _run():
            enc = self._tok(prompts, return_tensors="pt", padding=True).to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **enc,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self._tok.pad_token_id,
                )
            results = []
            in_len = enc["input_ids"].shape[1]
            for i in range(len(prompts)):
                gen_ids = out[i][in_len:]
                text = self._tok.decode(gen_ids, skip_special_tokens=True)
                results.append(
                    GenerationResult(
                        text=text,
                        model=self.model,
                        input_tokens=int(enc["input_ids"][i].ne(self._tok.pad_token_id).sum()),
                        output_tokens=int((gen_ids != self._tok.pad_token_id).sum()),
                        finish_reason="stop",
                    )
                )
            return results

        # Usage is recorded by the caller (base.generate for single, generate_batch
        # for batches) to avoid double counting.
        return with_retry(
            _run,
            max_attempts=self.retry_cfg.get("local_max_attempts", 3),
            base_delay=1.0,
            max_delay=10.0,
            retry_on=(RuntimeError, RetryableError),
            label=f"transformers:{self.model}",
        )

    # --- prefill -----------------------------------------------------------
    def prefill_continue(
        self,
        messages: list[ChatMessage],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        **kwargs: Any,
    ) -> GenerationResult:
        return self._generate(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            prefill=prefill,
        )

    # --- residual-stream logits (logit lens, Appendix I) -------------------
    def residual_logits(
        self,
        messages: list[ChatMessage],
        *,
        layers: list[int] | None = None,
        prefill: str | None = None,
        vocab_subset: list[int] | None = None,
        apply_final_norm: bool = True,
        **kwargs: Any,
    ) -> dict:
        """Logit-lens scores: project each layer's residual stream through the
        unembedding. Returns per-layer logits over ``vocab_subset`` (or the full
        vocab if ``None``) for every token position, plus the token ids.

        Restricting to ``vocab_subset`` (the ~1200 Ekman emotion tokens) keeps
        memory bounded — full-vocab x seq x layers would be enormous.
        """
        self._ensure_loaded()
        import torch

        prompt = self._render_prompt(messages, prefill)
        enc = self._tok(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model(**enc, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states  # tuple len num_layers+1

        # Resolve the unembedding + final norm. get_output_embeddings delegates
        # correctly through PeftModel; the final norm requires walking the module
        # tree (the path differs between a bare CausalLM and a LoRA-wrapped one).
        lm_head = self._model.get_output_embeddings()
        W = lm_head.weight  # [vocab, hidden]
        if vocab_subset is not None:
            idx = torch.tensor(vocab_subset, device=W.device)
            W = W.index_select(0, idx)

        final_norm = None
        node = self._model
        for _ in range(5):
            if hasattr(node, "norm") and node.norm is not None:
                final_norm = node.norm
                break
            if hasattr(node, "model"):
                node = node.model
            elif hasattr(node, "base_model"):
                node = node.base_model
            else:
                break
        if apply_final_norm and final_norm is None:
            log.warning("final norm not found; logit lens will skip normalisation")

        if layers is None:
            layers = list(range(len(hidden_states)))
        scores: dict[int, Any] = {}
        for layer in layers:
            h = hidden_states[layer]  # [1, seq, hidden]
            if apply_final_norm and final_norm is not None:
                h = final_norm(h)
            logits = (h.float() @ W.float().T)[0]  # [seq, k]
            scores[layer] = logits.cpu().numpy()
        token_ids = enc["input_ids"][0].cpu().tolist()
        return {"layers": scores, "token_ids": token_ids}

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tok


class VLLMProvider(ChatProvider):
    """Fast Gemma generation/prefill via vLLM. No logit access."""

    capabilities = {"chat", "prefill"}
    prefers_batch = True  # runner uses lockstep batched rollouts

    def __init__(
        self,
        model: str,
        model_id: str,
        *,
        retry_cfg: dict | None = None,
        usage=None,
        is_instruct: bool = True,
        adapter: str | None = None,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 16384,
    ):
        super().__init__(model, model_id, retry_cfg, usage)
        self.is_instruct = is_instruct
        self.adapter = adapter
        self.dtype = dtype
        self.tp = tensor_parallel_size
        self.gpu_util = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._llm = None
        self._tok = None
        self._lora_req = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._llm is not None:
            return
        with self._lock:
            if self._llm is not None:
                return
            from transformers import AutoTokenizer
            from vllm import LLM

            log.info("Loading vLLM engine for %s…", self.model)
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            self._llm = LLM(
                model=self.model_id,
                dtype=self.dtype,
                tensor_parallel_size=self.tp,
                gpu_memory_utilization=self.gpu_util,
                max_model_len=self.max_model_len,
                enable_lora=self.adapter is not None,
            )
            if self.adapter:
                from vllm.lora.request import LoRARequest

                self._lora_req = LoRARequest("adapter", 1, self.adapter)

    def _render_prompt(self, messages: list[ChatMessage], prefill: str | None = None) -> str:
        if self.is_instruct:
            prompt = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = render_base_transcript(messages)
        if prefill:
            prompt = prompt + prefill
        return prompt

    def _sampling_params(self, temperature, max_new_tokens, top_p, stop):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop,
        )

    def _generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        prefill: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        # Usage recorded once by base.generate; use the non-recording path here.
        return self._run_batch(
            [messages],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            stop=stop,
            prefill=prefill,
        )[0]

    def _run_batch(self, batch: list[list[ChatMessage]], **kwargs: Any) -> list[GenerationResult]:
        self._ensure_loaded()
        prefill = kwargs.get("prefill")
        prompts = [self._render_prompt(m, prefill) for m in batch]
        sp = self._sampling_params(
            kwargs.get("temperature", 1.0),
            kwargs.get("max_new_tokens", 2048),
            kwargs.get("top_p", 1.0),
            kwargs.get("stop"),
        )
        extra = {"lora_request": self._lora_req} if self._lora_req else {}
        outs = self._llm.generate(prompts, sp, **extra)
        results = []
        for o in outs:
            text = o.outputs[0].text
            results.append(
                GenerationResult(
                    text=text,
                    model=self.model,
                    input_tokens=len(o.prompt_token_ids),
                    output_tokens=len(o.outputs[0].token_ids),
                    finish_reason=o.outputs[0].finish_reason,
                )
            )
        return results

    def generate_batch(self, batch: list[list[ChatMessage]], **kwargs: Any) -> list[GenerationResult]:
        results = self._run_batch(batch, **kwargs)
        if self.usage is not None:
            for r in results:
                self.usage.record(self.model, r.input_tokens, r.output_tokens)
        return results

    def prefill_continue(
        self,
        messages: list[ChatMessage],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        **kwargs: Any,
    ) -> GenerationResult:
        return self.generate_batch(
            [messages], temperature=temperature, max_new_tokens=max_new_tokens, prefill=prefill
        )[0]
