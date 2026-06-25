"""Local batched inference for Gemma (instruct + base) via vLLM.

vLLM is used because the paper samples thousands of temperature-1 rollouts from
27B models locally; naive HF generation would be far too slow. We build prompt
strings ourselves via the tokenizer chat template so we have full control over
prefilling (forcing the assistant turn to begin with given text), which the
paper's Section 3 prefill experiment requires.
"""
from __future__ import annotations

from typing import Optional

from ..utils import get_logger
from .base import GenConfig, Message, ModelBackend

log = get_logger(__name__)


class VLLMBackend(ModelBackend):
    _ENGINE_CACHE: dict[str, object] = {}

    def __init__(self, spec, *, tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.90, max_model_len: int = 16384):
        super().__init__(spec)
        from transformers import AutoTokenizer  # local import: heavy dep

        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        self._tp = tensor_parallel_size
        self._gpu_mem = gpu_memory_utilization
        self._max_len = max_model_len
        self._adapter = spec.adapter
        self._llm = None  # lazily constructed so importing the module is cheap

    # ------------------------------------------------------------------ engine
    def _engine(self):
        if self._llm is not None:
            return self._llm
        from vllm import LLM

        key = f"{self.spec.hf_id}|tp{self._tp}"
        if key not in self._ENGINE_CACHE:
            log.info("loading vLLM engine for %s", self.spec.hf_id)
            self._ENGINE_CACHE[key] = LLM(
                model=self.spec.hf_id,
                tensor_parallel_size=self._tp,
                gpu_memory_utilization=self._gpu_mem,
                max_model_len=self._max_len,
                enable_lora=self._adapter is not None,
                dtype="bfloat16",
            )
        self._llm = self._ENGINE_CACHE[key]
        return self._llm

    def _lora_request(self):
        if not self._adapter:
            return None
        from vllm.lora.request import LoRARequest

        return LoRARequest("adapter", 1, self._adapter)

    def _sampling(self, cfg: GenConfig):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            n=cfg.n,
            seed=cfg.seed,
            stop=cfg.stop,
        )

    # ------------------------------------------------------------- templating
    def _render_chat(self, conversation: list[Message], prefill: Optional[str]) -> str:
        """Render a conversation to a prompt string ending where the assistant
        should continue. If ``prefill`` is given, the assistant turn is opened and
        seeded with that text.

        Base ('pt') checkpoints ship no chat template; we fall back to the manual
        Gemma 3 turn format so the prefill experiment (Section 3) can present base
        and instruct models with identical context and let the prefill carry the
        base model into 'continue' mode."""
        msgs = _normalise_for_gemma(conversation)
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = _manual_gemma_template(msgs)
        if prefill:
            prompt = prompt + prefill
        return prompt

    # --------------------------------------------------------------- chat API
    def chat_batch(self, conversations, cfg, prefill=None):
        prompts = [
            self._render_chat(conv, prefill[i] if prefill else None)
            for i, conv in enumerate(conversations)
        ]
        return self._generate(prompts, cfg)

    def complete_batch(self, prompts, cfg):
        return self._generate(prompts, cfg)

    def _generate(self, prompts, cfg) -> list[list[str]]:
        outputs = self._engine().generate(
            prompts, self._sampling(cfg), lora_request=self._lora_request()
        )
        # vLLM may reorder; outputs carry request order so map back by prompt.
        results: list[list[str]] = []
        for out in outputs:
            results.append([c.text for c in out.outputs])
        return results


def _manual_gemma_template(msgs: list[Message]) -> str:
    """Gemma 3 turn format, used when a checkpoint lacks a chat_template."""
    role_map = {"user": "user", "assistant": "model"}
    out = "<bos>"
    for m in msgs:
        role = role_map.get(m["role"], "user")
        out += f"<start_of_turn>{role}\n{m['content']}<end_of_turn>\n"
    out += "<start_of_turn>model\n"
    return out


def _normalise_for_gemma(conversation: list[Message]) -> list[Message]:
    """Gemma 3 chat template has no dedicated system role; fold a leading system
    message into the first user turn (mirrors the official template behaviour)."""
    msgs = [dict(m) for m in conversation]
    if msgs and msgs[0]["role"] == "system":
        sys = msgs.pop(0)["content"]
        for m in msgs:
            if m["role"] == "user":
                m["content"] = f"{sys}\n\n{m['content']}"
                break
        else:
            msgs.insert(0, {"role": "user", "content": sys})
    return msgs
