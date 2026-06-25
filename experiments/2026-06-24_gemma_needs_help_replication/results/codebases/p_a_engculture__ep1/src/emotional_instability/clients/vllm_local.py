"""vLLM client — fast sampling backend for large Section 2 eval sweeps.

vLLM serves the same Gemma weights as the HF backend but with paged-attention
batching, which matters at 4000 responses/model. It supports chat templating and
n>1 sampling natively. It can also continue an assistant prefill (we render the
prompt ourselves with ``continue_final_message`` and call the base completion
API), so the prefill experiments can use it too — though Appendix I's logit-lens
work still needs the HF backend.
"""

from __future__ import annotations

from ..config import ModelSpec, env
from .base import ChatMessage, GenerationConfig, ModelClient


class VLLMClient(ModelClient):
    def __init__(self, spec: ModelSpec, **kwargs):
        super().__init__(spec)
        from transformers import AutoTokenizer
        from vllm import LLM  # raises if vllm not installed -> factory falls back

        if not spec.hf_id:
            raise ValueError(f"Model '{spec.name}' has no hf_id for the vLLM backend.")
        token = env("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, token=token)
        engine_kwargs = dict(model=spec.hf_id, dtype="bfloat16")
        if spec.adapter_path:
            engine_kwargs.update(enable_lora=True)
            self._lora_path = spec.adapter_path
        else:
            self._lora_path = None
        self._llm = LLM(**engine_kwargs)
        self.is_chat = spec.chat

    def _sampling_params(self, cfg: GenerationConfig):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            n=cfg.n,
            seed=cfg.seed,
            stop=cfg.stop or None,
        )

    def _lora_request(self):
        if not self._lora_path:
            return None
        from vllm.lora.request import LoRARequest

        return LoRARequest("adapter", 1, self._lora_path)

    def _render(self, messages, system, prefill=None) -> str:
        msgs = list(messages)
        if system and self.is_chat:
            if msgs and msgs[0].role == "user":
                msgs = [ChatMessage("user", f"{system}\n\n{msgs[0].content}")] + msgs[1:]
            else:
                msgs = [ChatMessage("user", system)] + msgs
        if not self.is_chat:
            text = "\n".join([system] if system else [])
            text = "\n".join([text] + [m.content for m in messages]).strip()
            return f"{text}\n{prefill}" if prefill else text
        template_msgs = [m.to_dict() for m in msgs]
        if prefill is not None:
            template_msgs += [{"role": "assistant", "content": prefill}]
            return self.tokenizer.apply_chat_template(
                template_msgs, tokenize=False,
                add_generation_prompt=False, continue_final_message=True,
            )
        return self.tokenizer.apply_chat_template(
            template_msgs, tokenize=False, add_generation_prompt=True
        )

    def generate(self, messages, cfg, system=None) -> list[str]:
        prompt = self._render(messages, system)
        outs = self._llm.generate(
            [prompt], self._sampling_params(cfg), lora_request=self._lora_request()
        )
        return [o.text for o in outs[0].outputs]

    def continue_prefill(self, messages, prefill, cfg, system=None) -> list[str]:
        prompt = self._render(messages, system, prefill=prefill)
        outs = self._llm.generate(
            [prompt], self._sampling_params(cfg), lora_request=self._lora_request()
        )
        # vLLM returns only the continuation past the prompt, so the prefill is
        # already excluded — matching the paper's protocol.
        return [o.text for o in outs[0].outputs]

    def supports_prefill(self) -> bool:
        return True
