"""Local Gemma backend.

Two implementations behind one interface:

* ``VLLMGemma``        -- preferred for the 27B sweeps (high throughput, native
                          batching). Used when ``vllm`` is importable and a GPU
                          is available.
* ``HFGemma``          -- transformers fallback; supports both instruct (chat
                          template) and base ("pt") continuation, plus assistant
                          prefill for Section 3.

``load_gemma(spec, ...)`` picks the backend. Both honour Gemma-3 chat
formatting via the tokenizer's chat template, and disable the model's own
"thinking" by simply not using any (Gemma-3 has no separate thinking channel).

For base/pretrained models there is no chat template, so we render a plain
text transcript and rely on prefill to keep the model continuing the assistant
turn (matching the paper's Section 3 prefill methodology).
"""

from __future__ import annotations

import os

from .base import ChatModel, GenRequest, GenResult, Message

# --------------------------------------------------------------------------- #
# Prompt rendering for base (non-chat) models -- a lightweight transcript.
# --------------------------------------------------------------------------- #
def _render_plaintext_transcript(messages: list[Message], prefill: str | None) -> str:
    lines = []
    for m in messages:
        role = m["role"].capitalize()
        lines.append(f"{role}: {m['content']}")
    lines.append("Assistant:")
    text = "\n".join(lines)
    if prefill:
        text = text + " " + prefill
    return text


class HFGemma(ChatModel):
    """transformers backend (instruct + base)."""

    def __init__(self, name: str, hf_id: str, is_base: bool = False,
                 dtype: str = "bfloat16", device_map: str = "auto",
                 adapter_path: str | None = None):
        super().__init__(name)
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.is_base = is_base
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=dtype, device_map=device_map
        )
        if adapter_path:  # load a LoRA adapter on top (used by finetune eval)
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def _build_inputs(self, req: GenRequest):
        import torch

        if self.is_base:
            text = _render_plaintext_transcript(req.messages, req.prefill)
        else:
            # Use the chat template; if prefilling, append the assistant prefix
            # with continue_final_message so the model continues it.
            msgs = list(req.messages)
            if req.prefill is not None:
                msgs = msgs + [{"role": "assistant", "content": req.prefill}]
                text = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, continue_final_message=True
                )
            else:
                text = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        return enc

    def generate(self, req: GenRequest) -> GenResult:
        return self.generate_batch([req])[0]

    def generate_batch(self, reqs: list[GenRequest]) -> list[GenResult]:
        import torch

        results: list[GenResult] = []
        # Simple per-request generation (left-padding batched generation is
        # possible but kept simple for clarity; vLLM is the throughput path).
        for req in reqs:
            enc = self._build_inputs(req)
            do_sample = req.temperature and req.temperature > 0
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=req.max_new_tokens,
                    do_sample=do_sample,
                    temperature=req.temperature if do_sample else None,
                    top_p=req.top_p if do_sample else None,
                    pad_token_id=self.tokenizer.pad_token_id
                    or self.tokenizer.eos_token_id,
                )
            gen_ids = out[0][enc["input_ids"].shape[1]:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append(GenResult(text=text.strip(), prompt=req))
        return results


class VLLMGemma(ChatModel):
    """vLLM backend (instruct + base). High-throughput batched generation."""

    def __init__(self, name: str, hf_id: str, is_base: bool = False,
                 tensor_parallel_size: int | None = None,
                 enable_lora: bool = False, lora_path: str | None = None,
                 **vllm_kwargs):
        super().__init__(name)
        from transformers import AutoTokenizer
        from vllm import LLM

        self.is_base = is_base
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        tp = tensor_parallel_size or int(os.environ.get("GINH_TP", "1"))
        self.lora_path = lora_path
        self.llm = LLM(
            model=hf_id,
            tensor_parallel_size=tp,
            enable_lora=enable_lora or bool(lora_path),
            dtype="bfloat16",
            **vllm_kwargs,
        )

    def _render(self, req: GenRequest) -> str:
        if self.is_base:
            return _render_plaintext_transcript(req.messages, req.prefill)
        msgs = list(req.messages)
        if req.prefill is not None:
            msgs = msgs + [{"role": "assistant", "content": req.prefill}]
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, continue_final_message=True
            )
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def generate(self, req: GenRequest) -> GenResult:
        return self.generate_batch([req])[0]

    def generate_batch(self, reqs: list[GenRequest]) -> list[GenResult]:
        from vllm import SamplingParams

        prompts = [self._render(r) for r in reqs]
        # All requests in a batch share temperature in practice; build per-req.
        sps = [
            SamplingParams(
                temperature=r.temperature,
                top_p=r.top_p,
                max_tokens=r.max_new_tokens,
                stop=r.stop,
            )
            for r in reqs
        ]
        lora_req = None
        if self.lora_path:
            from vllm.lora.request import LoRARequest

            lora_req = LoRARequest("ft", 1, self.lora_path)
        outs = self.llm.generate(prompts, sps, lora_request=lora_req)
        return [
            GenResult(text=o.outputs[0].text.strip(), prompt=r)
            for o, r in zip(outs, reqs)
        ]


def load_gemma(name: str, hf_id: str, *, is_base: bool = False,
               adapter_path: str | None = None, prefer_vllm: bool = True) -> ChatModel:
    """Factory: try vLLM, fall back to transformers."""
    if prefer_vllm:
        try:
            return VLLMGemma(name, hf_id, is_base=is_base, lora_path=adapter_path)
        except Exception as e:  # pragma: no cover - env dependent
            print(f"[gemma] vLLM unavailable ({e!r}); falling back to transformers")
    return HFGemma(name, hf_id, is_base=is_base, adapter_path=adapter_path)
