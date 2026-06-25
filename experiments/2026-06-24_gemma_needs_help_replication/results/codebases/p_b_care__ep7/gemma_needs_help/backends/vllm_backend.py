"""Local Gemma inference via vLLM.

Handles three things the experiments need:
  * batched chat generation (Section 2 / 4 evals),
  * prefill continuations (Section 3), where we seed the assistant turn and
    return only the model-written continuation, and
  * LoRA adapters (Section 4 finetunes and the Appendix I layer ablations),
    loaded on top of the base instruct weights.

Imports of torch/vllm are deferred to construction time so the module can be
imported in environments without a GPU (e.g. for static analysis).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import config
from .base import GenerationRequest
from .gemma_format import GEMMA_STOP, format_gemma_prompt

if TYPE_CHECKING:  # pragma: no cover
    from ..config import ModelSpec


class VLLMBackend:
    """Serve one Gemma model (optionally with a LoRA adapter) via vLLM."""

    supports_prefill = True

    def __init__(self, spec: "ModelSpec", **engine_kwargs):
        from vllm import LLM  # deferred import
        from vllm.lora.request import LoRARequest

        self.spec = spec
        self.spec_name = spec.name
        self._LoRARequest = LoRARequest
        self._lora_request = None

        if spec.kind == "finetune":
            # model_id is the LoRA adapter path; base weights are the instruct
            # model the finetune was trained from.
            base_model = config.BASE_FINETUNE_MODEL.model_id
            self._lora_request = LoRARequest(spec.name, 1, spec.model_id)
            engine_kwargs.setdefault("enable_lora", True)
            engine_kwargs.setdefault("max_lora_rank", config.DPO.lora.r)
        else:
            base_model = spec.model_id

        engine_kwargs.setdefault("dtype", "bfloat16")
        engine_kwargs.setdefault("trust_remote_code", True)
        # Gemma-3 is multimodal; we only need the text tower for these evals.
        self.llm = LLM(model=base_model, **engine_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    # ------------------------------------------------------------------ #
    def _sampling_params(self, req: GenerationRequest):
        from vllm import SamplingParams

        stop = list(GEMMA_STOP)
        if req.stop:
            stop.extend(req.stop)
        return SamplingParams(
            n=req.n,
            temperature=req.temperature,
            top_p=1.0,
            max_tokens=req.max_tokens,
            stop=stop,
            seed=req.seed,
        )

    def _prompt(self, req: GenerationRequest) -> str:
        return format_gemma_prompt(
            req.messages,
            add_generation_prompt=True,
            prefill=req.prefill,
        )

    # ------------------------------------------------------------------ #
    def generate(self, request: GenerationRequest) -> list[str]:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: list[GenerationRequest]) -> list[list[str]]:
        if not requests:
            return []
        prompts = [self._prompt(r) for r in requests]
        # vLLM requires uniform SamplingParams length == prompts length.
        sps = [self._sampling_params(r) for r in requests]
        outputs = self.llm.generate(
            prompts,
            sps,
            lora_request=self._lora_request,
            use_tqdm=len(prompts) > 1,
        )
        # vLLM preserves input order.
        results: list[list[str]] = []
        for out in outputs:
            results.append([o.text for o in out.outputs])
        return results
