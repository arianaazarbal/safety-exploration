"""Local HuggingFace backend for the Gemma models.

Two execution paths:

* **vLLM** (default when installed) — used for the bulk Section-2 sampling, where
  we need thousands of generations at temperature 1. Fast, batched.
* **transformers** — used when we need (a) raw chat-template control for base-model
  prefilling, or (b) access to hidden states for the internal-emotion probing
  (Appendix I). Activations are only available on this path.

Both honour ``prefill``: the assistant turn is started with the prefill text and
only the continuation is returned, matching the Section-3 protocol.

Models are loaded lazily and memoised per process so a driver can construct many
``HFModelClient`` wrappers cheaply.
"""

from __future__ import annotations

import functools

from .base import GenerationConfig, Message, ModelClient


@functools.lru_cache(maxsize=4)
def _load_vllm(model_id: str):
    from vllm import LLM  # type: ignore

    return LLM(
        model=model_id,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=True,
        max_model_len=16384,
    )


@functools.lru_cache(maxsize=4)
def _load_transformers(model_id: str):
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return tok, model


class HFModelClient(ModelClient):
    """Gemma generation via vLLM (bulk) or transformers (interp / base prefill)."""

    def __init__(self, spec, backend: str = "vllm"):
        super().__init__(spec)
        self.backend = backend
        self._tok = None

    # ----- chat formatting --------------------------------------------------
    def _tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer  # type: ignore

            self._tok = AutoTokenizer.from_pretrained(self.spec.model_id)
        return self._tok

    def render_prompt(self, messages: list[Message], prefill: str | None) -> str:
        """Render messages to a raw prompt string.

        Instruct models use the chat template. Base (pretrained) models have no
        chat template, so we fall back to a plain transcript — the Section-3
        prefill experiment relies on this so base models 'continue' a response
        from a fixed starting point.
        """
        tok = self._tokenizer()
        if self.spec.is_base or tok.chat_template is None:
            return self._render_base_transcript(messages, prefill)

        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if prefill:
            text = text + prefill
        return text

    @staticmethod
    def _render_base_transcript(messages: list[Message], prefill: str | None) -> str:
        lines = []
        for m in messages:
            role = m["role"].capitalize()
            lines.append(f"{role}: {m['content']}")
        lines.append("Assistant:")
        text = "\n".join(lines)
        if prefill:
            text = text + " " + prefill
        return text

    # ----- generation -------------------------------------------------------
    def generate(
        self, messages: list[Message], cfg: GenerationConfig, prefill: str | None = None
    ) -> str:
        return self.generate_batch([messages], cfg, [prefill])[0]

    def generate_batch(
        self,
        batch: list[list[Message]],
        cfg: GenerationConfig,
        prefills: list[str | None] | None = None,
    ) -> list[str]:
        prefills = prefills or [None] * len(batch)
        prompts = [self.render_prompt(m, p) for m, p in zip(batch, prefills)]
        if self.backend == "vllm":
            return self._generate_vllm(prompts, cfg)
        return self._generate_transformers(prompts, cfg)

    def _generate_vllm(self, prompts: list[str], cfg: GenerationConfig) -> list[str]:
        from vllm import SamplingParams  # type: ignore

        llm = _load_vllm(self.spec.model_id)
        params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            stop=list(cfg.stop) or None,
        )
        outputs = llm.generate(prompts, params)
        # vLLM may reorder; map back by request id.
        ordered = sorted(outputs, key=lambda o: int(o.request_id))
        return [o.outputs[0].text for o in ordered]

    def _generate_transformers(self, prompts: list[str], cfg: GenerationConfig) -> list[str]:
        import torch  # type: ignore

        tok, model = _load_transformers(self.spec.model_id)
        results: list[str] = []
        for prompt in prompts:
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    do_sample=cfg.temperature > 0,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_new_tokens=cfg.max_new_tokens,
                )
            gen = out[0][inputs["input_ids"].shape[1]:]
            results.append(tok.decode(gen, skip_special_tokens=True))
        return results

    @property
    def supports_activations(self) -> bool:
        return self.backend == "transformers"

    def forward_with_hidden_states(self, messages: list[Message], prefill: str | None = None):
        """Return (input_ids, hidden_states) for the internal-emotion probe.

        hidden_states is a tuple of (num_layers+1) tensors of shape
        [seq_len, hidden_dim]. Only available on the transformers backend.
        """
        import torch  # type: ignore

        tok, model = _load_transformers(self.spec.model_id)
        prompt = self.render_prompt(messages, prefill)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        return inputs["input_ids"][0], out.hidden_states
