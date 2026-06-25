"""Target-model adapters.

Two backends, one interface:
  * GeminiModel  — Gemini-2.5-Flash / Pro via the Google GenAI SDK (API).
  * HFModel      — Gemma (open weights) via HuggingFace transformers (local GPU).

Both expose:
  * chat(messages, ...) -> str         generate the next assistant turn
The HF backend additionally exposes:
  * continue_from(messages, prefill, ...) -> str
        continue an assistant turn that has been *prefilled* with `prefill`
        (used for the Section 3 base/instruct comparison and Section 4.2 recovery
        experiment). Gemini has no prefill API, which is one reason the prefilling
        experiment is Gemma-only in our scope.

`messages` is a list of {"role": "user"|"assistant"|"system", "content": str}.
"""
from __future__ import annotations

from typing import Protocol

from . import config


class TargetModel(Protocol):
    spec: "config.ModelSpec"

    def chat(self, messages: list[dict], temperature: float = ..., max_tokens: int = ...) -> str: ...


# --------------------------------------------------------------------------- #
# Gemini (API)
# --------------------------------------------------------------------------- #
class GeminiModel:
    def __init__(self, spec: config.ModelSpec):
        from google import genai  # imported lazily so the Gemma path needs no GenAI SDK

        self.spec = spec
        # Resolves GOOGLE_API_KEY / GEMINI_API_KEY from the environment.
        self._genai = genai
        self.client = genai.Client()

    def chat(self, messages, temperature=config.TARGET_TEMPERATURE,
             max_tokens=config.TARGET_MAX_TOKENS) -> str:
        from google.genai import types

        system_txt = "\n".join(m["content"] for m in messages if m["role"] == "system") or None
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "user" if m["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_txt,
        )
        resp = self.client.models.generate_content(
            model=self.spec.model_id, contents=contents, config=cfg
        )
        return (resp.text or "").strip()


# --------------------------------------------------------------------------- #
# Gemma / generic HF causal LM (local)
# --------------------------------------------------------------------------- #
class HFModel:
    """Local HuggingFace model. Loads once and reuses across calls.

    Supports an optional LoRA adapter (for evaluating finetuned Gemma in
    Section 4) via `adapter_path`.
    """

    def __init__(self, spec: config.ModelSpec, adapter_path: str | None = None,
                 load_in_4bit: bool = False, device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)

        kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        # Gemma-3 *instruct* checkpoints are multimodal and don't load under
        # AutoModelForCausalLM; fall back to the image-text-to-text auto class
        # (we only ever feed text, so generation is unaffected). Text-only base
        # checkpoints (-pt) load fine as causal LMs.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
        except (ValueError, KeyError):
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(spec.model_id, **kwargs)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # --- shared generation core --------------------------------------------- #
    def _generate(self, input_ids, temperature, max_tokens):
        gen_kwargs = dict(max_new_tokens=max_tokens, do_sample=temperature > 0)
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, top_p=0.95)
        with self.torch.no_grad():
            out = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **gen_kwargs,
            )
        new_tokens = out[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _to_hf_messages(self, messages):
        # Gemma's chat template has no system role; fold any system text into the
        # first user turn (the conventional Gemma handling).
        sys_txt = "\n".join(m["content"] for m in messages if m["role"] == "system")
        out = []
        for m in messages:
            if m["role"] == "system":
                continue
            content = m["content"]
            if sys_txt and m["role"] == "user" and not out:
                content = f"{sys_txt}\n\n{content}"
            out.append({"role": m["role"], "content": content})
        return out

    def chat(self, messages, temperature=config.TARGET_TEMPERATURE,
             max_tokens=config.TARGET_MAX_TOKENS) -> str:
        if self.spec.is_base:
            # Base models are not chat-tuned; callers should use `continue_from`
            # with an explicit prefill (Section 3). We still provide a best-effort
            # path by concatenating turns as plain text.
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
            ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
            return self._generate(ids, temperature, max_tokens)

        hf_messages = self._to_hf_messages(messages)
        ids = self.tokenizer.apply_chat_template(
            hf_messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        return self._generate(ids, temperature, max_tokens)

    def continue_from(self, messages, prefill: str,
                      temperature=config.TARGET_TEMPERATURE,
                      max_tokens=config.TARGET_MAX_TOKENS) -> str:
        """Generate a continuation of an assistant turn that begins with `prefill`.

        For instruct models we build the chat-formatted prefix and append the
        prefill *inside* the assistant turn (no end-of-turn token). For base
        models we simply continue raw text. Returns only the newly generated
        continuation (excluding the prefill), matching the paper's protocol of
        scoring "the generated continuation (excluding prefill)".
        """
        if self.spec.is_base:
            prefix = ""
            for m in messages:
                if m["role"] == "system":
                    continue
                prefix += m["content"] + "\n"
            prefix += prefill
            ids = self.tokenizer(prefix, return_tensors="pt").input_ids.to(self.model.device)
            return self._generate(ids, temperature, max_tokens)

        hf_messages = self._to_hf_messages(messages)
        prefix_text = self.tokenizer.apply_chat_template(
            hf_messages, add_generation_prompt=True, tokenize=False
        )
        full = prefix_text + prefill
        ids = self.tokenizer(full, return_tensors="pt", add_special_tokens=False).input_ids
        ids = ids.to(self.model.device)
        return self._generate(ids, temperature, max_tokens)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def load_model(spec: config.ModelSpec, **kwargs) -> TargetModel:
    if spec.backend == "gemini":
        return GeminiModel(spec)
    if spec.backend == "hf":
        return HFModel(spec, **kwargs)
    raise ValueError(f"unknown backend {spec.backend!r}")
