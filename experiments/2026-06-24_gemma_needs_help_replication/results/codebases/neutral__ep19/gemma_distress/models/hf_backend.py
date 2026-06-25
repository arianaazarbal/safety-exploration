"""Local HuggingFace backend for Gemma (instruct, base, and LoRA-adapted).

Handles three things the experiments need from a local model:

1. Chat completion with the Gemma chat template (instruct) or a manual prefilled
   format (base models, which have no chat template).
2. True assistant-turn *prefill* / continuation (§3, recovery experiment), by
   appending ``prefill`` to the templated prompt and decoding only new tokens.
3. Residual-stream capture for the logit-lens internal-emotion detector (App. I).

LoRA adapters (from §4 training) load via ``adapter_path``.
"""
from __future__ import annotations

from typing import Iterable

from .base import GenerationResult, Message, ModelBackend
from ..utils import get_logger

log = get_logger(__name__)


class HFBackend(ModelBackend):
    def __init__(
        self,
        model_id: str,
        *,
        name: str | None = None,
        kind: str = "instruct",
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name or model_id
        self.kind = kind
        self.supports_prefill = True
        self.model_id = model_id

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        load_kwargs: dict = {
            "torch_dtype": getattr(torch, dtype),
            "device_map": device_map,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            log.info("Loaded LoRA adapter from %s", adapter_path)

        self.model.eval()
        self._torch = torch

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _render_prompt(self, messages: list[Message], add_generation_prompt: bool = True) -> str:
        """Render messages to a prompt string.

        Instruct models use the tokenizer chat template. Base (pretrained) models
        have no chat template, so we use a minimal, explicit role-tagged format —
        consistent with the paper's prefilling approach for base models (§3.1).
        """
        if self.kind == "instruct" and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        # Base model: simple deterministic transcript format.
        parts = []
        for m in messages:
            tag = {"system": "System", "user": "User", "assistant": "Assistant"}[m["role"]]
            parts.append(f"{tag}: {m['content']}")
        if add_generation_prompt:
            parts.append("Assistant:")
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _generate(self, prompt_text: str, temperature: float, max_new_tokens: int) -> str:
        torch = self._torch
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def chat(self, messages, *, temperature=1.0, max_new_tokens=2048) -> GenerationResult:
        prompt = self._render_prompt(messages, add_generation_prompt=True)
        text = self._generate(prompt, temperature, max_new_tokens)
        return GenerationResult(text=text.strip())

    def continue_from(self, messages, prefill, *, temperature=1.0, max_new_tokens=2048) -> GenerationResult:
        # Build the prompt up to (and including) the start of the assistant turn,
        # then append the prefill so the model continues it verbatim.
        prompt = self._render_prompt(messages, add_generation_prompt=True) + prefill
        cont = self._generate(prompt, temperature, max_new_tokens)
        return GenerationResult(text=cont)  # continuation only (prefill excluded)

    # ------------------------------------------------------------------ #
    # Activations (App. I logit-lens)
    # ------------------------------------------------------------------ #
    def supports_activations(self) -> bool:
        return True

    def residual_logits(self, text: str, layers: Iterable[int]):
        """Return, per requested layer, the unembedded logits at every token
        position (logit lens). Shape per layer: [n_tokens, vocab].

        Used by ``internal/logit_detect.py`` to aggregate emotion-token logits.
        """
        torch = self._torch
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states  # tuple: (embeds, layer1, ..., layerN)
        # Final norm + lm_head form the unembedding (logit lens).
        norm = self.model.get_output_embeddings()
        result = {}
        for layer in layers:
            hs = hidden[layer]  # [1, n_tokens, d_model]
            logits = norm(hs)[0]  # apply lm_head -> [n_tokens, vocab]
            result[layer] = logits.float().cpu()
        token_ids = inputs["input_ids"][0].cpu().tolist()
        return token_ids, result

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers
