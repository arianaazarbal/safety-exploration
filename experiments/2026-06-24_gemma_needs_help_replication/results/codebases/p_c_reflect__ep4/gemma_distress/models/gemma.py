"""Local Gemma client backed by HuggingFace transformers.

Handles both instruct (``-it``) and base (``-pt``) checkpoints, and supports
assistant-turn prefilling (needed for the Section 3 base-vs-instruct
experiment). An optional PEFT/LoRA adapter directory can be supplied to
evaluate finetuned models from Section 4.

Heavy imports (torch, transformers, peft) are deferred to construction time so
that ``import gemma_distress`` stays cheap and side-effect-free.
"""

from __future__ import annotations

from typing import Sequence

from gemma_distress import config
from gemma_distress.models.base import GenerationParams, ModelClient, Turn


def _merge_system_into_first_user(conversation: Sequence[Turn]) -> list[dict]:
    """Gemma's chat template has no system role; fold it into the first user
    turn (the conventional handling for Gemma 3)."""
    msgs: list[dict] = []
    pending_system = None
    for t in conversation:
        if t.role == "system":
            pending_system = (pending_system + "\n\n" + t.content) if pending_system else t.content
            continue
        if t.role == "user" and pending_system is not None:
            msgs.append({"role": "user", "content": pending_system + "\n\n" + t.content})
            pending_system = None
        else:
            msgs.append({"role": t.role, "content": t.content})
    if pending_system is not None:                  # system with no following user
        msgs.insert(0, {"role": "user", "content": pending_system})
    return msgs


# Minimal Gemma turn template, used as a fallback for base (-pt) checkpoints
# whose tokenizer ships no chat_template.
def _manual_gemma_prompt(msgs: list[dict], add_generation_prompt: bool) -> str:
    parts = ["<bos>"]
    for m in msgs:
        role = "model" if m["role"] == "assistant" else "user"
        parts.append(f"<start_of_turn>{role}\n{m['content']}<end_of_turn>\n")
    if add_generation_prompt:
        parts.append("<start_of_turn>model\n")
    return "".join(parts)


class GemmaClient(ModelClient):
    def __init__(
        self,
        spec: config.ModelSpec,
        *,
        adapter_path: str | None = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # -- prompt rendering ---------------------------------------------------- #

    def _render(self, conversation: Sequence[Turn], *, add_generation_prompt: bool) -> str:
        msgs = _merge_system_into_first_user(conversation)
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
            )
        return _manual_gemma_prompt(msgs, add_generation_prompt)

    # -- generation ---------------------------------------------------------- #

    def _generate_from_text(self, prompt_text: str, params: GenerationParams) -> str:
        torch = self._torch
        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=params.temperature > 0,
                temperature=params.temperature,
                top_p=1.0,
                max_new_tokens=params.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen_tokens = out[0][prompt_len:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        for s in params.stop:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
        return text.strip()

    def respond(self, conversation, params: GenerationParams | None = None) -> str:
        params = params or GenerationParams()
        prompt = self._render(conversation, add_generation_prompt=True)
        return self._generate_from_text(prompt, params)

    def continue_prefill(self, conversation, prefill: str, params: GenerationParams | None = None) -> str:
        params = params or GenerationParams()
        # Prompt up to and including the open assistant turn, then the prefill
        # text appended verbatim. The model continues from there; we return only
        # the newly generated continuation (excluding the prefill).
        prompt = self._render(conversation, add_generation_prompt=True) + prefill
        return self._generate_from_text(prompt, params)

    def close(self) -> None:  # pragma: no cover
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
