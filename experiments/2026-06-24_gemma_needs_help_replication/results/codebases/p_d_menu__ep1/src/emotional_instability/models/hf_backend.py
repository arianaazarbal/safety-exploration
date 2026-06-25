"""Local HuggingFace backend for Gemma subject models (Appendix B.1).

Handles:
  * instruct chat models (gemma-3-*-it) via the chat template;
  * base/pretrained models (gemma-3-*-pt), which are prefill-only;
  * optional LoRA adapters (Section 4 DPO/SFT finetunes);
  * prefilled assistant continuation (Section 3), excluding the prefill from
    the returned text.

Inference is at temperature 1 (paper default). Model loading is lazy and cached
per process so the 27B weights are only materialised once.
"""
from __future__ import annotations

from dataclasses import dataclass

from .base import GenerationResult, Message

# torch/transformers are heavy; import lazily inside methods so the rest of the
# package (configs, prompts, analysis) imports without a GPU stack present.


@dataclass
class HFBackend:
    name: str
    hf_id: str
    is_chat: bool = True
    adapter_path: str | None = None
    dtype: str = "bfloat16"
    device_map: str = "auto"
    load_in_4bit: bool = False

    def __post_init__(self) -> None:
        self.supports_chat = self.is_chat
        self.supports_prefill = True
        self._model = None
        self._tokenizer = None

    # ---- loading ----------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.hf_id)
        kwargs: dict = {"device_map": self.device_map}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = getattr(torch, self.dtype)

        model = AutoModelForCausalLM.from_pretrained(self.hf_id, **kwargs)
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
            model = model.merge_and_unload()
        model.eval()
        self._model, self._tokenizer = model, tok

    # ---- prompt construction ---------------------------------------------
    def _render(self, messages: list[Message], prefill: str | None) -> str:
        """Render messages to a single prompt string.

        For chat models we use the chat template; when a prefill is supplied we
        request continuation of the final assistant message. For base models we
        fall back to a plain concatenation (base models are always used with a
        prefill in this codebase).
        """
        tok = self._tokenizer
        if self.is_chat:
            msgs = list(messages)
            if prefill is not None:
                msgs = msgs + [{"role": "assistant", "content": prefill}]
                return self._apply_template(tok, msgs, continue_final_message=True)
            return self._apply_template(tok, msgs, add_generation_prompt=True)
        # Base model: no chat template. Concatenate turns plainly and append the
        # prefill so the model continues it.
        parts = []
        for m in messages:
            parts.append(m["content"])
        if prefill is not None:
            parts.append(prefill)
        return "\n\n".join(parts)

    @staticmethod
    def _apply_template(tok, msgs: list[Message], **kwargs) -> str:
        """Apply the chat template, folding a leading system message into the
        first user turn if the model's template rejects the system role (some
        Gemma chat templates do). This keeps the welfare opt-out's system
        delivery working across templates."""
        try:
            return tok.apply_chat_template(msgs, tokenize=False, **kwargs)
        except Exception:
            if msgs and msgs[0]["role"] == "system":
                folded = list(msgs[1:])
                sys_text = msgs[0]["content"]
                for i, m in enumerate(folded):
                    if m["role"] == "user":
                        folded[i] = {
                            "role": "user",
                            "content": f"{sys_text}\n\n{m['content']}",
                        }
                        break
                return tok.apply_chat_template(folded, tokenize=False, **kwargs)
            raise

    def _run(
        self,
        prompt_text: str,
        *,
        temperature: float,
        max_new_tokens: int,
        seed: int | None,
        stop: list[str] | None,
    ) -> GenerationResult:
        import torch

        self._ensure_loaded()
        tok, model = self._tokenizer, self._model
        if seed is not None:
            torch.manual_seed(seed)
        inputs = tok(prompt_text, return_tensors="pt").to(model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=1.0,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = tok.decode(new_tokens, skip_special_tokens=True)
        if stop:
            text = _truncate_at_stop(text, stop)
        return GenerationResult(
            text=text, n_new_tokens=int(new_tokens.shape[0]), meta={"backend": "hf"}
        )

    # ---- public API -------------------------------------------------------
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        if not self.is_chat:
            raise ValueError(
                f"{self.name} is a base model; use continue_text() with a prefill."
            )
        prompt_text = self._render(messages, prefill=None)
        return self._run(
            prompt_text,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
            stop=stop,
        )

    def continue_text(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> GenerationResult:
        prompt_text = self._render(messages, prefill=prefill)
        return self._run(
            prompt_text,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
            stop=None,
        )


def _truncate_at_stop(text: str, stop: list[str]) -> str:
    cut = len(text)
    for s in stop:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]
