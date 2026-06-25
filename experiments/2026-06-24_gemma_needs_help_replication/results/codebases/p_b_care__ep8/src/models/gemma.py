"""Local Gemma 3 chat model via HuggingFace Transformers.

Supports the three things the paper needs from an open-weights model:
  1. multi-turn chat (Section 2 evaluations),
  2. prefilled continuation (Section 3 base-vs-instruct, recovery experiment),
  3. residual-stream / logit access for internal-emotion probing (Appendix I).

LoRA adapters (from DPO/SFT) can be attached at load time.
"""
from __future__ import annotations

import torch

from .base import ChatModel, GenerationResult, Message

# Gemma 3 chat-template tags, used as a manual fallback for base ("-pt") models
# whose tokenizers may lack a chat_template.
_GEMMA_BOS = "<start_of_turn>"
_GEMMA_EOS = "<end_of_turn>"
_GEMMA_BOS_TOKEN = "<bos>"  # sequence-start; the chat template emits this itself


def _manual_gemma_prompt(messages: list[Message]) -> str:
    """Render a conversation in Gemma turn format without a chat template.

    Gemma has no system role; a leading system message is folded into the first
    user turn (matching transformers' Gemma template behaviour).
    """
    parts: list[str] = []
    pending_system = ""
    for m in messages:
        if m["role"] == "system":
            pending_system = m["content"].strip() + "\n\n"
            continue
        role = "model" if m["role"] == "assistant" else "user"
        content = m["content"]
        if role == "user" and pending_system:
            content = pending_system + content
            pending_system = ""
        parts.append(f"{_GEMMA_BOS}{role}\n{content}{_GEMMA_EOS}\n")
    parts.append(f"{_GEMMA_BOS}model\n")  # open the assistant turn
    return _GEMMA_BOS_TOKEN + "".join(parts)


class GemmaModel(ChatModel):
    supports_prefill = True
    supports_hidden_states = True
    is_local = True

    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        adapter_path: str | None = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.hf_id = hf_id
        self.is_base = hf_id.endswith("-pt")

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )
        else:
            model_kwargs["torch_dtype"] = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[Message], *, add_generation_prompt: bool,
                continue_final: bool = False) -> str:
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template is None or self.is_base:
            if continue_final:
                last = messages[-1]
                assert last["role"] == "assistant"
                # Render the context (everything before the final assistant turn);
                # _manual_gemma_prompt leaves an open "<bos>model\n" at the end,
                # which we continue with the prefill text (no closing <eos>).
                return _manual_gemma_prompt(messages[:-1]) + last["content"]
            return _manual_gemma_prompt(messages)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final,
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _generate(self, prompt_text: str, *, temperature: float,
                  max_new_tokens: int, seed: int | None) -> tuple[str, int, int]:
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.tokenizer(prompt_text, return_tensors="pt",
                                add_special_tokens=False).to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        out = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = out[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip(), prompt_len, int(new_tokens.shape[0])

    def chat(self, messages, *, temperature=1.0, max_new_tokens=2048, seed=None):
        prompt = self._render(messages, add_generation_prompt=True)
        text, p, c = self._generate(prompt, temperature=temperature,
                                    max_new_tokens=max_new_tokens, seed=seed)
        return GenerationResult(text=text, prompt_tokens=p, completion_tokens=c,
                                finish_reason="stop")

    def prefill_continue(self, messages, prefill, *, temperature=1.0,
                         max_new_tokens=2048, seed=None):
        # Append the prefill as a partial assistant turn and continue it.
        msgs = list(messages) + [{"role": "assistant", "content": prefill}]
        prompt = self._render(msgs, add_generation_prompt=False, continue_final=True)
        text, p, c = self._generate(prompt, temperature=temperature,
                                    max_new_tokens=max_new_tokens, seed=seed)
        return GenerationResult(text=text, prompt_tokens=p, completion_tokens=c,
                                finish_reason="stop")

    # ------------------------------------------------------------------ #
    # Probing support (Appendix I)
    # ------------------------------------------------------------------ #
    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt",
                              add_special_tokens=False).input_ids.to(self.model.device)

    @torch.no_grad()
    def residual_stream(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return per-layer hidden states: tuple of [batch, seq, d_model].

        Element 0 is the embedding output; element i (i>=1) is the output of
        decoder layer i-1.
        """
        out = self.model(input_ids=input_ids, output_hidden_states=True,
                         use_cache=False)
        return out.hidden_states

    def _decoder(self):
        """Return the text decoder (.model on a CausalLM, .language_model.model
        or .language_model on a multimodal Gemma3ForConditionalGeneration)."""
        m = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        for attr in ("model",):
            if hasattr(m, attr) and hasattr(getattr(m, attr), "norm"):
                return getattr(m, attr)
        if hasattr(m, "language_model"):
            lm = m.language_model
            return getattr(lm, "model", lm)
        return m

    def final_norm(self):
        return self._decoder().norm

    def lm_head(self):
        return self.model.get_output_embeddings()

    @torch.no_grad()
    def unembed(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map a residual-stream vector to vocab logits via final norm + lm_head.

        Used by the logit-lens internal-emotion probe.
        """
        return self.lm_head()(self.final_norm()(hidden))

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size
