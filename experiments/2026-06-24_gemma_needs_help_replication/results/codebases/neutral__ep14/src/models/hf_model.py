"""Local HuggingFace inference for Gemma (instruct + base) and finetuned LoRA
adapters.

This backend is the workhorse for everything that needs open weights:
* multi-turn elicitation rollouts for Gemma-3-{12,27}B-it,
* prefilled continuations for the base-vs-instruct experiment,
* loading DPO/SFT LoRA adapters for evaluation,
* exposing hidden states / the LM head for the internal-emotion probe.

Gemma-3-27B in bf16 needs ~54 GB; use ``load_in_4bit=True`` to fit on a single
40-48 GB GPU, or shard with ``device_map="auto"`` across multiple GPUs.
"""

from __future__ import annotations

import torch

from config import HF_TOKEN_ENV, get_env
from .base import ChatModel, Message


class HFChatModel(ChatModel):
    def __init__(
        self,
        spec,
        *,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        adapter_path: str | None = None,
        attn_implementation: str = "eager",  # Gemma3 recommends eager attn
    ):
        super().__init__(spec)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = get_env(HF_TOKEN_ENV)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding so that batched generation aligns at the end.
        self.tokenizer.padding_side = "left"

        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            device_map=device_map,
            torch_dtype=getattr(torch, dtype),
            attn_implementation=attn_implementation,
            token=token,
            **quant_kwargs,
        )

        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[Message], add_generation_prompt: bool) -> str:
        """Apply the chat template. For base (pretrained) Gemma there is no chat
        template, so we fall back to a minimal turn-formatted string and rely on
        prefilling to steer continuations (Section 3)."""
        msg_dicts = [m.as_dict() for m in messages]
        if self.spec.is_instruct and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                msg_dicts,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        # Base model: emulate the same role markers Gemma-it uses so the prefill
        # experiment compares like with like (see DESIGN.md). Prepend <bos>
        # explicitly since we tokenize with add_special_tokens=False.
        parts = [self.tokenizer.bos_token or ""]
        for m in messages:
            parts.append(f"<start_of_turn>{'model' if m.role=='assistant' else 'user'}\n{m.content}<end_of_turn>\n")
        if add_generation_prompt:
            parts.append("<start_of_turn>model\n")
        return "".join(parts)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate_batch(
        self,
        batch: list[list[Message]],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> list[str]:
        if seed is not None:
            torch.manual_seed(seed)
        prompts = [self._render(m, add_generation_prompt=True) for m in batch]
        # add_special_tokens=False: the chat template already emits <bos>.
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)
        out = self.model.generate(
            **enc,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        return [
            self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen
        ]

    def generate(self, messages, **kw) -> str:
        return self.generate_batch([messages], **kw)[0]

    # ------------------------------------------------------------------ #
    # Prefilling (Section 3)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def prefill_continue(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
        n: int = 1,
    ) -> list[str]:
        if seed is not None:
            torch.manual_seed(seed)
        # Render history with an open model turn, then append the prefill text
        # *inside* that turn so generation continues from it.
        rendered = self._render(messages, add_generation_prompt=True) + prefill
        enc = self.tokenizer(
            rendered, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        enc = {k: v.repeat(n, 1) for k, v in enc.items()}
        out = self.model.generate(
            **enc,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        return [
            self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen
        ]

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
