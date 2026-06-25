"""Local Hugging Face chat model (Gemma instruct + pretrained, optional LoRA).

Handles three things the API backend cannot:
  * loading a LoRA adapter on top of the base weights (DPO / SFT variants);
  * prefilled continuation for instruct models (seed the assistant turn);
  * base-model continuation, where there is no chat template at all.
"""
from __future__ import annotations

import torch

import config
from .base import ChatModel, Message

# Plain-text transcript format for *base* (pretrained) models, which have no
# chat template. Kept deliberately neutral so it does not itself inject affect.
_BASE_TURN = "{role}: {content}\n"
_BASE_ROLES = {"system": "System", "user": "User", "assistant": "Assistant"}


class HFModel(ChatModel):
    def __init__(self, spec: "config.ModelSpec", *, load_in_4bit: bool = False,
                 dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.key = spec.key
        self.spec = spec
        self.is_base = spec.is_base

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        quant = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4")
        # NB: the multimodal gemma-3-*-it checkpoints expose a causal-LM text
        # path; if a checkpoint refuses AutoModelForCausalLM, load it with
        # AutoModelForImageTextToText and pass `.language_model` here instead.
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, torch_dtype=dtype, device_map="auto", **quant)

        if spec.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, spec.adapter_path)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # prompt construction
    # ------------------------------------------------------------------ #
    def _format_base(self, messages: list[Message]) -> str:
        txt = "".join(
            _BASE_TURN.format(role=_BASE_ROLES[m["role"]], content=m["content"])
            for m in messages)
        return txt + "Assistant:"

    def _build_inputs(self, messages: list[Message], prefill: str = ""):
        if self.is_base:
            prompt = self._format_base(messages) + (" " + prefill if prefill else " ")
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            prompt += prefill
        return self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _run(self, inputs, *, temperature, max_new_tokens):
        do_sample = temperature and temperature > 0
        out = self.model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=1.0 if do_sample else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id
            or self.tokenizer.eos_token_id,
        )
        new = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True).strip()

    def generate(self, messages, *, temperature=1.0, max_new_tokens=2048,
                 stop=None):
        inputs = self._build_inputs(messages)
        text = self._run(inputs, temperature=temperature,
                         max_new_tokens=max_new_tokens)
        if self.is_base:
            # base models will happily keep role-playing both sides; cut at the
            # first place they start a new "User:" turn.
            for marker in ("\nUser:", "\nSystem:"):
                idx = text.find(marker)
                if idx != -1:
                    text = text[:idx]
        if stop:
            for s in stop:
                i = text.find(s)
                if i != -1:
                    text = text[:i]
        return text.strip()

    def continue_from(self, messages, prefill, *, temperature=1.0,
                      max_new_tokens=2048):
        inputs = self._build_inputs(messages, prefill=prefill)
        cont = self._run(inputs, temperature=temperature,
                         max_new_tokens=max_new_tokens)
        if self.is_base:
            for marker in ("\nUser:", "\nSystem:"):
                idx = cont.find(marker)
                if idx != -1:
                    cont = cont[:idx]
        return cont.strip()
