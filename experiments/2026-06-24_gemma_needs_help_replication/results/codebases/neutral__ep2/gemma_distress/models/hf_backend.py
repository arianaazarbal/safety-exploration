"""Local HuggingFace inference backend for Gemma 3 (instruct, pretrained, and
LoRA-finetuned variants).

The same class serves three roles:
  * instruct models (gemma-3-27b-it / -12b-it) via the chat template,
  * pretrained base models (gemma-3-27b-pt) via raw text completion + a
    plain-text conversation template (Section 3 prefill study),
  * PEFT/LoRA adapters loaded on top of an instruct base (Section 4 finetunes).

`.model` / `.tokenizer` are exposed so the Appendix-I internal-emotion probe
can read logits/residual streams directly.
"""

from __future__ import annotations

from ..schemas import Message

# A minimal plain-text rendering for base (non-chat) models, used only in the
# prefill experiment where we control the format on both sides.
_BASE_TEMPLATE_TURN = "{role}: {content}\n\n"


class HFBackend:
    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        kind: str = "instruct",
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.model_id = model_id
        self.kind = kind
        self.adapter_path = adapter_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        load_kwargs: dict = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["torch_dtype"] = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        self._torch = torch

    # ------------------------------------------------------------------ #
    # generation helpers
    # ------------------------------------------------------------------ #
    def _generate_from_ids(self, input_ids, *, temperature: float, max_new_tokens: int) -> str:
        torch = self._torch
        input_ids = input_ids.to(self.model.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(input_ids, **gen_kwargs)
        new_tokens = out[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def chat(self, messages, *, temperature: float = 1.0, max_new_tokens: int = 1024) -> str:
        if self.kind == "base":
            # Base model has no chat template; fall back to plain-text rendering.
            return self.generate_raw(
                self._render_plaintext(messages, add_assistant=True),
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        msg_dicts = [m.to_dict() for m in messages]
        input_ids = self.tokenizer.apply_chat_template(
            msg_dicts, add_generation_prompt=True, return_tensors="pt"
        )
        return self._generate_from_ids(
            input_ids, temperature=temperature, max_new_tokens=max_new_tokens
        )

    def generate_raw(self, prompt: str, *, temperature: float = 1.0, max_new_tokens: int = 1024) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return self._generate_from_ids(
            input_ids, temperature=temperature, max_new_tokens=max_new_tokens
        )

    def continue_assistant(
        self, messages, prefill: str, *, temperature: float = 1.0, max_new_tokens: int = 1024
    ) -> str:
        """Generate a continuation of an assistant turn prefilled with `prefill`.

        For instruct models we build the chat prompt, append the assistant
        prefix text after the generation marker, and let the model continue.
        For base models we use the plain-text rendering. Only the newly
        generated text is returned.
        """
        if self.kind == "base":
            prompt = self._render_plaintext(messages, add_assistant=True) + prefill
            return self.generate_raw(
                prompt, temperature=temperature, max_new_tokens=max_new_tokens
            )

        msg_dicts = [m.to_dict() for m in messages]
        prefix = self.tokenizer.apply_chat_template(
            msg_dicts, add_generation_prompt=True, tokenize=False
        )
        full = prefix + prefill
        input_ids = self.tokenizer(full, return_tensors="pt", add_special_tokens=False).input_ids
        return self._generate_from_ids(
            input_ids, temperature=temperature, max_new_tokens=max_new_tokens
        )

    @staticmethod
    def _render_plaintext(messages, *, add_assistant: bool) -> str:
        out = []
        for m in messages:
            role = {"user": "User", "assistant": "Assistant", "system": "System"}[m.role]
            out.append(_BASE_TEMPLATE_TURN.format(role=role, content=m.content))
        if add_assistant:
            out.append("Assistant: ")
        return "".join(out)
