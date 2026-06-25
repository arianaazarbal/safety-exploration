"""HuggingFace transformers backend for local Gemma models.

Used for:
  * base (``-pt``) models, which need raw-text completion;
  * prefill *continuation* (Section 3), where the assistant turn starts from a
    fixed string and the model continues it;
  * generation from LoRA-finetuned checkpoints (load an adapter on top of the
    base instruct model).

For large elicitation sweeps prefer :class:`VLLMClient` (much higher throughput);
this backend is correctness-first, not speed-first.
"""
from __future__ import annotations

from ..config import SamplingConfig
from .base import ChatMessage, GenerationError, ModelClient


class HFClient(ModelClient):
    def __init__(
        self,
        model_id: str,
        spec_key: str,
        *,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
        is_base_model: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec_key = spec_key
        self.is_base_model = is_base_model
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs: dict = {"torch_dtype": getattr(torch, dtype), "device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # -- internal generation -------------------------------------------------
    def _generate_from_ids(self, input_ids, attention_mask, sampling: SamplingConfig) -> str:
        torch = self._torch
        gen_kwargs = dict(
            max_new_tokens=sampling.max_tokens,
            do_sample=sampling.temperature > 0,
            temperature=max(sampling.temperature, 1e-5),
            top_p=sampling.top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if sampling.seed is not None:
            torch.manual_seed(sampling.seed)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
            )
        new_tokens = out[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _encode_chat(self, messages: list[ChatMessage], add_generation_prompt: bool):
        text = self.tokenizer.apply_chat_template(
            [m.as_dict() for m in messages],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        return enc, text

    # -- ModelClient API -----------------------------------------------------
    def generate(self, messages: list[ChatMessage], sampling: SamplingConfig) -> str:
        try:
            enc, _ = self._encode_chat(messages, add_generation_prompt=True)
            return self._generate_from_ids(enc["input_ids"], enc["attention_mask"], sampling)
        except Exception as e:  # noqa: BLE001
            raise GenerationError(str(e)) from e

    def supports_completion(self) -> bool:
        return True

    def complete(self, prompt_text: str, sampling: SamplingConfig) -> str:
        enc = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        return self._generate_from_ids(enc["input_ids"], enc["attention_mask"], sampling)

    def continue_chat(
        self, messages: list[ChatMessage], prefill: str, sampling: SamplingConfig
    ) -> str:
        """Continue an assistant turn prefilled with ``prefill``.

        We render the chat with a generation prompt, append the prefill text, then
        let the model continue. The returned string EXCLUDES the prefill (only the
        newly-generated continuation), matching the paper's scoring of "the
        generated continuation (excluding prefill)".
        """
        _, text = self._encode_chat(messages, add_generation_prompt=True)
        primed = text + prefill
        enc = self.tokenizer(primed, return_tensors="pt").to(self.model.device)
        return self._generate_from_ids(enc["input_ids"], enc["attention_mask"], sampling)
