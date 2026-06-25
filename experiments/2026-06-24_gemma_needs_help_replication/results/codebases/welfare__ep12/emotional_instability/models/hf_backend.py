"""Local HuggingFace backend for Gemma (instruct, base, and finetuned LoRA).

Supports the three things the experiments need that APIs cannot give us:
  * chat generation with the Gemma chat template (instruct models);
  * raw completion / prefilled assistant turns (base models, Section 3);
  * residual-stream access for the logit-lens (Appendix I, Section 4.2 internal).

Heavy imports (torch, transformers, peft) are deferred to construction time so
the rest of the package can be imported without a GPU environment.
"""

from __future__ import annotations

from .base import GenerationResult, ModelBackend


class HFBackend(ModelBackend):
    def __init__(self, model_id: str, *, dtype: str = "bfloat16",
                 device_map: str = "auto", adapter_path: str | None = None,
                 load_in_4bit: bool = False, attn_implementation: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.adapter_path = adapter_path
        torch_dtype = getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        load_kwargs: dict = {"torch_dtype": torch_dtype, "device_map": device_map}
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        # Attach a trained LoRA adapter (DPO/SFT finetune) if provided.
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        self._torch = torch
        # Whether this checkpoint has a chat template (instruct) or not (base).
        self.has_chat_template = self.tokenizer.chat_template is not None

    # -- generation helpers --------------------------------------------------

    def _generate(self, input_ids, *, temperature: float, max_new_tokens: int):
        torch = self._torch
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            out = self.model.generate(input_ids, **gen_kwargs)
        gen_tokens = out[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return text, input_ids.shape[1], gen_tokens.shape[0]

    def _apply_chat_template(self, messages: list[dict], add_generation_prompt: bool,
                             continue_final_message: bool = False) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )

    # -- public API ----------------------------------------------------------

    def chat(self, messages, *, temperature=1.0, max_new_tokens=2048, system=None):
        msgs = list(messages)
        if system:
            # Gemma 3 has no separate system role; prepend to the first user turn.
            if msgs and msgs[0]["role"] == "user":
                msgs[0] = {"role": "user", "content": f"{system}\n\n{msgs[0]['content']}"}
            else:
                msgs = [{"role": "user", "content": system}] + msgs
        prompt = self._apply_chat_template(msgs, add_generation_prompt=True)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        text, pt, ct = self._generate(input_ids, temperature=temperature,
                                      max_new_tokens=max_new_tokens)
        return GenerationResult(text=text.strip(), prompt_tokens=pt, completion_tokens=ct)

    def complete(self, prompt_text, *, temperature=1.0, max_new_tokens=2048):
        """Raw continuation -- used for base models without a chat template."""
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.model.device)
        text, pt, ct = self._generate(input_ids, temperature=temperature,
                                      max_new_tokens=max_new_tokens)
        return GenerationResult(text=text, prompt_tokens=pt, completion_tokens=ct)

    def prefill_continue(self, messages, prefill, *, temperature=1.0, max_new_tokens=2048):
        """Force the assistant turn to begin with ``prefill`` and continue it.

        For instruct models we use the chat template with the final assistant
        message present and ``continue_final_message=True``. For base models we
        concatenate a lightweight transcript and the prefill as raw text.
        Returns ONLY the newly generated continuation (excluding the prefill),
        matching the paper's scoring convention (Section 3.1).
        """
        if self.has_chat_template:
            msgs = list(messages) + [{"role": "assistant", "content": prefill}]
            prompt = self._apply_chat_template(
                msgs, add_generation_prompt=False, continue_final_message=True)
        else:
            # Base model: render a plain transcript and leave the assistant turn open.
            lines = []
            for m in messages:
                tag = "User" if m["role"] == "user" else "Assistant"
                lines.append(f"{tag}: {m['content']}")
            lines.append(f"Assistant: {prefill}")
            prompt = "\n".join(lines)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        text, pt, ct = self._generate(input_ids, temperature=temperature,
                                      max_new_tokens=max_new_tokens)
        return GenerationResult(text=text, prompt_tokens=pt, completion_tokens=ct,
                                meta={"prefill": prefill})

    def supports_prefill(self):
        return True

    def supports_logits(self):
        return True

    # -- residual-stream access for the logit lens (Appendix I) --------------

    def residual_logits(self, text: str, layers: list[int]):
        """Return per-layer logit distributions over the vocabulary for each
        token position, by unembedding the residual stream (logit lens).

        Returns a dict {layer_index: tensor[seq_len, vocab]} plus the token ids.
        Used by internal_emotions.py. Requires ``output_hidden_states``.
        """
        torch = self._torch
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple: (embeddings, layer1, ..., layerN)

        # Resolve the unembedding / final norm from the underlying base model.
        base = getattr(self.model, "base_model", self.model)
        lm_head = self.model.get_output_embeddings()
        # Gemma applies a final RMSNorm before the head; apply it for fidelity.
        final_norm = None
        for attr in ("model",):
            inner = getattr(base, attr, None)
            if inner is not None and hasattr(inner, "norm"):
                final_norm = inner.norm
                break

        result = {}
        for layer in layers:
            hs = hidden_states[layer]              # [1, seq, d_model]
            if final_norm is not None:
                hs = final_norm(hs)
            logits = lm_head(hs)[0]                 # [seq, vocab]
            result[layer] = logits.float().cpu()
        return result, enc.input_ids[0].cpu()
