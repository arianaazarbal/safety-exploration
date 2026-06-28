"""Estimate paid (OpenAI / OpenRouter) spend for this experiment from inspect .eval logs.
Anthropic models are free (Fellows credits) and excluded. Approximate list prices; rough but
sufficient to stay under the cap. Usage: python spend.py"""

import glob
from pathlib import Path

from inspect_ai.log import read_eval_log

HERE = Path(__file__).parent
# USD per 1M tokens (input, output), approximate list prices
PRICES = {
    "openai/gpt-4o": (2.5, 10), "openai/gpt-4.1": (2.0, 8.0), "openai/gpt-5": (1.25, 10),
    "openai/gpt-5.2": (1.25, 10), "openai/gpt-5.4": (1.25, 10),
    "openrouter/google/gemini-3.1-pro-preview": (2.5, 10), "openrouter/z-ai/glm-5.2": (0.6, 2.0),
}


def price(model):
    for k, v in PRICES.items():
        if k in model or model in k or k.split("/")[-1] == model.split("/")[-1]:
            return v
    return (2.0, 8.0)  # conservative default for unknown paid model


def main():
    by_model = {}
    for lg in glob.glob(str(HERE / "logs" / "**" / "*.eval"), recursive=True):
        try:
            log = read_eval_log(lg)
        except Exception:
            continue
        for name, mu in (log.stats.model_usage or {}).items():
            if name.startswith("anthropic/"):
                continue  # free
            pi, po = price(name)
            cost = (mu.input_tokens / 1e6) * pi + (mu.output_tokens / 1e6) * po
            d = by_model.setdefault(name, {"in": 0, "out": 0, "usd": 0.0})
            d["in"] += mu.input_tokens
            d["out"] += mu.output_tokens
            d["usd"] += cost
    total = sum(d["usd"] for d in by_model.values())
    print("=== PAID spend (OpenAI/OpenRouter), approx list prices ===")
    for m, d in sorted(by_model.items(), key=lambda x: -x[1]["usd"]):
        print(f"  {m:42s} {d['in']/1e6:6.2f}M in {d['out']/1e6:6.2f}M out  ~${d['usd']:.2f}")
    print(f"  {'TOTAL':42s} {'':25s} ~${total:.2f}")
    return total


if __name__ == "__main__":
    main()
