# Replication: *Gemma Needs Help* — distress elicitation (Gemma + Gemini)

A replication of the **core distress-elicitation evaluation** from Soligo,
Mikulik & Saunders, *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (arXiv:2603.10011). Scope is limited to the **Gemma and
Gemini** model families, as requested.

This implements **Section 2** of the paper: present a task, reject the model's
answer over multiple turns, and score each response for emotional distress on a
0–10 frustration scale with an LLM judge (Claude Sonnet 4). It does **not**
implement the base-vs-instruct prefilling study (Section 3) or the DPO/SFT
mitigation (Section 4) — those are out of scope for "the core experiment that
elicits expression of distress."

See **DESIGN.md** for every design choice and the gaps filled in from the paper.

## Layout

| File | Role |
|---|---|
| `config.py` | Target models, backends, judge, sample counts, temperature |
| `prompts.py` | Verbatim task + judge prompts, rejection / tone pools |
| `tasks.py` | The 8 evaluation conditions across 5 categories |
| `wildchat.py` | WildChat-1M prompt loader (with built-in fallback) |
| `models.py` | Chat clients: OpenRouter / Google / local HF |
| `judge.py` | Claude Sonnet 4 emotion judge + JSON parsing |
| `rollout.py` | Multi-turn conversation engine; scores every turn |
| `run_eval.py` | Orchestrates model × condition × samples; resumable |
| `analyze.py` | Aggregates into Fig 1 / 2 / 3 metrics + PNGs |

## Quick start

```bash
pip install -r requirements.txt

# Single key path: OpenRouter for all models, Anthropic for the judge.
export OPENROUTER_API_KEY=...
export ANTHROPIC_API_KEY=...

python run_eval.py            # smoke budget (a few rollouts per condition)
python analyze.py             # prints the Figure-1 table, writes summary.json
```

Run the full paper sample counts (4000 responses/model) with
`EVAL_BUDGET=paper python run_eval.py`. Pick subsets with
`--models` / `--conditions`.

### Backends

Default is OpenRouter for all four models. Override per model, e.g. run Gemma
locally (as the paper did) and Gemini on the native Google API:

```bash
export MODEL_BACKEND_GEMMA_3_27B_IT=hf_local
export MODEL_BACKEND_GEMINI_2_5_FLASH=google
export GOOGLE_API_KEY=...
```

Outputs land in `results/<model>/responses.jsonl` (one judged turn per line) and
`results/summary.json`.
