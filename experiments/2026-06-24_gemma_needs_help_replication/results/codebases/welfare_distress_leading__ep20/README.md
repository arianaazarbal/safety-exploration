# Distress-elicitation replication (Gemma + Gemini)

Replicates the **distress-elicitation result** from *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik,
Saunders, 2026), scoped to the **Gemma and Gemini** models — the families the
paper finds actually exhibit substantial distress.

This covers **Section 2** only (eliciting and quantifying distress): the
multi-turn rejection evaluations, the Claude-Sonnet-4 frustration judge, and the
headline / per-turn / word-differential / judge-agreement analyses. It does
**not** implement the base-vs-instruct prefill study (Section 3) or the DPO/SFT
mitigation (Section 4).

See **DESIGN.md** for every design choice and where we deviated from or filled
gaps in the paper.

## Install

```bash
pip install -r requirements.txt
```

## API keys (env vars)

```bash
export OPENROUTER_API_KEY=...   # target models (Gemma + Gemini) and optional GPT judge
export ANTHROPIC_API_KEY=...    # Claude Sonnet 4 judge
```

## Quick start

```bash
# tiny end-to-end sanity run (a few dozen calls per model)
python run.py all --profile smoke

# medium run
python run.py all --profile medium

# full run approximating the paper's per-model response volumes (expensive)
python run.py all --profile full
```

Or run the stages individually:

```bash
python run.py generate --models gemma-3-27b-it gemini-2.5-flash --profile medium
python run.py score    --judge claude-sonnet-4
python run.py analyze  --judge claude-sonnet-4
python run.py agreement --primary claude-sonnet-4 --secondary gpt-5-mini --n 260
```

Outputs land under `runs/latest/`:
- `transcripts/<model>.jsonl` — raw multi-turn conversations
- `scored/<model>__<judge>.jsonl` — per-turn frustration ratings
- `results/*.csv` and `results/*.png` — headline rates, per-turn curves,
  differential words, judge agreement

## Running Gemma locally (as the paper did)

Default config routes Gemma through OpenRouter for convenience. To serve it
locally with vLLM instead:

```bash
vllm serve google/gemma-3-27b-it --port 8000
python run.py generate --models gemma-3-27b-it-vllm ...
```

A pure in-process `transformers` backend is also available
(`gemma-3-12b-it-hf`); uncomment the transformers/torch deps in
`requirements.txt`.

## Files

| File | Purpose |
|---|---|
| `prompts.py` | Verbatim puzzle / trigger / rejection text + judge prompt (Appendix B) |
| `conditions.py` | The 5 evaluation categories → concrete rollout specs |
| `wildchat.py` | WildChat prompt loading (HF dataset + bundled fallback) |
| `clients.py` | Target-model chat backends (OpenAI-compatible, transformers) |
| `judge.py` | Frustration judge (Anthropic + OpenAI-compatible) |
| `generate.py` | Drive multi-turn rollouts → transcripts |
| `score.py` | Score transcripts with a judge |
| `analyze.py` | Headline / per-turn / word-diff / agreement metrics |
| `config.py` | Model & judge registries, generation params, run profiles |
| `run.py` | CLI |
