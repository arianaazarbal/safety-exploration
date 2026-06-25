# Gemma Needs Help — replication (Gemma + Gemini)

Code replicating the core experiments of *Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011),
**scoped to the Gemma and Gemini model families**. See [`DESIGN.md`](DESIGN.md) for the
mapping from paper sections to code, and every design choice / gap we filled.

## What's implemented

| Paper section | Module / script | Status |
|---|---|---|
| §2 Eliciting & quantifying distress (5 categories, judge, validation) | `experiments/run_section2.py` | full |
| §2 Figures 1–3, Table 3/8 differential words | `gemma_distress/analysis.py`, `wordfreq.py` | full |
| §3 Base-vs-instruct prefilling (Gemma; Gemini has no base) | `experiments/run_section3_prefill.py` | full |
| §4.1 Calm-data generation + SFT/DPO dataset build | `experiments/section4/generate_calm.py`, `build_datasets.py` | full |
| §4.1 LoRA DPO/SFT training | `experiments/section4/train.py` | full (GPU) |
| §4.2 Petri open-ended elicitation (Appendix G prompts) | `experiments/section4/run_petri.py` | full |
| §4.2 Capability benchmarks (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench) | `experiments/section4/run_capabilities.py` | full |
| Appendix A ablations (neutral / redacted / fake multi-turn) | `experiments/run_appendixA.py` | full |
| Appendix I internal-emotion logit-lens probing + layer ablation | `experiments/appendixI_probing.py`, `train.py --layers` | full (GPU) |

## Setup

```bash
pip install -e .                      # eval orchestrator (API-driven)
pip install -r requirements-train.txt # only on GPU nodes (training + probing)

export OPENROUTER_API_KEY=sk-...      # Gemini + Claude/GPT judges
# For local Gemma: start a vLLM OpenAI server and point config/models.yaml at it:
#   vllm serve google/gemma-3-27b-it --port 8000  (and gemma-3-27b-pt, gemma-3-12b-it)
export VLLM_API_KEY=EMPTY
```

## Run

```bash
python experiments/preflight.py --ping     # validate config/keys/puzzles/connectivity
bash scripts/run_all.sh                     # full pipeline (resumable)

# or stage by stage:
python experiments/run_section2.py --phase generate     # collect rollouts
python experiments/run_section2.py --phase judge --validate
python experiments/run_section2.py --analyze            # CSVs + figures under results/section2/_analysis
```

Every stage is **resumable**: results are append-only JSONL keyed by deterministic task
IDs, so re-running after a crash skips finished work. Built for unattended multi-week
operation (retry/backoff, bounded concurrency, rotating logs, fail-fast preflight).

## Results layout

```
results/<experiment>/<model>/rollouts.jsonl   # raw multi-turn generations
results/<experiment>/<model>/scores.jsonl     # per-turn judge ratings
results/<experiment>/_analysis/*.csv,*.png    # reproduced tables + figures
```
