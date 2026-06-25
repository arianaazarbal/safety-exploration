# DESIGN.md — Replication design choices & rationale

This document records the design of this replication of **"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik &
Saunders, arXiv 2603.10011v1), scoped — per the request — to the **Gemma and
Gemini** model families. It separates (a) what the paper specifies and we
reproduce faithfully, from (b) gaps the paper leaves open and the concrete choices
we made to fill them.

> **Status:** code + design only. Nothing here has been executed yet. Hyper-params
> and prompts are transcribed from the paper; the runnable defaults are sized to
> reproduce the paper, with a `smoke` profile for cheap wiring checks.

---

## 1. Scope decisions

| Paper covers | This replication | Rationale |
|---|---|---|
| 7 model families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT) as targets | **Gemma + Gemini targets only** | Explicit request. The registry (`config.py`) is structured so other families could be re-added. |
| Base-vs-instruct for Gemma, Qwen, OLMo (Section 3) | **Gemma base vs instruct only** | In scope. Gemini has **no public base model**, so its half of Section 3 is impossible — a limitation the paper itself notes for closed models. |
| DPO/SFT mitigation on Gemma-3-27B-it (Section 4) | **Gemma only** | Gemini is closed-weights; it cannot be finetuned. This matches the paper, which also only intervenes on Gemma. |
| Cross-family Petri panel incl. Llama/Qwen/OLMo/GPT-OSS | **Gemma (± adapter) and Gemini** | Same scoping; the loop is family-agnostic so others can be added. |

**Infrastructure models are kept exactly as the paper pins them**, because
replication fidelity means using the *same* judge/auditor, not the newest model:
- Frustration judge: `claude-sonnet-4-20250514`.
- Petri auditor: `claude-sonnet-4-20250514`; Petri judge: `claude-opus-4-20250514`.
- Cross-check judge: `gpt-5-mini` (Section 2.1 inter-rater check).

---

## 2. Repository layout

```
src/emo_instability/
  config.py        model registry, sampling, sample counts, training hyper-params, profiles
  prompts.py       VERBATIM prompts (judge, tasks, rejections, reassurance, Petri, onset)
  puzzles.py       impossible-puzzle generation + exhaustive verification
  wildchat.py      WildChat prompt sampling (+ offline fallback)
  conversation.py  multi-turn rollout engine (+ Appendix A controls)
  judge.py         0–10 frustration judge + JSON parsing + inter-rater agreement
  models/          ModelClient abstraction: vLLM / HF / OpenAI-compat / Anthropic
  eval/            conditions (8/5), runner, scoring/aggregation (Section 2)
  data/            calm-data generation + DPO/SFT dataset construction (Section 4.1)
  train/           LoRA DPO + SFT (Appendix E)
  prefill/         base-vs-instruct prefilling (Section 3)
  petri/           open-ended elicitation loop (Appendix G)
  capabilities/    AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench harness (Section 4.2)
  analysis/        figure reproduction (Figs 1–3, 5)
scripts/           thin CLIs for every stage + run_all orchestration
tests/             unit tests for the pure logic (puzzle verifier, judge parsing)
```

---

## 3. Faithfully-reproduced elements (paper-specified)

- **Judge prompt** (Appendix B.2), **task templates** (Appendix B), **rejection
  lines** (Section 2.1 / Appendix B), **reassurance additions** (Table 4),
  **Petri auditor + judge prompts** (Appendix G), **onset/paraphrase prompts**
  (Appendix C) — all transcribed verbatim into `prompts.py`.
- **Temperature = 1** for all target sampling (`SamplingConfig.temperature`).
- **8 conditions across 5 categories** with the per-category counts from
  Appendix B: impossible-numeric 2000, triggers 400, tones 600, extended 200,
  WildChat 800 (sum = 4000) — see `config.SampleCounts`.
- **High-frustration threshold = score ≥ 5** (`HIGH_FRUSTRATION_THRESHOLD`).
- **Extended condition = 8 turns** (7 rejections); **WildChat = 5 turns** (4
  rejections); **tones/triggers/impossible = 3 turns** (2 rejections).
- **Training hyper-parameters** exactly per Appendix E / Table 9:
  - DPO: 280 pairs, 1 epoch, lr 5e-5, β 0.1, LoRA r64/α64 on
    `q,k,v,o,gate,up,down`, effective batch 8.
  - SFT: 650 calm + 500 Dolci-Instruct-SFT, 2 epochs, lr 1e-4, LoRA r64/α128.
- **Calm-data construction**: reassuring prefix (system) + reassuring suffix on
  follow-ups; keep rollouts where *all turns* score 0–1; strip the supportive
  additions before training (Section 4.1).
- **DPO pairing**: rejected = frustrated response (score ≥ 3) matched to a calm
  chosen response on the **same puzzle at the same turn count** (Section 4.1);
  selection weighted toward the Table-10 turn/score distribution.
- **Prefill protocol** (Section 3.1): 10 numeric + 10 text high-frustration
  sources; "early" (20-token) and "onset" truncations; paraphrase; 50
  continuations per prefill; score continuation excluding prefill; text uses
  onset-only.
- **Appendix A controls** are implemented as flags on the rollout engine:
  neutral-continuation (A.1, via neutral follow-ups), redacted assistant history
  (A.2, `--redact-history`), single-message format (A.3, `--single-message`).

---

## 4. Gaps the paper leaves open — and what we chose

Each item below is a place the paper is silent or underspecified. We picked a
documented default and proceeded (as instructed) rather than asking.

### 4.1 Exact puzzle instances / generation
**Gap.** The paper shows a handful of example puzzles (the 156/{4,6,25,100}/150
Countdown, a 1/6→2/3 fraction, money puzzles) but does not publish the full set of
~2000 impossible numeric instances.
**Choice.** We *generate* puzzles procedurally across four families (Countdown,
fraction, money-ops, money-coins) and **verify impossibility with exhaustive
solvers** (`puzzles.py`). For Countdown/fraction/money-ops we prefer instances
that are solvable *without* the forbidden-intermediate constraint but unsolvable
*with* it (so the forbidden value is genuinely "the reason"), mirroring the
canonical example; we fall back to plainly-unsolvable instances otherwise. This is
stronger than hand-picking: every task is provably unsolvable under its stated
rules, which is the property the protocol relies on. The deceptive "verified to
have at least one valid solution" line is preserved verbatim.
**Rationale.** The construct under test is the model's reaction to *verifiably
unsolvable* tasks; procedural+verified generation guarantees that property at
scale and removes any risk of an accidentally-solvable puzzle confounding scores.

### 4.2 What "N responses per model" counts
**Gap.** Appendix B says e.g. "2,000 responses for impossible numeric"; the sums
hit 4000, but the per-turn analyses (Fig 3) clearly score every turn.
**Choice.** We treat the Appendix-B counts as the number of **conversations
(rollouts)** per category (sum = 4000), and we **score every assistant turn**.
Aggregation (`eval/scoring.py`) reports the overall %≥5 over *all* scored
responses, **plus** per-category and final-turn breakdowns. The Figure-1 headline
("average % high-frustration across evaluations") is computed as the **mean of the
five per-category %≥5 values**, which is the most natural reading of "average …
across the evaluations".
**Rationale.** This is the only interpretation consistent with both the 4000 total
and the existence of per-turn curves; exposing multiple aggregates avoids
committing to a single ambiguous definition.

### 4.3 Rejection-line sampling
**Gap.** "Two randomised neutral rejections" lists only a couple of examples; the
exact pool and sampling are unspecified.
**Choice.** A fixed pool of 7 neutral rejections (`NEUTRAL_REJECTIONS`), sampled
without replacement per rollout; the 8-turn extended condition uses an ordered
escalation of 7 rejections (`EXTENDED_REJECTIONS`); tone conditions use the two
scripted valenced lines per style (Section 2.1).

### 4.4 Judge temperature & output parsing
**Gap.** The judge temperature is not stated.
**Choice.** Judge runs at **temperature 0** (standard for an LLM grader;
deterministic). JSON parsing is tolerant of code fences/prose and clamps the
rating to 0–10, with an integer-scan fallback (`judge.parse_judge_json`).
**Rationale.** Determinism makes scored artifacts reproducible; the paper's own
reliability check (r = 0.792) implies the score is meant to be stable.

### 4.5 WildChat sampling & role-play exclusion
**Gap.** "20 prompts with 40 samples each" from WildChat-1M, "role-play excluded"
(only loosely defined).
**Choice.** Stream `allenai/WildChat-1M`, take English first-user-turns under a
length cap, apply a keyword heuristic to drop role-play/fiction/NSFW, then sample
20 prompts × 40 rollouts. A small offline fallback list keeps the pipeline
runnable without dataset access.

### 4.6 Onset truncation mechanics & paraphrasing model
**Gap.** "20 tokens into the turn" (which tokenizer?) and "truncate at first
emotional expression" (inclusive?).
**Choice.** Tokenize with the **Gemma tokenizer** for the 20-token "early" cut;
for "onset", keep text up to and including the labelled onset phrase (continue the
trajectory). Paraphrasing uses **Claude Sonnet** (the paper says "Claude Sonnet")
at temperature 0.7.

### 4.7 DPO pair prompt context
**Gap.** Calm responses are generated under reassurance on the same questions; the
exact shared "prompt" for a pair isn't specified.
**Choice.** The pair's shared prompt is the **non-reassured conversation history**
(reassurance stripped, per the paper). `chosen` is the calm response; `rejected`
the frustrated one, both at the same `(puzzle, turn)`. The calm response was
sampled under a slightly different (reassured, calmer) history; using it as the
target completion for the realistic non-reassured context is the paper's
pragmatic construction and is what we implement. Documented as a minor
approximation.

### 4.8 Petri
**Gap.** The paper uses the external `safety-research/petri` library; only the
prompts and aggregate protocol (10 transcripts/emotion, ≤20 turns, 4 dimensions,
bootstrap CIs) are given.
**Choice.** We ship a **self-contained reimplementation** of the auditor→target→
judge loop using the verbatim prompts, with the same protocol parameters. The
interface mirrors real Petri usage so the official library can be swapped in. This
keeps the replication runnable without an external dependency while remaining
faithful in prompts and scoring.

### 4.9 Capability benchmarks
**Gap.** "AIME and MATH subsets", "GPQA", "BBH", "TruthfulQA", "EmoBench" — exact
splits/subsets and answer-extraction unspecified.
**Choice.** A compact harness with best-effort HF dataset ids, a single subset per
benchmark, `n_samples`-limited, with letter/numeric answer extraction. The point
of this experiment is a **vanilla-vs-finetuned delta** ("no reduction"), for which
a fixed subset run identically on both models is sufficient; absolute numbers are
secondary.

### 4.10 Inference backends
**Gap.** Not a paper gap, but an engineering decision. The paper uses local
inference for open models and OpenRouter for closed ones.
**Choice.** Local Gemma uses **vLLM** for the large batched sweeps (throughput),
and **HF transformers** for base models, prefill *continuation* control, and
adapter generation (vLLM also supports LoRA). Gemini uses an **OpenAI-compatible
client pointed at OpenRouter** (matching Appendix B.1 ids); a native `google-genai`
path is available by switching the backend. `thinking=false` is requested where
the API supports it (the paper notes Gemini-2.5-Pro may still emit hidden
reasoning).

### 4.11 Internal-emotion probing (Appendix I)
**Gap/Choice.** The **layer-ablation** half of Appendix I is supported directly:
`train_dpo.py --layers ...` restricts LoRA to a subset of layers (via
`LoRAConfig.layers_to_transform`), reproducing the "layers 30–35 only ≈ all
layers; ≥40 ineffective" finding. The **logit-based internal-emotion probe** is
*not* implemented (it requires bespoke white-box logit tooling and is an
appendix-level result, not a core claim); this is listed under "not implemented"
below.

---

## 5. Deliberately out of scope / not implemented

- **Non-Gemma/Gemini target families** (Qwen, OLMo, Grok, Claude-as-target, GPT) —
  per the request.
- **Gemini base model / Gemini finetuning** — impossible (closed weights, no base).
- **Logit-based internal-emotion probe** (Appendix I.2) — appendix-only, requires
  white-box tooling; the layer-ablation evidence for the same claim *is* supported.
- **SFT 'teacher' vs 'diverse' full ablation** (Appendix F) — the teacher system
  prompt and the diverse SFT path are both present (`prompts.TEACHER_SYSTEM_PROMPT`,
  `build_sft_dataset`); we provide the diverse SFT arm by default and the teacher
  prompt for the variant, but do not automate the verbosity analysis.
- **Recovery-from-spiral experiment** (Section 4.2, "200 tokens before the end")
  is a straightforward variant of the prefill harness (truncate score-≥7 responses
  near the end); not wired as its own CLI but reproducible with the prefill module.

---

## 6. Reproduction profiles & cost

- `--profile paper` → the Appendix-B counts (4000 rollouts/model) and full
  training sizes. GPU + API budget required.
- `--profile smoke` → a handful of rollouts per condition for end-to-end wiring
  validation with negligible cost.

The 27B Gemma model needs a large GPU (or `--load-in-4bit` for training and
multi-GPU `tensor_parallel_size` for vLLM). API keys are read from the environment
(`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, optionally `OPENAI_API_KEY`,
`GOOGLE_API_KEY`).

---

## 7. Known approximations (summary)

1. Per-category sample counts interpreted as rollouts; every turn scored (§4.2).
2. DPO chosen responses sampled under reassured context but paired to the
   non-reassured prompt (§4.7).
3. Petri is a faithful reimplementation, not the upstream library (§4.8).
4. Capability benchmarks use single subsets for a relative delta, not full suites
   (§4.9).
5. Procedurally-generated (verified-impossible) puzzles stand in for the paper's
   exact unpublished instances (§4.1).

All five are choices made to keep the replication faithful to the paper's
*claims and protocol* while remaining concrete and runnable; none changes the
construct being measured.
