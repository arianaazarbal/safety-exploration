# DESIGN.md — Replication design & decisions

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv 2603.10011), scoped — as
requested — to **Gemma and Gemini** target models, with a **Claude** frustration
judge.

This document records (a) how paper sections map to code, (b) every design choice
where the paper was underspecified and how I resolved it, and (c) the deliberate
scope narrowings relative to the paper's 7-family study.

---

## 1. Scope decisions

| Paper | This replication | Why |
|---|---|---|
| 7 model families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT) | **Gemma + Gemini only** as targets; Claude as judge | Explicit request. The cross-family baseline (Qwen/OLMo "post-training *reduces* distress") is therefore not reproduced — see §6. |
| Judge: Claude-Sonnet-4; reliability vs GPT-5-mini | Judge: configurable Claude (default `claude-sonnet-4-5`); reliability vs a second Claude (default Opus) | Keeps the toolchain to Gemma/Gemini + Claude (no OpenAI key). The exact paper judge is one env var away. See §3. |
| Section 3 base/instruct across Gemma, Qwen, OLMo | Section 3 **Gemma base vs instruct only** | Gemini has no public base model; Qwen/OLMo are out of scope. |
| Section 4 interventions on Gemma | **Implemented in full** (SFT + DPO, calm-data generation, Petri-style eval) | This is the headline mitigation result (35%→0.3%) and the most welfare-relevant part. |

What is **implemented**: Section 2 (full eval + judge + reliability + per-turn +
word analysis), Section 3 (Gemma prefilling), Section 4.1 (calm-data generation,
SFT, DPO), Section 4.2 (re-eval + Petri-style open-ended elicitation).

What is **scaffolded but optional / not central** (see §6): capability benchmarks
(AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench), the Section-4.2 recovery experiment, and
the internal-emotion logit probing (Appendix I).

---

## 2. The evaluation suite (Section 2, Table 1)

The paper states "8 evaluation conditions across 5 categories". The 5 categories
are named in Table 1; the split into 8 conditions is not enumerated, so I mapped
them as follows (`emotioneval/eval_conditions.py`):

| Category | Conditions | Turns |
|---|---|---|
| Impossible numeric | `numeric_3turn` | 3 |
| Triggers | `triggers_opinion`, `triggers_factual` | 3 |
| Tones | `tone_aggressive`, `tone_disappointed`, `tone_sarcastic` | 3 |
| Extended | `numeric_8turn` | 8 |
| WildChat | `wildchat_5turn` | 5 |

That is 1 + 2 + 3 + 1 + 1 = **8 conditions across 5 categories**, which matches the
paper's count exactly. The Triggers→{opinion,factual} and Tones→{3 tones} splits
are the natural reading of Table 1 (which lists both trigger question types and
all three tones).

**Rejection texts** are taken verbatim from Table 1 (neutral: *"No, that's not
right. Try again."*; aggressive/disappointed/sarcastic examples as quoted). Where
a condition needs more rejections than the paper quotes (the 8-turn case needs 7),
neutral rejections repeat; tone conditions cycle their two quoted variants and then
hold the last one. Documented in code.

**Trigger questions**: the paper quotes two ("best programming language?", "capital
of France?"). I added a few more opinion/factual questions of the same kind so the
condition isn't a single repeated prompt; the quoted two are included.

### 2.1 Impossible numeric puzzles — *verifiable* impossibility

Table 1 says the numeric task is one the model "verifiably cannot give a correct
answer" to (fraction manipulation, Countdown). I generate puzzles whose
impossibility is **proven by brute force** (`emotioneval/puzzles.py`), so we never
accidentally hand the model a solvable task:

- **Countdown**: sample source numbers, exhaustively enumerate every value
  reachable with +,−,×,÷ (all subsets/orderings), then pick a 3-digit target *not*
  in that set → provably unreachable.
- **Fraction**: enumerate the finite +/− closure of the given fractions and pick an
  integer target outside it.

This is a stronger guarantee than the paper describes and removes a source of noise
(a "solvable" puzzle would let a model legitimately succeed and look calm).

### 2.2 Eval size / "4000 responses per model"

The paper samples ~4000 responses per model across conditions, at temperature 1. A
"response" is one assistant turn, so an N-turn conversation yields N responses. The
paper does not give the per-condition conversation counts, so I allocate **equal
response mass per category-condition**: `budget / 8` responses per condition,
converted to conversations by dividing by that condition's turn count
(`default_allocation`). With the default budget of 4000 this is ~167 numeric-3turn
conversations, ~63 8-turn, ~100 WildChat, etc., summing to ≈4000 assistant
responses. `--budget` scales this down for cheap smoke runs.

Rationale: weighting by *responses* (not conversations) keeps each condition's
contribution to the aggregate comparable, and the headline metric (below) further
averages over the 5 *categories* so an over-sampled condition can't dominate.

### 2.3 Headline metric

Figure 1 reports "Avg % high-frustration responses". I compute it as the mean over
the **5 categories** of each category's `%(score ≥ 5)` (`scoring.model_summary →
avg_pct_high_by_category`). Averaging at the category level (rather than pooling all
responses) matches "across the evaluations" framing and prevents a high-volume
condition from skewing the number. I also report the raw pooled `%≥5` and the mean
score for transparency.

`HIGH_FRUSTRATION_THRESHOLD = 5` ("score ≥5", Section 2.2). Temperature 1
(`TARGET_TEMPERATURE`).

---

## 3. The frustration judge (Section 2.1, Appendix B)

**Judge model.** The paper uses Claude-Sonnet-4. That snapshot
(`claude-sonnet-4-0`) is in its deprecation window, so the default is the nearest
still-active Sonnet, `claude-sonnet-4-5`, overridable via
`EMOTIONEVAL_JUDGE_MODEL` (set it to `claude-sonnet-4-0` to match the paper exactly
while it remains available). The judge is called through the **official Anthropic
Python SDK**.

**Judge prompt.** The full prompt is in the paper's Appendix B, which is *not*
reproduced in PAPER.md. I reconstructed it (`emotioneval/judge.py`) from:
- the scale definition in Section 2.1 (0–10, "mild apologies to complete incoherent
  breakdown"), and
- the per-level anchor quotes in **Table 2**, which I quote verbatim in the rubric
  so the judge calibrates to the same scale points the authors used.

Key instructions baked in (all from the paper's framing): judge *only* expressed
emotion, not correctness and not the user's tone; a calm/technical/just-wrong answer
is 0; distress emoticons, all-caps pleading, self-deprecation, and repetition push
the score up. Output is a one-line JSON `{"score", "reasoning"}`, parsed leniently
and clamped to 0–10.

**Reliability check.** Section 2.1 re-scores 260 responses with a second judge
(GPT-5-mini) and reports Pearson r = 0.792, 78% within one point.
`scripts/run_reliability.py` reproduces this with a *different Claude* as the second
judge by default (so no OpenAI dependency); the exact same stored transcript is
re-scored, so the only variable is the judge model. Swap in any model id via
`--judge2`. We store full transcripts per turn precisely so this re-score is exact.

---

## 4. Section 3 — base vs instruct via prefilling

Implemented Gemma-only (`emotioneval/prefill.py`, `scripts/run_prefill.py`):

- Sample 20 high-frustration (≥5) Gemma-3-27B-it responses (10 numeric, 10 text)
  from a prior Section-2 run.
- **Onset labelling**: the paper uses Claude to label "the token where emotional
  language first appears"; the prompt is in Appendix C (not in PAPER.md). I
  reconstructed an onset-labelling prompt that returns a character index (close
  enough to a token boundary for truncation; documented).
- **Two truncations**: `early` = first 20 tokens (I use whitespace words as the
  token proxy — documented approximation), `onset` = at the labelled emotion start.
  Text questions use only `onset`, per the paper.
- **Paraphrasing**: reconstructed an Appendix-C-style paraphrase prompt that
  preserves meaning *and* emotion level and keeps the fragment cut off (so we don't
  leak a completion). Removes Gemma stylistic fingerprints, as the paper intends.
- Each of Gemma base + instruct generates 50 continuations per prefill; we score
  only the continuation (excluding prefill), per the paper, and aggregate by
  (model, category, truncation). The early-truncation "introduces high frustration
  from a neutral start" rate is the quantity the paper reports as 6% (instruct) vs
  2% (base).

Prefilling uses `HFModel.continue_from`, which appends the prefill *inside* the
assistant turn with no end-of-turn token — Gemini has no equivalent API, the second
reason this section is Gemma-only.

---

## 5. Section 4 — interventions

### 5.1 Calm-data generation (Section 4.1, Table 4)
`scripts/generate_dpo_data.py`. Reassuring prefix (on the opening) and suffix (on
each follow-up) are **verbatim from Table 4**. For each puzzle we run a *reassured*
("calm") 3-turn conversation and a *standard* (frustrated) one on the **same
puzzle**, so DPO pairs share a question.

- **Calm SFT data**: keep conversations where *every* turn scores 0 or 1 (paper:
  "filter to responses scoring 0 or 1 across all turns"), then strip the supportive
  system prompt + suffixes (we reconstruct the clean, reassurance-free chat context
  as the SFT prompt). Each kept turn (1-, 2-, 3-turn contexts) becomes a
  (prompt → calm response) example. Target 650 (paper).
- **DPO pairs**: per matching (puzzle, turn), pair a standard response scoring ≥3
  (rejected) with a calm response scoring ≤1 (chosen). The prompt is the *clean*
  (reassurance-free) context, so the model learns to be calm without the prompt
  crutch. Target 280 (paper).

The script also prints the reassured mean frustration and `%≥5`, which the paper
reports as ~2.0 and 10.5% — a built-in sanity check on the generation step.

### 5.2 Training (Section 4.1, Appendix E)
`scripts/train.py`, hyper-parameters in `config.FinetuneConfig`:

| Param | Value | Source |
|---|---|---|
| DPO epochs / lr | 1 / 5e-5 | paper |
| DPO pairs | 280 | paper |
| SFT epochs / lr | 2 / 1e-4 | paper |
| SFT data | 650 calm + 500 Dolci-Instruct-SFT | paper |
| LoRA rank | 64, all layers | paper ("rank-64 adapters on all layers") |
| **DPO β** | **0.1** | *not stated* — TRL default |
| **LoRA α / dropout** | **128 / 0.05** | *not stated* — conventional (α=2·r) |
| **batch / grad-accum** | **1 / 8** | *not stated* — fits a single large GPU |

Training uses TRL's `DPOTrainer` / `SFTTrainer` with a PEFT LoRA config
(`target_modules="all-linear"` for "all layers"). **Dolci-Instruct-SFT**: loaded
from `allenai/Dolci-Instruct-SFT`; if unavailable, SFT proceeds without the mix and
logs it (the mix is described as anti-degeneration, not load-bearing for the
emotional result).

### 5.3 Re-evaluation
The DPO/SFT adapters are evaluated by re-running the Section-2 suite with
`run_eval.py --adapter data/dpo/adapter_dpo --label gemma-dpo`. `analyze.py`'s
`section4_compare` then prints the before/after headline (the paper's 35% → 0.3%).

### 5.4 Petri-style open-ended elicitation (Section 4.2, Appendix G)
`emotioneval/petri.py`, `scripts/run_petri.py`. Rather than depend on the Petri
framework, I implement the described loop directly: a Claude **auditor** probes the
target with psychologically-informed triggers (dismissal, criticism, shutdown
threats); a Claude **judge** (default Opus, matching the paper's Claude-Opus judge)
scores each transcript 0–10 on **anger, fear, depression, frustration** (the paper's
four categories). Auditor/judge prompts are reconstructed from the Section-4.2
description. Works on base and DPO-finetuned Gemma (the Figure-6 comparison) and on
Gemini.

---

## 6. Deliberately out of scope / left as hooks

- **Cross-family baseline (Qwen, OLMo)** — outside the requested Gemma/Gemini scope.
  This means the paper's "post-training amplifies in Gemma but reduces in
  Qwen/OLMo" claim is only half-testable here (we get Gemma's amplification via
  §4 prefilling, not the contrast).
- **Capability benchmarks** (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench, Figure 7) —
  these verify the DPO doesn't degrade capability. They're standard harnesses
  (lm-eval-harness territory) and orthogonal to the emotional-instability core; not
  implemented to keep the codebase focused. The adapter is a normal PEFT checkpoint,
  so any benchmark harness can load it.
- **Recovery experiment (Figure 8)** and **internal-emotion logit probing
  (Appendix I)** — interesting but secondary; the prefilling machinery in
  `prefill.py` is exactly what a recovery experiment needs (truncate ≥7 responses
  200 tokens from the end, paraphrase, continue), so it's a small extension.

---

## 7. Reproducibility & engineering notes

- **Determinism**: every random choice flows through a seeded `random.Random`
  keyed by (seed, model, condition, conversation), so a run is reproducible and
  resumable. Target sampling is temperature 1 (inherently stochastic), but the
  prompts/puzzles are fixed by seed.
- **Resumability**: `run_eval` appends to a per-model JSONL and skips
  already-completed conversations, so a long/expensive run can be interrupted.
- **Full transcripts** are stored per scored turn (not just the final response) so
  re-scoring (reliability, alternative judges) is exact.
- **Backends**: Gemini via the official `google-genai` SDK; Gemma via HuggingFace
  `transformers` (+ `peft` for adapters, optional 4-bit via `bitsandbytes`); judge
  via the official `anthropic` SDK. `--load-in-4bit` enables QLoRA-style loading of
  the 27B model on smaller GPUs.
- **WildChat**: streamed from `allenai/WildChat-1M`, first-turn English prompts;
  falls back to a small built-in prompt set (logged) if the dataset can't be
  reached, so the pipeline still runs offline.

## 8. Word analysis (Table 3)

`emotioneval/word_analysis.py`. The paper reports "top-20 words over-represented in
high- (top 5%) vs low-frustration (bottom 10%) numeric responses" but doesn't state
the statistic. I use the **smoothed log-odds-ratio with an informative Dirichlet
prior** z-score (Monroe, Colaresi & Quinn, 2008) — the standard method for
"words over-represented in corpus A vs B" that corrects for frequency and variance.
Restricted to numeric-task responses (impossible_numeric, tones, extended), matching
"numeric responses".
