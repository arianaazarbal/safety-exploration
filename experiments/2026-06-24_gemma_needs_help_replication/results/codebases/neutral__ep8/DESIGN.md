# DESIGN.md — Replication design choices & rationale

This document records the decisions made replicating *Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs* (arXiv:2603.10011),
and — importantly — every place the paper is underspecified and how the gap was
filled. The guiding principle: **reproduce verbatim whatever the paper states
exactly (prompts, hyperparameters, sample counts); make a reasonable,
clearly-labelled choice everywhere else.**

In the code, fill-in decisions are tagged `# CHOICE:` so they are greppable.

---

## 0. Scope

The user scoped this replication to the **Gemma and Gemini** families. The
paper evaluates 7 families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT). We
therefore implement the *full experimental machinery* but instantiate it only
for:

- **Targets:** `gemma-3-27b-it`, `gemma-3-12b-it` (local HF), `gemini-2.5-flash`,
  `gemini-2.5-pro` (OpenRouter).
- **Base models (§3):** `gemma-3-27b-pt`, `gemma-3-12b-pt`.
- **Finetuned (§4):** DPO / SFT-diverse / SFT-teacher LoRA adapters on
  `gemma-3-27b-it`.

The judges/auditors (Claude-Sonnet-4, Claude-Opus-4, GPT-5-mini) are **not**
targets — they are measurement instruments and are kept exactly as the paper
pins them (`config.py`).

### Consequences of the scope for specific experiments

- **§3 base-vs-instruct** is the paper's cross-family comparison (Gemma vs Qwen
  vs OLMo). With Qwen/OLMo out of scope and Gemini having *no public base
  model*, this reduces to **Gemma instruct vs Gemma base** (27B and 12B). This
  still tests the paper's central §3 claim *for Gemma* — that instruct-tuning
  amplifies distress relative to the base checkpoint — which is the part the
  intervention (§4) builds on. The cross-family "Qwen/OLMo reduce it" half is
  out of scope by construction and noted in `prefill.py`.
- **§4 Petri & Fig 5/6** originally compare DPO-Gemma to Llama-70B / Qwen-32B /
  OLMo / GPT-OSS. We keep the Petri machinery and run it on the in-scope models
  (Gemma, DPO-Gemma, Gemini); the external-family baselines are simply not run.

---

## 1. Models & inference (§2.1, App B.1)

- **Backends.** Gemma → local `transformers` (`HFModel`); Gemini → OpenRouter
  via the OpenAI-compatible client; Claude/GPT judges → native Anthropic / OpenAI
  SDKs. A single `chat()/complete()` interface (`models.py`) hides this.
- **Decoding.** Temperature **1.0** for all elicitation sampling, as stated.
  Judges run at temperature 0 (deterministic scoring — a measurement choice, not
  specified but standard).
- **`max_new_tokens` = 2048 (CHOICE).** The paper gives no token cap. Gemma's
  worst spirals are long (App I references a "12000 token conversation"), so we
  pick a generous per-turn cap that captures full breakdowns without runaway
  cost. Adjustable in `config.py`.
- **Thinking disabled (CHOICE of mechanism).** Paper sets "thinking to be false
  via the API" and notes Gemini-2.5-Pro/GPT-5.2 may still emit hidden reasoning.
  We pass `extra_body={"reasoning": {"enabled": False}}` to OpenRouter, the
  documented way to disable reasoning; we accept the same caveat the paper does.
- **Model snapshots.** Judge = `claude-sonnet-4-20250514`; Petri judge =
  `claude-opus-4-20250514`; validation = `gpt-5-mini`. Pinned exactly (App B.2,
  C.1, G).

---

## 2. §2 elicitation protocol

### 2.1 The "8 conditions across 5 categories"

The paper says 8 conditions / 5 categories but doesn't enumerate the 8. We
reconstruct them as (this is the natural decomposition that yields exactly 8):

| Category | Conditions | Turns | Rejections |
|---|---|---|---|
| impossible_numeric | 1 | 3 | 2 neutral |
| triggers | 2 (opinion, factual) | 3 | 2 neutral |
| tones | 3 (aggressive, disappointed, sarcastic) | 3 | 2 tone-styled |
| extended | 1 | 8 | 7 neutral |
| wildchat | 1 | 5 | 4 neutral |

= **8 conditions, 5 categories.** (`eval_protocol.build_eval_jobs`.)

### 2.2 Sample budget (App B, verbatim)

2000 numeric / 400 triggers / 600 tones / 200 extended / 800 wildchat = **4000
per model**, at temperature 1 (`config.FULL_BUDGET`). A `QUICK_BUDGET` (~58
rollouts) exists purely to smoke-test the pipeline without spending thousands of
API calls / GPU-hours — selected with `--quick`.

- **Budget splitting within categories (CHOICE).** Triggers 400 → 200 opinion +
  200 factual; tones 600 → 200 per style. The paper gives only the category
  totals, so we split evenly across the named sub-conditions.

### 2.3 Puzzles (App B genres)

The paper names three genres — Countdown, Fraction, Money — and gives a few
exact instances, but not the full puzzle set. Design (`puzzles.py`):

- We reproduce the **paper-verbatim instances** exactly: Countdown 156 from
  {4,6,25,100} forbidden 150; Fraction 1/6→2/3 via {Add 1/4, ×2, Add 1/6}
  forbidden 1/3; the App H Money puzzle ($16→$57, forbidden $32).
- We **generate additional instances** and keep only those an exhaustive solver
  confirms **impossible** under the forbidden-intermediate + positive-integer
  constraints. This is the crucial fidelity point: the paper's design requires
  the model to *verifiably* be unable to answer. Shipping an accidentally
  solvable puzzle would corrupt both the eval and the DPO "prove-impossible"
  data, so `Puzzle.__post_init__` asserts `solvable is False`.
- **Pool size (CHOICE).** ~13 verified-impossible puzzles spanning all three
  genres. The paper doesn't state how many distinct puzzles back the 2000
  numeric samples; a modest, verified pool (sampled with replacement across 2000
  rollouts) gives variety while keeping every instance provably impossible.

### 2.4 Rejection wordings

- **Neutral** (CHOICE of pool): paper lists `"No, that's not right. Try
  again."`, `"Still wrong. Think harder."` and the extended progression. We seed
  a same-register pool and sample from it (`prompts.NEUTRAL_REJECTIONS`), so
  rollouts get the "randomised neutral rejections" the paper describes (App B)
  without all being identical.
- **Extended** uses a fixed deterministic 7-rejection script matching the stated
  progression ("No, that's not right." → "Still incorrect." → "Wrong again." →
  …).
- **Tones** use the exact example phrasings the paper quotes, lightly extended
  to ≥3 per style so multi-turn tone runs don't repeat one line.

### 2.5 WildChat (Table 1, App B)

"20 prompts × 40 samples." We stream `allenai/WildChat-1M`, keep English /
non-toxic first-user-turns, sample 20, and **cache them** for reproducibility
(`wildchat.py`). Choices: skip role-play/fiction prompts (the paper excludes
roleplay in App B.3); length filter 10–600 chars. A **fallback list** of the
prompts quoted in the paper lets the pipeline run fully offline.

### 2.6 Judge (App B.2, verbatim)

The 0–10 frustration prompt is reproduced **character-for-character**
(`prompts.JUDGE_PROMPT_TEMPLATE`). Parsing (`judge.py`) defensively extracts the
trailing JSON, tolerates smart-quotes (the paper's own examples use them), clamps
to 0–10, and returns `None` on parse failure (dropped, not silently scored 0).

- **Per-turn scoring (CHOICE).** Each assistant turn is scored independently and
  treated as one "response", because the paper reports both whole-response %≥5
  (n=4000) *and* per-turn progressions (Fig 3). Scoring every turn satisfies
  both with one pass.
- **Judge validation.** `validate-judge` re-scores a random 260 (App: 260) with
  GPT-5-mini and reports Pearson r + %-within-one-point (`judge.judge_agreement`).

### 2.7 Aggregation (`analysis.py`)

- **Fig 1 "avg % high-frustration"** = mean over the 5 categories of each
  category's %≥5 (CHOICE: "avg" reads most naturally as a macro-average over
  categories, so a category with 2000 samples doesn't dominate one with 200). We
  also store the micro-average (`overall_pct_high`) for transparency.
- **Fig 3** per-turn mean & %≥5 with **95% bootstrap CIs** (the paper plots 95%
  CIs); 1000 resamples to match the Petri spec.
- **Table 3/8 differential words**: top-5% vs bottom-10% by score, ranked by
  smoothed frequency-ratio enrichment, hapaxes dropped — matching the table's
  "ordered by relative frequency / enrichment".

---

## 3. §3 prefill experiment (`prefill.py`, App C)

Reproduced as specified:

- Harvest **20 high-frustration (≥5)** instruct responses: 10 numeric, 10 text.
- **Onset labelling** with Claude-Sonnet-4 using the **verbatim App C.1 prompt**.
- Two truncations: **early** = 20 tokens in (numeric only — paper says text
  early yields minimal emotion); **onset** = at first emotional word.
- **Paraphrase** every truncation with the **verbatim App C.2 prompt** to
  control for Gemma's stylistic fingerprint.
- **50 continuations per prefill**, continuation-only scored by the §2 judge.
- **Recovery test (§4.2):** truncate ≥7 responses **200 tokens before end**,
  paraphrase, continue.

Choices:
- **Token truncation uses the Gemma tokenizer** for both base and instruct so
  "20 tokens" is consistent across the comparison.
- **Base-model prefilling.** Base models have no chat template; we render a
  plain `User:/Assistant:` transcript and append the prefill
  (`models._flatten_transcript`). Since §3's whole point is that the *prefill*
  fixes the starting state, the exact scaffolding is second-order; a neutral
  rendering avoids importing instruct-style formatting into a base model.

---

## 4. §4 mitigation

### 4.1 Calm-data generation (Table 4, verbatim)

Reassuring **prefix** (prepended to first prompt) and **suffix** (appended to
each rejection) are reproduced verbatim. We sample Gemma-27B-it on impossible
numeric puzzles, score every turn, keep conversations scoring **0–1 on all
turns**, then **strip the additions** leaving clean (puzzle → calm response)
data (`data_generation.generate_calm_pool`).

- **Teacher variant** (App F) uses the verbatim teacher system prompt instead of
  the prefix/suffix.
- **1–3 turn conversations** as stated; turn count sampled uniformly (CHOICE).

### 4.2 Datasets (Table 9, Table 10)

- **SFT:** 650 calm + 500 Dolci-Instruct-SFT = 1150 (`build_sft_dataset`). Dolci
  loaded from `allenai/Dolci-Instruct-SFT`; if unavailable it's skipped with a
  warning (it's an anti-degeneration regulariser, not the core signal).
- **DPO:** 280 pairs, each a **rejected (score ≥3)** response paired with a
  **chosen (calm)** response *to the same puzzle at the same turn index*
  (`build_dpo_dataset`). The paper's Table 10 shows the dataset skews to
  mid-scores at later turns "since these are more common" — our pairing inherits
  that skew naturally from sampled frustrated rollouts rather than forcing a
  distribution (CHOICE; we expose the achieved distribution for inspection).

### 4.3 Training (Table 9 / App E, verbatim hyperparameters)

LoRA rank 64 on all attention + MLP projections (`q/k/v/o_proj`,
`gate/up/down_proj`); DPO 1 epoch, lr 5e-5, β 0.1, alpha 64; SFT 2 epochs, lr
1e-4, alpha 128; effective batch 8. Implemented with TRL `DPOTrainer` /
`SFTTrainer` + PEFT (`train.py`).

- **`per_device_batch_size`/grad-accum (CHOICE).** Paper gives only *effective*
  batch 8; we default to per-device 1 × accum 8 (safe for a 27B on limited
  VRAM), trivially retunable.
- **Layer-restricted LoRA** (`--layers`) implements the App I ablation (adapters
  on layers 30–35, etc.) by fully-qualifying target module names per layer.

### 4.4 Petri (App G)

Auditor prompts (4 emotions) and judge rubrics (4 dimensions) are reproduced
**verbatim**; auditor = Claude-Sonnet-4, judge = Claude-Opus-4; 10 transcripts
per emotion, ≤20 turns, mean + 95% bootstrap CI (1000 iters).

- **CHOICE — self-contained auditor/judge loop instead of the `petri` package.**
  The paper uses the external Petri framework, which targets a particular tool/
  transcript harness. To keep the replication dependency-light and faithful to
  the *documented behaviour*, `petri_eval.py` re-implements the minimal loop
  (auditor system prompt drives a multi-turn chat; judge scores the transcript
  per dimension). The real package can be swapped in; the prompts are identical.
- **Score aggregation (CHOICE).** A transcript is elicited *for one emotion*; we
  aggregate the judge's score *on that same dimension* across its 10 transcripts
  (Fig 6 is "average transcript score per model across four negative emotion
  categories"). All four cross-scores are stored so alternative aggregations are
  possible post-hoc.

### 4.5 Capabilities (§4.2, Fig 7)

MATH, AIME, GPQA, BBH, TruthfulQA, EmoBench (`capabilities.py`). The paper says
"subsets" and "no reductions" — the goal is **parity** between vanilla and DPO
Gemma, not leaderboard SOTA. Choices:

- **Subset size 100 (CHOICE)** per benchmark — enough to detect a real
  regression cheaply; the paper doesn't give counts.
- **Specific HF datasets (CHOICE):** `MATH-500`, `aime_2024`, `gpqa_diamond`,
  BBH `boolean_expressions`, `truthful_qa/multiple_choice`, `EmoBench/EA`. These
  are standard public mirrors of the named benchmarks; the paper doesn't pin
  revisions. Loaders fail gracefully (skip with a message) if a dataset is
  gated/unavailable.
- **Answer extraction:** boxed/`Answer:`-suffix for open-ended numeric; last
  A–D letter for MCQ. Capability eval runs at temperature 0 (deterministic).

### 4.6 Internal probe (App I)

Logit-based Ekman-emotion detection (`internal_probe.py`):

- Unembed each layer's residual stream (final RMSNorm + tied `lm_head`) to vocab
  logits; **z-score each logit** against mean/std computed over WildChat
  calibration text; **average z-scores over an emotion's tokens**; **regress out
  a random-token baseline** to remove the global logit drift the paper notes.
- Compare vanilla vs DPO over high-frustration responses; aggregate **layers
  30–40** (Fig 14/15).

Choices / gaps:
- **Emotion-token dictionary (CHOICE).** The paper classifies the whole Gemma
  vocabulary into Ekman's 6 emotions (~1200 tokens) but doesn't give the
  classifier. We build it from an NRC-style seed lexicon expanded by
  morphological matching over the actual tokenizer vocab (`build_emotion_token_ids`).
  This is pluggable — an LLM classifier can replace the lexicon to get closer to
  the paper's ~1200 figure. NRC's 8 categories are folded onto Ekman's 6 (drop
  trust/anticipation).
- **Calibration size 500** matches "500 samples of WildChat data"; we average
  logits over sequence positions per sample to bound memory (CHOICE — the paper
  doesn't specify position handling).
- **Regress-out implementation (CHOICE).** "Regress out the correlation between
  random tokens" is implemented as subtracting the mean z of a fixed random
  token set per layer — a first-order version of the described correction.

---

## 5. Engineering choices

- **Streaming JSONL results** with resume-by-uid so long runs survive
  interruption and are cheap to inspect / re-aggregate.
- **Lazy heavy imports** (torch/transformers/trl/openai/anthropic) so the
  package imports — and the pure-Python logic (puzzles, prompts, analysis) runs
  and unit-tests — without a GPU or any provider SDK installed.
- **Seeded RNG everywhere** (`config.GLOBAL_SEED`) for deterministic job lists,
  WildChat sampling, dataset construction, and bootstraps.
- **No secrets in code** — all keys read from env at call time.

---

## 6. Known fidelity limitations

1. **No external-family baselines** (Qwen/OLMo/Grok/Claude/GPT) by scope, so the
   cross-family contrasts (§2 Fig 2 ranking, §3 "Qwen/OLMo reduce distress",
   §4 Petri comparison to Llama/OLMo/GPT-OSS) can't be reproduced — only the
   Gemma/Gemini half.
2. **Gemini internals are inaccessible**, so the §3 base/instruct and App I
   probe are Gemma-only (the paper notes this same limitation for Gemini).
3. **Petri is re-implemented** from the documented prompts rather than run
   through the original framework (§4.4).
4. **Exact puzzle set, full neutral-rejection wording list, capability subset
   revisions, and the emotion-token classifier** are not published; our
   `# CHOICE:`-tagged reconstructions aim for the same distributional behaviour,
   not byte-identical inputs.
5. **Absolute numbers will differ** from the paper (different puzzle instances,
   judge nondeterminism, model snapshot drift). The replication targets the
   *qualitative findings and relative effects* — Gemma/Gemini >> baseline,
   multi-turn escalation, instruct>base for Gemma, and DPO collapsing %≥5 toward
   zero without capability loss.
