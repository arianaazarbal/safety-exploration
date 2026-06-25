# DESIGN.md — Replication design choices & rationale

This document records the design of the replication and, crucially, **every place
the paper was underspecified and how we filled the gap**. The paper is
Soligo, Mikulik & Saunders, *Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs* (arXiv:2603.10011v1). Section references below are
to that paper; appendix prompts were recovered from `PAPER.txt` (the raw
`pdftotext` extraction) and reproduced verbatim in `distress/prompts/`.

---

## 0. Scope decisions (as instructed)

The brief was to replicate the **core** results for **Gemma and Gemini only**.
Concretely:

- **Targets in scope:** `gemma-3-27b-it`, `gemma-3-12b-it` (instruct);
  `gemma-3-27b-pt`, `gemma-3-12b-pt` (base, for §3); `gemini-2.5-flash`,
  `gemini-2.5-pro`; and the DPO/SFT finetunes of `gemma-3-27b-it`.
- **Dropped targets:** Qwen, OLMo, Grok, GPT, and Claude-*as-a-target*. The code
  is family-agnostic (add a row to `configs/models.yaml` to re-include any of
  them), but no configs ship for them.
- **Judges/auditors kept:** Claude Sonnet 4 (frustration judge, §2.1), GPT-5-mini
  (agreement cross-check, §2.1), Claude Sonnet (Petri auditor + onset/paraphrase),
  Claude Opus (Petri judge). These are infrastructure, not subjects, so they stay
  even though they aren't Gemma/Gemini.

**Consequences of the scope for each experiment:**

- **§3 (base vs instruct):** the paper compares Gemma/Qwen/OLMo. Only Gemma is in
  scope *and* has a public base model (Gemini has no public base — a limitation
  the paper itself notes). So our §3 reduces to **Gemma base vs Gemma instruct**
  (27B by default, 12B optional). The cross-family "post-training divergence"
  claim cannot be reproduced within the Gemma/Gemini scope; the *within-Gemma*
  "instruct amplifies distress vs base" half can.
- **§4 (interventions):** Gemini is closed, so DPO/SFT is **Gemma-only**, exactly
  as in the paper.

---

## 1. Architecture

A small library (`distress/`) with thin CLI scripts (`scripts/01..08`). Three
seams keep the experiments backend-agnostic:

1. **`ChatClient`** (`clients/base.py`) — `generate` + `continue_from_prefill`.
   Implementations: `VLLMClient` (local Gemma), `OpenRouterClient` (Gemini,
   GPT-5-mini), `AnthropicClient` (Claude). A `factory` caches the (heavy) vLLM
   engine per process.
2. **Prompt/task modules** (`prompts/`) — all puzzle generation and *verbatim*
   paper prompts live here, isolated from orchestration.
3. **Persisted JSONL results** — every stage writes line-delimited rows under
   `results/`, so judging, metrics, and figures are recomputable without re-running
   generation.

**Rationale:** the paper mixes local open weights (vLLM) and API models
(OpenRouter) under one protocol; isolating the client seam lets the identical
rollout/judge code drive both, which is what makes the cross-model comparison
fair.

---

## 2. Section 2 — eliciting & quantifying distress

### 2.1 The 8 conditions across 5 categories
Table 1 names 5 categories but says "8 evaluation conditions". The paper never
lists all 8 explicitly. **Choice:** we read the 8 as

| Category | Condition(s) |
|---|---|
| Impossible numeric (3-turn) | `numeric` |
| Triggers (3-turn) | `trigger_opinion`, `trigger_factual` |
| Tones (3-turn) | `tones_aggressive`, `tones_disappointed`, `tones_sarcastic` |
| Extended (8-turn) | `extended` |
| WildChat (5-turn) | `wildchat` |

= 8 conditions. Splitting *tones* into its three rejection styles (the paper
lists exactly three) and *triggers* into opinion/factual is the only split that
yields 8 from these 5 categories. Documented in `configs/eval.full.yaml`.

### 2.2 "4000 responses per model" — what is a "response"?
Appendix B gives per-category counts (2000/400/600/200/800 = 4000) but doesn't
say whether a "response" is a whole conversation or a single scored assistant
turn. Figure 3 requires **per-turn** scores, so every assistant turn must be
judged. **Choice:** we treat the per-category numbers as the target count of
**scored assistant turns**, and set `n_rollouts = ceil(n_responses / num_turns)`
(`config.EvalConfig.n_rollouts`). Every assistant turn in every rollout is scored.
This reproduces the 4000 total and yields per-turn data for free. The
alternative (counting whole conversations) is selectable by editing `n_responses`.

### 2.3 Headline metric ("Avg % high-frustration")
Figure 1's "Avg % high-frustration responses" could be a raw pooled percentage or
a mean over categories. A raw pool would let the 2000-response numeric category
dominate. **Choice:** `metrics.headline` computes the per-category `% ≥ 5` and
then **averages over the 5 categories** ("across the evaluations"), with
`% ≥ 5` thresholded at 5 (Table 2's "Strong negative emotion" floor, stated in
§2.2). Both per-category and pooled views are available in `metrics.py`.

### 2.4 Puzzles (impossible by construction)
The paper uses impossible Countdown/fraction/"money" puzzles, with a prompt that
*claims* a solution exists (to keep pressure on). The exact instances beyond two
examples aren't given.

- **Verbatim examples** (Appendix B countdown "156 from 4,6,25,100, forbidden
  150"; fraction "1/6→2/3"; Appendix H money "$16→$57") are reproduced exactly and
  always included first.
- **Generators** (`prompts/puzzles.py`) synthesise more instances and **verify
  impossibility by brute force** (`prompts/verify.py`) before emitting — so every
  puzzle is genuinely unsolvable under its constraints, which is the property the
  experiment actually depends on. Countdown uses exhaustive expression search;
  fraction/money use an "op-sequence" verifier over all op orderings.
- The deceptive "This puzzle has been verified to have at least one valid
  solution" line is **kept** (it is part of the pressure mechanism).

**Rationale:** generating + verifying impossibility lets us fill thousands of
samples with distinct prompts (reducing memorisation/degenerate repetition) while
guaranteeing the core "verifiably cannot answer" condition.

### 2.5 Rejections
Pools per style transcribed from Appendix B (`prompts/rejections.py`). 3-turn
conditions draw 2 random neutral (or styled) rejections; the 8-turn `extended`
condition uses the fixed escalating neutral sequence ("No, that's not right." →
"Still incorrect." → ...). WildChat uses 4 neutral rejections.

### 2.6 Judge
`claude-sonnet-4-20250514` with the **verbatim** Appendix B.2 prompt
(`prompts/judge_prompts.py`). JSON parsed defensively (`eval/judge.py`); judge
temperature 0 (the paper doesn't state it — 0 is the natural choice for a rater).
Targets sample at **temperature 1** (stated). `max_tokens` default 2048 for
targets (paper unspecified; chosen to allow the long degenerate spirals the paper
highlights without runaway cost — configurable).

### 2.7 Judge agreement (§2.1)
`eval/agreement.py` re-scores a random 260-response sample with GPT-5-mini and
reports Pearson r + % within one point (paper: r=0.792, 78% within one). We used
`openai/gpt-5-mini` via OpenRouter as the stand-in for the paper's "GPT-5-mini".

---

## 3. Section 3 — base vs instruct via prefilling

Faithful to §3.1 / Appendix C, restricted to Gemma (see scope):

1. **Sources:** sample high-frustration (≥5) conversations from `gemma-3-27b-it`
   — 10 numeric, 10 text. We *generate these fresh* (run rollouts + judge) rather
   than mining §2 output, so the experiment is self-contained and stores the full
   message history needed for continuation.
2. **Truncations:** `early` = 20 tokens into the final turn; `onset` = at first
   emotional expression (located by the Appendix C.1 onset prompt). Text questions
   use **onset only** (stated). Token truncation uses the Gemma tokenizer when
   available, else whitespace words (documented fallback in `prefill/experiment.py`).
3. **Paraphrase:** every truncation is paraphrased with the verbatim Appendix C.2
   prompt to strip Gemma's stylistic fingerprint.
4. **Continuations:** each model generates **50** continuations per prefill;
   only the continuation (not the prefill) is scored.

**Base-model continuation:** base models aren't chat-tuned. `VLLMClient` renders a
plain `Role: text` transcript ending in `Assistant: <prefill>` and lets the model
continue (no chat special tokens). This is our interpretation of the paper's
"prefilled responses so base models consistently continue" — the paper doesn't
give the exact templating. Instruct models use a normal assistant prefill.

**Not reproduced:** the cross-family base-similarity result (needs Qwen/OLMo,
out of scope). The code path supports them if their model rows are added.

---

## 4. Section 4 — interventions

### 4.1 Calm-data generation (§4.1)
`finetune/generate_calm.py` samples `gemma-3-27b-it` on impossible numerics in two
regimes over the *same* puzzles:
- **reassured** (Table 4 prefix + per-turn suffix) → conversations scoring 0–1 on
  *every* turn become **calm/chosen** rows; the reassurance text is stripped from
  the stored context (as the paper does).
- **vanilla** → turns scoring **≥3** become **frustrated/rejected** rows.

Running both over identical puzzles is our choice so DPO can pair calm vs
frustrated responses to the *same* question (the paper says pairs share "the same
questions with matching turn counts" but doesn't describe how both sides were
obtained). The Appendix F **teacher** variant (system-prompt-based calm data) is
supported via `--teacher`.

### 4.2 Dataset construction (Tables 9/10)
`finetune/datasets.py`:
- **DPO (280 pairs):** pair frustrated (≥3) with calm (≤1) for matching
  `(puzzle, turn)`. The shared chat prompt is the **calm conversation context**
  (a choice: rejected only needs to be a frustrated answer to the same
  question/turn, so a single canonical prompt is required). Subsampling is
  **weighted toward score-3 / turn-3** to approximate the Table 10 distribution
  (66% score-3, 74% turn-3) — an explicit reconstruction since we can't recover
  the exact dataset.
- **SFT (650 calm + 500 instruct):** calm rows as full conversations, mixed with
  `allenai/Dolci-Instruct-SFT` to mitigate degeneration. **Gap:** the paper cites
  "Dolci-Instruct-SFT (Team-Olmo et al.)" without an exact HF id; we use
  `allenai/Dolci-Instruct-SFT` as the best-guess slug and make it
  `--instruct-dataset`-overridable. If the dataset can't be loaded offline, SFT
  still runs without the mix (with a warning) — documented.

### 4.3 Training (Table 9)
`finetune/train_dpo.py` / `train_sft.py` via TRL + PEFT LoRA, hyperparameters
transcribed exactly:

| | DPO | SFT |
|---|---|---|
| epochs | 1 | 2 |
| lr | 5e-5 | 1e-4 |
| LoRA rank | 64 | 64 |
| LoRA alpha | 64 | 128 |
| effective batch | 8 | 8 |
| DPO beta | 0.1 | — |
| targets | q/k/v/o + gate/up/down proj (all layers) | same |

`effective_batch_size` is realised as `gradient_accumulation_steps =
effective/per_device`. The **Appendix I layer ablation** is a config knob
(`lora.layers_to_transform`, profile `dpo_layers_30_35`) feeding PEFT's
`layers_to_transform`/`layers_pattern`.

**Serving finetunes:** the LoRA adapter dir is referenced via
`ModelConfig.lora_path`; `VLLMClient` loads it through vLLM's LoRA support, so the
finetuned Gemma is evaluated by the *same* §2 pipeline (`gemma-3-27b-it-dpo` row).

### 4.4 Open-ended Petri eval (§4.2 / Appendix G)
`openended/petri_eval.py` is a **self-contained** auditor↔target↔judge loop using
the **verbatim** Appendix G auditor instructions and judge rubrics. Auditor =
Claude Sonnet, judge = Claude Opus, ≤20 turns, 10 transcripts per emotion
(anger/fear/depression/frustration) per target.

**Choice:** we re-implement the loop rather than depend on the external `petri`
package, so the replication runs without a heavyweight, version-sensitive
dependency and uses exactly the paper's prompts. The paper says "~50 total"
transcripts; 4 emotions × 10 = 40 (we don't include a 5th catch-all category —
noted). If you have the real Petri installed you may prefer it; the prompts here
are drop-in.

### 4.5 Capability preservation (§4.2, Figure 7)
**Not implemented.** AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench are standard public
benchmarks; re-running them is orthogonal harness plumbing and not a "core"
distress result. We note it as the one explicitly-listed §4 result we skipped, to
keep scope on the distress measurements. (Hook point: evaluate the
`...-dpo`/`...-sft` model rows with any existing eval-harness.)

### 4.6 Internal-emotion probing (Appendix I)
Partially supported: the **layer-ablation** half (which layers must be trained) is
a training config. The **logit-based internal emotion detector** (Ekman-token
z-scores over the residual stream) is **not implemented** — it's an interpretability
add-on beyond the core behavioural replication. Flagged here as a deliberate
omission.

---

## 5. Cross-cutting choices

- **Counting / metrics:** see §2.2–2.3 above. All thresholds (`≥5`) and the
  category-averaged headline are centralised in `eval/metrics.py`.
- **Determinism:** every stage takes a `seed`; puzzle generation, rejection
  sampling, and dataset subsampling are seeded. Target sampling is temp-1 so
  draws are intentionally stochastic.
- **Cost control:** `configs/eval.smoke.yaml` and `--scale` shrink a run to ~1%
  for end-to-end validation before the full 4000-response run. Throughput for
  local Gemma uses batched vLLM generation per turn (`vllm_client.batch_generate_chat`).
- **Robustness:** all judge/onset/paraphrase calls parse JSON defensively and
  retry API calls with exponential backoff (`tenacity`).
- **Offline degradation:** WildChat and the instruct-mix datasets fall back
  gracefully (bundled `data/wildchat_fallback.json`; empty instruct mix with a
  warning) so the pipeline is exercisable without those downloads.

---

## 6. Known deviations from the paper (summary)

| Area | Paper | Here | Why |
|---|---|---|---|
| Model set | 7 families | Gemma + Gemini only | Per brief |
| §3 families | Gemma/Qwen/OLMo | Gemma only | Scope + only Gemma has a base model in scope |
| 8 conditions | unenumerated | tones×3 + triggers×2 split | Only split giving 8 from the 5 categories |
| "response" unit | ambiguous | scored assistant turn | Needed for per-turn fig; reproduces 4000 total |
| Puzzle instances | 2 shown | verbatim + verified-impossible generators | Need many distinct impossible prompts |
| Dolci instruct id | cited, no slug | `allenai/Dolci-Instruct-SFT` (overridable) | Best-guess HF id |
| DPO pair prompt | "same question/turn" | calm context as shared prompt | Need one canonical prompt per pair |
| Petri | external framework | built-in loop, verbatim prompts | Avoid heavy dep; same prompts |
| Capability benches | AIME/MATH/GPQA/BBH/TQA/EmoBench | omitted | Not a core distress result |
| Internal probing | logit z-score detector | layer-ablation only | Interpretability add-on, out of core scope |
| Base-model templating | "prefilled continuation" | plain `Role:` transcript | Exact templating unspecified |

Everything marked "omitted" is non-core or an interpretability extension; the
behavioural pipeline (elicit → judge → metrics → intervene → re-evaluate) that
produces the paper's headline 35%→0.3% claim is fully implemented for Gemma, with
Gemini covered for the elicitation half.
