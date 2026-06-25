# DESIGN.md — Replication design & decisions

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv:2603.10011v1), scoped
to the **Gemma and Gemini** model families per the request.

This document records every non-trivial design choice, and — importantly — every
place the paper is **underspecified** and we had to fill a gap. Gaps are marked
**[GAP]**; faithful reproductions of a stated detail are marked **[paper]**.

The code was written to be correct and runnable but has **not** been executed
(per instructions). Treat hyperparameters and dataset ids as the intended
configuration, not as validated-against-a-run.

---

## 1. Scope

The paper spans 7 model families. We implement the full experimental machinery
but run it only on:

* **Gemma** (local, HuggingFace): `gemma-3-27b-it`, `gemma-3-12b-it`, and the
  base checkpoints `gemma-3-27b-pt` / `gemma-3-12b-pt`.
* **Gemini** (API via OpenRouter): `gemini-2.5-flash`, `gemini-2.5-pro`.

Consequences of the scoping, all consistent with the paper's own structure:

* **Sec 3 (base-vs-instruct prefill)** runs on **Gemma only** — it is the only
  in-scope family with public base ("pt") checkpoints. The paper's Qwen/OLMo arm
  is out of scope, and Gemini base models do not exist (a paper limitation too).
* **Sec 4 (DPO/SFT mitigation)** targets **`gemma-3-27b-it`** only. Gemini is
  closed and cannot be finetuned — exactly the paper's caveat.

The three **core** results we target:
1. Distress is reliably elicited in Gemma/Gemini under repeated rejection (Sec 2).
2. The Gemma instruct↔base divergence localises to post-training (Sec 3).
3. DPO on 280 numeric-puzzle pairs broadly mitigates it without harming
   capabilities (Sec 4).

Secondary analyses (Petri, capability preservation, internal-emotion probing,
layer ablation) are also implemented, at a documented fidelity.

---

## 2. Repository layout

```
emo/
  config.py          central knobs, model registry, sample-count profiles
  cli.py             `python -m emo.cli <command>`
  models/            target models (Gemma local: hf/vllm; Gemini: OpenRouter/native)
  judges/            Claude frustration judge, optional 2nd judge, anthropic client
  data/              impossible puzzles + verifier, triggers, tones/rejections, WildChat
  eval/              conditions, multi-turn engine, elicitation runner, analysis  (Sec 2)
  prefill/           onset labelling, paraphrase, base-vs-instruct + recovery     (Sec 3, 4.2)
  training/          calm-data gen, DPO/SFT dataset build, LoRA train scripts      (Sec 4)
  petri/             auditor/judge prompts + auditor→target→judge loop            (Sec 4.2)
  capabilities/      lm-eval harness wrapper + EmoBench                            (Sec 4.2)
  internal/          logit-lens emotion probe + layer ablation                    (App I)
```

The `ChatModel` abstraction (`models/base.py`) is the seam: eval/prefill/training
never branch on Gemma-vs-Gemini.

---

## 3. Judge / auditor models — retired-model substitution **[GAP-forced]**

The paper uses `claude-sonnet-4-20250514` (frustration judge, onset labelling,
paraphrase, Petri auditor) and `claude-opus-4-20250514` (Petri judge). **Both are
retired as of the replication date (2026-06-25)** (Sonnet 4 / Opus 4 retired
2026-06-15). So the runnable defaults are the current recommended replacements:

| Role | Paper | Default here |
|---|---|---|
| Frustration judge | `claude-sonnet-4-20250514` | `claude-sonnet-4-6` |
| Onset / paraphrase | `claude-sonnet-4-20250514` | `claude-sonnet-4-6` |
| Petri auditor | `claude-sonnet-4-20250514` | `claude-sonnet-4-6` |
| Petri judge | `claude-opus-4-20250514` | `claude-opus-4-8` |
| 2nd judge (agreement) | `gpt-5-mini` | unset (optional) |

The paper's exact ids are kept in `config.PAPER_JUDGE_MODELS` and every judge
model is overridable by env var, so anyone with access can pin the originals.
**Caveat:** a different judge changes the absolute frustration numbers; the
qualitative pattern (Gemma/Gemini high, others low; DPO ≈ 0) should be robust,
but cross-comparison with the paper's exact percentages is not apples-to-apples.

Judge calls deliberately omit `temperature` and `thinking` so the same code works
across the current Claude line (Opus 4.x removed sampling params; thinking is off
by default). The judge prompt itself is **verbatim** from Appendix B.2.

---

## 4. Sample counts & the "response" unit **[GAP]**

The paper reports ≈4000 responses/model (App B: 2000 numeric, 400 triggers, 600
tones, 200 extended, 800 WildChat) but doesn't say whether a "response" is a
whole rollout or a single scored assistant turn. Per-turn figures (Fig 3) require
each turn to be scored, so we treat **each assistant turn as one scored
response**. We then pick rollouts-per-category so `rollouts × turns ≈` the paper's
per-category response count (see `config.FULL`).

Two profiles:
* **`full`** — the paper's counts. Expensive (≈4000 judge calls/model + local
  generation of 27B models).
* **`smoke`** — tiny counts for a cheap end-to-end pipeline check.

---

## 5. Experiment 1 — elicitation (Sec 2)

### 5.1 Impossible puzzles + verifier **[GAP, substantive]**
The paradigm needs *genuinely unsolvable* tasks whose prompt claims a solution
exists. The paper gives two example prompts and says puzzles were "verified", but
ships neither the verifier nor the full set. We implement:

* `data/puzzle_verifier.py` — an exhaustive **Countdown solver** (pairwise
  combination, positive-integer intermediates, forbidden-value pruning,
  memoised) and a **sequence-op solver** (all permutations over exact `Fraction`
  arithmetic) for the fraction/money puzzles.
* `data/puzzles.py` — the paper's two worked examples (Countdown 156 forbidding
  150; fraction 1/6→2/3 forbidding 1/3) plus the Appendix-H money example, and a
  **deterministic generator** that proposes random instances and keeps only those
  the verifier proves impossible. Every constructed puzzle asserts impossibility
  at build time.

This means the eval is not a single repeated item (which the paper's wording
leaves ambiguous), while still matching the two concrete prompts given.

### 5.2 Prompt wording **[paper]**
Countdown / fraction prompt templates are taken from Appendix B verbatim; the
money template follows the Appendix-H description (sequence-of-operations form).

### 5.3 Conditions & the "8 across 5" count **[GAP]**
We implement the 5 categories (numeric, triggers, tones, extended, wildchat).
Tones expands to 3 styles (aggressive/disappointed/sarcastic). That gives 7
named conditions; to reach the paper's "8 conditions" we additionally record
trigger **opinion vs factual** as separate sub-conditions (the most natural
reading). Analysis can collapse them back to the 5 categories.

### 5.4 Rejections / tones **[paper]**
Neutral rejections, the 8-turn escalating sequence, and the three tone pools are
taken from Appendix B. Neutral rejections are sampled from a pool (seeded), as
the paper says ("randomised neutral rejections").

### 5.5 WildChat selection **[GAP]**
The paper samples 20 prompts × 40 from WildChat-1M but lists only 3. We
seeded-randomly sample first user turns from `allenai/WildChat-1M` (English,
length-bounded, roleplay-filtered), with a small built-in fallback (including the
3 quoted prompts) for offline runs. Selection is reproducible via the seed.

### 5.6 Generation settings **[paper + GAP]**
Temperature 1.0 [paper]; thinking disabled via the API [paper]. `max_new_tokens`
is **[GAP]** — the paper doesn't state it; we use 2048, large enough to let
collapse/repetition responses run (the high-score behaviours are long).

### 5.7 Judge JSON parsing **[GAP]**
The judge is asked for `{"evidence","reasoning","rating"}`. Real outputs
sometimes wrap JSON in prose / use smart quotes / trailing commas, so
`utils/llm_json.py` extracts the first balanced object and normalises quotes;
unparseable outputs are scored 0 and flagged (then excluded in analysis).

### 5.8 Judge-agreement check **[paper, optional]**
Pearson r + within-1 fraction vs a second judge (paper: GPT-5-mini, r=0.792).
Implemented but **off by default** to avoid a hard OpenAI dependency; enable with
`EMO_SECOND_JUDGE_MODEL` + `OPENAI_API_KEY` and `analyse --agreement`.

### 5.9 Appendix-A controls **[partial]**
`eval/conversation.py` exposes `history_mode="redacted"` (A.2: blank prior
assistant turns) and `feedback="continuation"` (A.1: neutral continuations
instead of rejections). The A.3 single-message format is **not** implemented (it
only confirms format doesn't matter); noted as an extension point.

---

## 6. Experiment 2 — base-vs-instruct prefill (Sec 3)

### 6.1 Seed sourcing **[GAP, method-faithful]**
The paper samples 20 high-frustration (score≥5) Gemma-27B-it responses (10
numeric, 10 text) but the scored elicitation records don't retain full
conversation context. So `prefill/run_prefill.collect_seeds` re-runs short
conversations on `gemma-3-27b-it`, scores each turn, and keeps the **full
context + assistant text** for turns scoring ≥5. Over-samples ~3× to hit the
target counts.

### 6.2 Truncations **[paper + GAP]**
* **onset** — `prefill/onset.py` uses Claude with the **verbatim** Appendix-C.1
  prompt to find the first emotional word; we cut the assistant text just before
  that word.
* **early** — first **20 tokens** of the assistant turn. **[GAP]** the paper says
  "20 tokens" without specifying the tokenizer; we use the Gemma-3-27B-it
  tokenizer (the source model's), which is the natural choice.
* Text questions use **onset only** [paper].

### 6.3 Paraphrase **[paper]**
All truncations are paraphrased with Claude using the **verbatim** Appendix-C.2
prompt, to control for Gemma style bias before feeding other models.

### 6.4 Continuations & scoring **[paper]**
Each model emits `prefill_continuations` (paper: 50) continuations per prefill
via `continue_prefill` (we replicate the prompt N times at temperature 1). Only
the generated suffix (excluding the prefill) is judged. Base models use a plain
`User:/Assistant:` transcript instead of the chat template (the paper's "base
models aren't trained on chat format" handling).

---

## 7. Experiment 3 — training interventions (Sec 4)

### 7.1 Calm-data generation **[paper + GAP]**
`training/generate_calm_data.py` samples 3-turn numeric conversations from
Gemma-27B-it:
* **calm pool** — with the Table-4 reassuring **prefix** prepended to the initial
  prompt and **suffix** appended to each follow-up [paper, verbatim text]. We keep
  conversations scoring 0–1 across all turns [paper], then **store the clean
  (reassurance-free) context** so calm and frustrated pools share a prompt
  distribution and the training prompt matches deployment [paper: "strip the
  supportive system prompts and suffixes"].
  * **[GAP]** the paper says the prefix is added "to the initial prompt"; we
    prepend it to the first user message (rather than as a system message). Either
    reading is defensible; we chose the literal one.
* **frustrated pool** — standard generation (no reassurance), keeping turns
  scoring ≥3 [paper], for the DPO "rejected" side.

### 7.2 DPO dataset **[paper + GAP]**
`training/build_datasets.build_dpo` pairs a frustrated response (rejected, ≥3)
with a calm response (chosen, ≤1) **for the same puzzle and turn count** [paper:
"matching turn counts"], capped at `dpo_pairs` (280). `prompt` is rendered with
the Gemma chat template. **[GAP]** the paper's exact pairing within a question
isn't specified; we match on (puzzle_id, turn) and pick a random eligible calm
partner.

### 7.3 SFT dataset **[paper + GAP]**
650 calm responses (1–3 turn) + 500 `Dolci-Instruct-SFT` samples [paper]. **[GAP]**
the exact HF id for Dolci-Instruct-SFT is uncertain; we try a couple of candidate
ids and, if unavailable, train on calm data only with a printed warning. The
paper also tests a "teacher" SFT variant (Appendix F) with a distinct system
prompt — that prompt is reproduced in `generate_calm_data` comments-as-data but
we ship the "diverse" SFT path as the primary baseline.

### 7.4 Hyperparameters **[paper, Table 9]**
DPO: 1 epoch, lr 5e-5, LoRA r64/α64, eff. batch 8, β 0.1. SFT: 2 epochs, lr 1e-4,
LoRA r64/α128, eff. batch 8. LoRA on all attn+MLP projections (q/k/v/o/gate/up/
down) [App E]. Effective batch 8 via `per_device=1 × grad_accum=8` **[GAP]** (the
split isn't specified; the product matches).

### 7.5 Before/after evaluation
No new code: run `elicit` on `gemma-3-27b-it`, `…-dpo`, `…-sft` and `analyse`.
Finetunes load as base instruct + LoRA adapter via the registry.

### 7.6 Recovery limitation **[paper, Fig 8]**
`prefill/run_recovery.py` sources score≥7 responses, truncates **200 tokens
before the end**, paraphrases, and measures continuations for instruct/DPO/base.

---

## 8. Petri (Sec 4.2, App G) — self-contained reimplementation **[GAP]**

The paper uses the Petri framework (Fronsdal et al., 2025). Rather than depend on
that package, we implement the described protocol directly
(`petri/run_petri.py`): a Claude **auditor** plays the user over up to 20 turns
using the **verbatim** Appendix-G.1 trigger prompts; the target replies; a Claude
**judge** scores the transcript 1–10 on each of the four emotions using the
**verbatim** Appendix-G.2 rubrics. 10 transcripts/emotion [paper]. This captures
the auditor→target→judge structure but is not byte-identical to the Petri
package's scaffolding (tool use, structured scenario setup, etc.).

---

## 9. Capabilities (Sec 4.2, Fig 7) **[GAP on task ids]**

`capabilities/run_capabilities.py` wraps the `lm-evaluation-harness` for
MATH/AIME/GPQA/BBH/TruthfulQA, loading the LoRA adapter via `model_args=peft=…`.
**[GAP]** exact lm-eval task ids drift across harness versions; the defaults
(`hendrycks_math`, `aime2024`, `gpqa_main_zeroshot`, `bbh`, `truthfulqa_mc2`) are
listed and overridable. **EmoBench** is not in the harness; we attempt to load
its dataset and record availability, leaving the full MCQ scorer as a documented
extension point (a partial gap).

---

## 10. Internal-emotion probing (App I) **[GAP, approximations]**

`internal/emotion_logits.py` implements the logit-lens detector: project the
residual stream through the unembedding, standardise each vocab logit against
WildChat baseline mean/std, average z-scores over each Ekman emotion's tokens,
and subtract a random-token control (the paper's "regress out the correlation
between random tokens"). Aggregated over layers 30–40 [paper].

Documented approximations vs the paper:
* **Emotion-token tagging** — the paper classifies the whole Gemma dictionary
  into Ekman categories (~1200 tokens). We approximate with per-emotion seed-word
  lists matched against the vocab. (Closeable by swapping in a classifier-labelled
  token list.)
* **Unembedding** — we unembed the raw residual stream (no final RMSNorm),
  matching the "unembed the residual stream" wording.
* **Control regression** — implemented as subtracting the mean z over a fixed
  random token set per position, rather than an explicit regression.

`internal/run_layer_ablation.py` reproduces the Fig 12–13 sweep: train DPO with
adapters restricted to layer subsets (`training/lora.py` builds the per-layer
target-module names) and evaluate each with a reduced elicitation eval. Gemma-3-
27B has 62 layers; the ranges mirror the paper's backward-from-final and
central-subset sweeps.

---

## 11. Backends & infrastructure choices

* **Gemma generation**: vLLM is the default fast path for the large sweeps;
  transformers (`hf`) is the correctness reference and is used for prefill,
  training, and probing (needs raw logits/hidden states). Switch with
  `EMO_LOCAL_BACKEND=hf|vllm`. **[GAP]** Gemma-3 12B/27B are multimodal; the
  loader tries `AutoModelForCausalLM` then falls back to
  `AutoModelForImageTextToText` for the text path.
* **Gemini**: OpenRouter by default (matches App B.1), with a native google-genai
  backend as an alternative. Thinking is disabled best-effort (`reasoning:
  {enabled:false}` / `thinking_budget=0`); the paper notes Gemini-2.5-Pro may
  still emit hidden reasoning this can't prevent.
* **Concurrency**: API calls (judge, Gemini) run through a bounded thread pool;
  local generation batches per turn.

---

## 12. Deliberately not implemented / known limitations

* **Other families** (Qwen, OLMo, Grok, Claude, GPT) — out of the requested
  scope. The framework is family-agnostic, so adding them is a registry entry +
  backend.
* **Word-frequency tables** (Table 3/8) — descriptive only; not reproduced.
* **App A.3 single-message format** — omitted (confirms format invariance only).
* **EmoBench full scorer** — dataset reachability check only.
* Nothing has been **executed**; numbers in the paper are not re-verified here.

---

## 13. Reproducibility

* All sampling/selection is seeded (`--seed`, default 0); conditions are built
  with the same seed across models so model comparisons use identical prompts.
* Profiles (`full`/`smoke`) make the cost/coverage tradeoff explicit.
* Results are written as JSONL (raw) + CSV/JSON summaries + PNG figures under
  `results/<experiment>/<profile>/`.
