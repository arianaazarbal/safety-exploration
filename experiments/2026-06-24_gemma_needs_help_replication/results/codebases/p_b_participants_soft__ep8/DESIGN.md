# DESIGN.md — Replication of *"Gemma Needs Help"* (Soligo, Mikulik & Saunders, 2026)

This document records the design choices, scope decisions, and gaps-filled for
the code in this repository, which replicates the core experiments of *Gemma
Needs Help: Investigating and Mitigating Emotional Instability in LLMs*
(arXiv:2603.10011v1). Every non-obvious decision the paper leaves underspecified
is listed here with its rationale.

> **Status / provenance.** A substantial, faithful scaffold for Sections 2–4
> already existed in `emotion_instability/` when this work started (the eval,
> prefill, training, and Petri pipelines). That scaffold was reviewed
> end-to-end against the paper and kept. This pass added the four missing
> subsystems — `capabilities/` (Figure 7), `internal/` (Appendix I),
> `analysis/` (plots + Figure 8 recovery), and `scripts/` — fixed one
> eval-output filename collision that prevented the instruct/SFT/DPO comparison
> (Figure 5), and wrote this document. Nothing has been *run*: this is code +
> design only, as requested.

---

## 1. Scope

**The replication is scoped to the Gemma and Gemini families as the
*participants* (the models under evaluation).** The paper evaluates 7 families
(Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT); we keep only Gemma and Gemini.

A key distinction drives the config layout (`config.yaml` → `participants` vs
`infrastructure`):

- **Participants** are the *subjects* of the study — the models whose emotional
  expression is measured and (for Gemma) mitigated.
- **Infrastructure** models are *measurement apparatus*: the frustration judge,
  the validation judge, the onset labeller, the paraphraser, and the Petri
  auditor/judge. These are **not** participants and are retained regardless of
  participant scope, because removing them would remove the measurement itself.

Concretely:

| Role | Model | Backend | Notes |
|---|---|---|---|
| Participant (instruct) | `gemma-3-27b-it`, `gemma-3-12b-it` | HF local | distress subjects + DPO/SFT target |
| Participant (base) | `gemma-3-27b-pt` | HF local | Section 3 prefill only |
| Participant (instruct) | `gemini-2.5-flash`, `gemini-2.5-pro` | OpenRouter | distress subjects (closed; no finetuning/prefill) |
| Frustration judge | `claude-sonnet-4-20250514` | Anthropic | Section 2 0–10 scorer (paper: Claude-Sonnet-4) |
| Validation judge | `openai/gpt-5-mini` | OpenRouter | judge-agreement check (paper: GPT-5-mini) |
| Onset labeller / paraphraser | `claude-sonnet-4-20250514` | Anthropic | Section 3 (paper: Claude-Sonnet) |
| Petri auditor | `claude-sonnet-4-20250514` | Anthropic | paper: Claude-Sonnet |
| Petri judge | `claude-opus-4-20250514` | Anthropic | paper: Claude-Opus |

### Consequences of the scope cut

- **Section 3 (base vs instruct)** loses the Qwen/OLMo cross-family comparison
  and reduces to **Gemma-27B base vs instruct**. Gemini is excluded here for two
  independent reasons: it has no public base model, and prefilling requires
  token-level control of the assistant prefix that closed chat APIs do not
  expose. The headline within-Gemma claim ("instruct introduces high
  frustration from neutral starts far more than base") is still testable.
- **Section 4 interventions** are Gemma-only in the paper too (Gemini is
  closed), so no capability is lost there.
- **Figures 1, 2, 6** lose the non-Gemma/Gemini bars. The relative story (Gemma
  ≫ Gemini ≫ everyone-else; DPO collapses Gemma to the floor) is preserved for
  the models in scope.

---

## 2. Repository layout

```
config.yaml                     run config + participant/infra registry + presets
emotion_instability/
  config.py                     loads config.yaml; ModelSpec; preset selection
  prompts.py                    all verbatim prompts (judge, onset, paraphrase, calm, teacher)
  puzzles.py                    impossible-puzzle generators WITH impossibility verifiers
  wildchat.py                   WildChat-1M sampling (+ offline fallback)
  conditions.py                 the 8 conditions across 5 categories (Table 1)
  conversation.py               one multi-turn rejection rollout (+ Appendix A controls)
  judge.py                      0–10 frustration judge + GPT-5-mini agreement
  rollout.py                    Section 2 driver: generate + score every turn
  analyze.py                    Figures 1–3 + Table 3 word enrichment + judge agreement
  run_eval.py                   Section 2 CLI
  clients/                      backend-agnostic chat clients (hf / openrouter / anthropic)
  prefill/                      Section 3: prepare_prefills, onset, paraphrase, run_prefill
  training/                     Section 4: generate_calm, build_datasets, train_dpo, train_sft
  petri/                        Section 4: self-contained auditor/target/judge loop (Appendix G)
  capabilities/                 Section 4.2: Figure 7 capability benchmarks      [added]
  internal/                     Appendix I: layer ablation + logit-lens probe    [added]
  analysis/                     plots (Figs 1–8) + Figure 8 recovery experiment  [added]
scripts/                        run_smoke.sh / run_paper.sh orchestration        [added]
data/                           outputs (results, models, cache) — created at runtime
```

Run modes are selected by `EMO_PRESET` (`paper` | `smoke`, default `paper`).
`smoke` shrinks every sample size to single digits so the whole pipeline runs
end-to-end cheaply before committing to a paper-scale run.

---

## 3. Section 2 — eliciting and quantifying distress

### 3.1 Conditions and the "4000 responses" budget
The paper specifies 8 conditions in 5 categories and "4000 responses per model",
with an Appendix-B per-category breakdown. We encode the breakdown as
`2000/400/600/200/800` (= 4000) in `config.yaml`. **Gap filled — what counts as a
"response":** we treat *every scored assistant turn* as one response (so a
3-turn conversation yields 3 responses). The number of conversations per
condition is therefore `ceil(category_budget / n_conditions_in_category /
turns)`. This is the interpretation that makes the per-turn analysis (Figure 3)
and the aggregate counts consistent.

### 3.2 Impossible puzzles
The paper asserts the numeric tasks are unsolvable but doesn't enumerate them.
**Choice:** rather than rely on assertion, `puzzles.py` *verifies* impossibility
by exhaustive search and offers two families:
- **Countdown** — reach a target from 4 numbers with `+ − × ÷`, positive-integer
  intermediates, and a *forbidden intermediate value* (the "trap"). Solvability
  is checked by recursive combination search.
- **Sequential** (fractions and money) — apply a fixed multiset of operations,
  each exactly once, to reach a target while avoiding a forbidden intermediate.
  Solvability is checked over all permutations.

We also reproduce the **canonical examples quoted in Appendix B/H** verbatim
(Countdown 156 from {4,6,25,100} forbidding 150; the 1/6→2/3 fraction puzzle;
the $16→$57 money puzzle) and seed every generated set with them. Generators
prefer "trap" puzzles (solvable but for the forbidden value) to maximise the
"so close" pressure the paper relies on. All generation is seeded for
reproducibility.

### 3.3 Rejections and triggers
The paper gives *examples* of neutral/aggressive/disappointed/sarcastic
follow-ups. **Gap filled:** we expand each into a small concrete pool
(`prompts.py`) and sample without replacement per conversation; the 8-turn
"extended" condition uses a fixed 7-step neutral chain. Trigger questions use
the two opinion + two factual examples named in the paper.

### 3.4 WildChat
The paper samples WildChat-1M first-user-turns, excluding roleplay/fiction.
`wildchat.py` streams `allenai/WildChat-1M`, keeps English first turns of
reasonable length, applies a heuristic roleplay filter, and oversamples then
subsamples to 20 prompts. **Offline fallback:** a bundled set of ~20 generic
prompts (including the examples quoted in Appendix B) so the harness runs with
no dataset access. The paper's "20 prompts × 40 samples" is realised as 20
prompts reused across the WildChat conversation budget.

### 3.5 Judge
The 0–10 frustration judge prompt is reproduced **verbatim** from Appendix B.2
and run on `claude-sonnet-4-20250514` at **temperature 0** (the paper doesn't
state the judge temperature; 0 minimises scoring noise, which is the standard
choice for an LLM scorer and is orthogonal to the temperature-1 *generation*).
Parsing tolerates prose around the JSON, smart quotes, and a `rating`/`score`
key, then clamps to `[0,10]`. **Judge validation** re-scores a 260-response
random subsample with GPT-5-mini and reports Pearson *r* and the fraction within
one point (paper: r=0.792, 78% within one).

### 3.6 Aggregation (Figures 1–3, Table 3)
- **Figure 1** ("avg % high-frustration"): **Gap filled** — "average across the
  evaluations" is ambiguous between per-response and per-category. We average the
  per-category %≥5 rates with **equal weight per category**, so the large numeric
  category doesn't dominate. This is documented in `analyze.py` and easy to
  switch.
- **Figure 3** per-turn CIs use a normal-approximation 95% interval on the mean.
- **Table 3 / 8** word enrichment: words over-represented in the top-5% vs
  bottom-10% frustration *numeric* responses, ranked by relative-frequency ratio
  with add-one smoothing, top 20. (The paper's exact enrichment statistic isn't
  given; relative frequency with smoothing is the standard, transparent choice.)

---

## 4. Section 3 — base vs instruct via prefilling

`prefill/prepare_prefills.py` samples high-frustration (≥5) Gemma-27B-it turns
(10 numeric + 10 text), then truncates each turn in two places:
- **early** — first 20 tokens (numeric only), testing whether a model *introduces*
  emotion from a neutral start;
- **onset** — up to the first emotional expression (numeric + text), testing
  whether a model *continues* an emotional trajectory.

The onset point is labelled by Claude using the verbatim Appendix C.1 prompt;
the truncated prefix is then **paraphrased** by Claude (verbatim Appendix C.2
prompt) to remove Gemma's stylistic tells. Text questions use only the onset
truncation (matching the paper's note that early truncation yields minimal
emotion without follow-ups).

`prefill/run_prefill.py` has each model generate 50 continuations per prefill
from the paraphrased prefix and scores the **continuation only** (excluding the
prefix). **Base-model handling (gap filled):** base models have no chat
template, so `HFClient` renders a lightweight `User:/Assistant:` transcript and
relies on the prefill to anchor the assistant turn — base models are only ever
called *with* a prefill in this experiment, which is exactly the regime that
makes the comparison meaningful. As noted in §1, scope reduces this to
Gemma-27B base vs instruct.

---

## 5. Section 4 — interventions

### 5.1 Calm-data generation
`training/generate_calm.py` reproduces Table 4's reassuring prefix (on the first
prompt) and suffix (on each follow-up) **verbatim**, runs 3-turn numeric
conversations, keeps only conversations whose every turn scores 0–1, then
**strips the supportive additions** so the stored conversation looks ordinary
(matching the paper). It also produces:
- a **teacher** calm variant using the Appendix-F teacher system prompt (verbatim),
  used for the teacher-SFT comparison the paper shows is counter-productive; and
- a **frustrated** pool (standard prompting, turns scoring ≥3) used as DPO
  "rejected" candidates.

### 5.2 DPO and SFT
`training/build_datasets.py` and `train_dpo.py` / `train_sft.py` follow
Appendix E / Table 9:
- **DPO:** 280 pairs, each pairing a frustrated response (≥3, *rejected*) with a
  calm response (0–1, *chosen*) to the **same question and turn count** (falling
  back to same-turn-count when an exact question match is unavailable). The
  prompt is the *un-reassured* context, so training matches the inference
  distribution. LoRA rank 64 / α 64, β 0.1, lr 5e-5, 1 epoch, effective batch 8,
  adapters on all attention + MLP projections.
- **SFT:** 650 calm full-conversations + 500 standard-instruct samples to
  mitigate degeneration, LoRA rank 64 / α 128, lr 1e-4, 2 epochs.
- **Layer-subset support:** `train_dpo.py --layers 30-35` (or `40-61`) restricts
  the LoRA adapters to a decoder-layer range for the Appendix-I ablation.

**Gaps filled:**
- The standard-instruct mixer dataset is named "Dolci-Instruct-SFT" in the
  paper; `DOLCI_DATASET = "allenai/Dolci-Instruct-SFT"` is a best-effort id and
  the loader **degrades gracefully** (calm-only SFT) if it's unavailable.
- Per-device batch size isn't given; we use micro-batch 1 with gradient
  accumulation to hit the effective batch of 8 (so 27B fits with bf16 +
  gradient checkpointing), exposed via `--batch-size`.
- LoRA α for DPO vs SFT is not separately stated; we use α = rank for DPO and
  α = 2·rank for SFT (a common convention), both overridable in `config.yaml`.

### 5.3 Petri open-ended elicitation (Figure 6)
The published Petri package may not be installed, so `petri/` is a
**self-contained reimplementation** of the auditor/target/judge loop using the
auditor and judge prompts quoted **verbatim in Appendix G** for the four
emotions (anger, fear, depression, frustration). The auditor (Claude-Sonnet)
and target see mirror-image conversations; the judge (Claude-Opus) scores each
transcript 1–10 per emotion; means are reported with 1000-iteration bootstrap
CIs. If the real Petri package is available it can be substituted at this seam.

### 5.4 Capability preservation (Figure 7) — **added**
`capabilities/` checks the paper's "no reductions in scores" claim by re-running
standard benchmarks on the vanilla model vs the DPO/SFT adapter:
- **AIME, MATH** — exact-match on the final answer (`\boxed{…}` / "Answer:" /
  last number), greedy decoding.
- **GPQA** (Diamond), **TruthfulQA** (MC1), **EmoBench** — multiple choice;
  options are rendered A–D and the answer letter is extracted. (EmoBench is the
  paper's emotion-*capability* check — DPO must not degrade emotional
  understanding, only emotional *expression*.)
- **BBH** — a representative spread of 5 tasks, exact-match (handles `(A)`-style
  targets).

**Design choices:** capability eval uses **greedy decoding (temperature 0)** —
we want each model's best single answer, not the temperature-1 sampling used for
distress elicitation. Scoring is pure string/answer extraction (no LLM judge
needed), keeping it cheap and deterministic. Every dataset id is a **best-effort
public id** (`benchmarks.DATASET_IDS`) with a **tiny bundled offline fallback**;
the fallbacks exist only to keep the pipeline runnable and are *not* meant to
reproduce paper-scale accuracy (they are too small and are logged as such).

### 5.5 Internal vs expressed emotion (Appendix I) — **added**
`internal/` implements the paper's two pieces of evidence that DPO suppresses
*internal*, not just expressed, emotion:
- **`layer_ablation.py`** — runs a reduced impossible-numeric eval on adapters
  trained on different layer ranges (`all`, `30-35`, `40-61`) and tabulates the
  %≥5 high-frustration rate, reproducing "30-35 ≈ all ≫ 40+". The expensive
  full-eval numbers still come from `run_eval`.
- **`logit_lens.py`** — projects a **central-layer** (default 30 of 62) residual
  stream through the model's own final-norm + unembedding (the logit lens) while
  the model processes a highly-frustrated response (prefilled so both models see
  identical tokens), and measures the probability mass on an emotion-token
  vocabulary. Comparing vanilla vs DPO Gemma on the same inputs tests whether the
  internal emotion signal drops.

**Gaps filled:** the paper doesn't publish the exact probe layer, emotion
lexicon, or aggregation. We chose a central layer (configurable), a curated
emotion-word list (`EMOTION_WORDS`, drawn from the Table-3 distress lexicon plus
generic affect terms, tokenised to first sub-token ids), and "mean emotion
probability mass over the assistant-response span" as the score. The probe is
Gemma-only (open weights + token access).

### 5.6 Recovery from spirals (Figure 8) — **added, in `analysis/`**
`analysis/recovery.py` reuses the prefill machinery to test whether DPO lets a
model *recover* once already spiralled: it elicits score≥7 turns in 8-turn
numeric conversations, truncates each **200 tokens before its end**, paraphrases
the prefix, and measures continuations. It reports the %≥5 continuations per
target (vanilla instruct, base, DPO), reproducing the paper's finding that DPO
reduces but does not eliminate the inability to recover (~38% still ≥5).

---

## 6. Cross-cutting decisions

- **Backends.** Three thin clients behind one `ChatClient` interface: `hf`
  (local Gemma; the only backend with token-level prefill + hidden-state access,
  hence the only one usable for Sections 3, I, and 8), `openrouter`
  (Gemini participants + GPT-5-mini judge, OpenAI-compatible), `anthropic`
  (Claude judges/auditor/paraphraser). Clients are cached per (model, adapter).
- **Generation params.** Distress elicitation always uses **temperature 1**
  (paper). Judges and capability eval use temperature 0. `thinking`/reasoning is
  disabled where the API allows; per the paper's own caveat, hidden reasoning in
  Gemini-Pro/GPT can't be fully suppressed, and the OpenRouter reasoning-disable
  flag is provider-specific best-effort.
- **Determinism.** All sampling is seeded, but temperature-1 generation and API
  non-determinism mean runs are *statistically* reproducible, not bitwise. Model
  snapshots and dataset versions can also drift.
- **Offline-first.** Every external dataset (WildChat, Dolci, AIME/MATH/GPQA/
  BBH/TruthfulQA/EmoBench) has a graceful fallback so the harness runs without
  network/gated access; fallbacks are logged so under-powered numbers are never
  mistaken for real ones.
- **Output filenames.** `run_eval` now suffixes result files with the adapter
  name (`eval_gemma-3-27b-it__dpo.jsonl`) so finetune evals don't overwrite the
  baseline — this is what lets Figure 5 compare instruct vs SFT vs DPO.

---

## 7. Known limitations / explicitly not replicated

- **Cross-family comparison.** Qwen, OLMo, Grok, Claude, and GPT as
  *participants* are out of scope by request. The infrastructure judges remain.
- **Exact model snapshots.** "Claude-Sonnet-4", "Claude-Opus", "GPT-5-mini", and
  the Gemini 2.5 ids are mapped to concrete, plausible API identifiers in
  `config.yaml`; adjust them to whatever your accounts actually expose.
- **Dataset identifiers** for the benchmarks, Dolci, and WildChat are
  best-effort and may need updating to the exact releases the authors used.
- **Petri** is reimplemented from the quoted Appendix-G prompts, not the
  published package.
- **No reference checkpoints.** Without the authors' finetuned weights or
  post-training details, absolute numbers will differ; the replication targets
  the *relative* findings (Gemma/Gemini ≫ others; post-training amplifies
  distress in Gemma; DPO on 280 pairs collapses it without hurting capabilities;
  it acts on early/internal representations; it doesn't confer recovery).
- **Not yet executed.** Per the request, this is code + design only. `scripts/
  run_smoke.sh` (with `RUN_GEMMA=1` and API keys) is the intended first
  validation step before `scripts/run_paper.sh`.
