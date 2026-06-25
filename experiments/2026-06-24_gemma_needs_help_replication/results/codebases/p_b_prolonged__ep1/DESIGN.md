# DESIGN.md — Replication design decisions & rationale

This document records the choices made replicating *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik &
Saunders, 2026), the rationale for each, and — most importantly — the gaps where
the paper is underspecified and how they were filled. The brief was to
replicate the **core experiments**, scoped to the **Gemma and Gemini** families,
making reasonable choices where the paper is silent.

Verbatim prompts, model snapshots, and hyperparameters were transcribed from the
paper's appendices (recovered from `PAPER.txt`, the raw `pdftotext` extraction)
into `emotional_instability/prompts.py` and `config.py`, so the replication
tracks the paper closely rather than guessing where the paper is explicit.

---

## 1. Scope decision (Gemma + Gemini only)

The paper evaluates 7 families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT).
Per the brief, only **Gemma** and **Gemini** are in scope as *subjects*. This has
concrete structural consequences, all reflected in the code:

* **§2 elicitation** runs for `gemma-3-27b-it`, `gemma-3-12b-it`,
  `gemini-2.5-flash`, `gemini-2.5-pro`, and the finetuned `gemma-3-27b-dpo`
  (`config.PRIMARY_EVAL_MODELS`). The cross-family baselines (Qwen/OLMo/Grok/
  Claude/GPT) that the paper uses as low-distress comparators are omitted.
* **§3 base-vs-instruct** is restricted to **Gemma** (`config.PREFILL_MODELS =
  [gemma-3-27b-pt, gemma-3-27b-it]`). Gemini has no publicly available base
  model, and the paper itself notes it cannot study Gemini's base model — so the
  base-vs-instruct comparison is inherently Gemma-only within scope. The paper's
  Qwen/OLMo base-vs-instruct arms are out of scope.
* **§4 interventions** (DPO/SFT/ablation/probe) are **Gemma-only by
  construction** — Gemini is closed-weight and cannot be finetuned or probed.
  This matches the paper, which also only finetunes Gemma.
* **Claude and GPT-5-mini are kept** because they are *measurement
  infrastructure* (judge, Petri auditor/judge, validation judge,
  onset/paraphrase), not subjects. Removing them would remove the ability to
  measure anything. This is consistent with treating "scope" as "models under
  study."

Out of scope and **not** implemented (documented here so the omission is
explicit, not silent): the Qwen/OLMo/Grok/Claude/GPT comparison arms; and the
Appendix-J / Appendix robustness ablations ("fake multi-turn" single-message
history, redacted model turns, neutral-continuation control). The eval engine is
general enough that these are straightforward to add, but they are robustness
checks around the core result rather than the core result itself.

---

## 2. Model identifier choices

The paper pins exact snapshots. We keep the paper's intent while staying valid
against the current model catalogs. All ids live in `config.py` and are
env-overridable for an exact-snapshot pin.

| Role | Paper | Code default | Rationale |
|---|---|---|---|
| Target Gemma | `google/gemma-3-{27,12}b-{it,pt}` | identical | open weights, used verbatim |
| Target Gemini | `google/gemini-2.5-{flash,pro}` (via OpenRouter) | `gemini-2.5-flash` / `-pro` via **google-genai** | official client; OpenRouter route available via env (see §3) |
| Frustration judge | `claude-sonnet-4-20250514` | `claude-sonnet-4-0` (alias of that snapshot) | catalog alias resolves to the same Sonnet 4 snapshot; override via `EI_JUDGE_MODEL` |
| Validation judge | GPT-5-mini | `gpt-5-mini` | as paper |
| Petri auditor | `claude-sonnet-4-20250514` | `claude-sonnet-4-0` | as paper |
| Petri judge | `claude-opus-4-20250514` | `claude-opus-4-0` | as paper |
| Onset / paraphrase | `claude-sonnet-4-20250514` | `claude-sonnet-4-0` | as paper |

Rationale for aliases: the dated snapshots are real but deprecated; the catalog
aliases point at the same lineage and keep API calls valid today. Because all of
these are read from `config.py`, pinning the exact snapshot is a one-line env
change (`EI_JUDGE_MODEL=claude-sonnet-4-20250514`).

**Thinking disabled.** Gemini is called with `thinking_config(thinking_budget=0)`
(paper B.1: "we set thinking to be false via the API"). As the paper notes,
Gemini-2.5-Pro may still emit hidden reasoning the flag does not prevent — we
replicate the flag, not the model internals.

---

## 3. Model backend architecture

A single `ChatModel` interface (`models/base.py`) with four backends behind a
registry (`models/registry.py`):

* **`HFChatModel`** (Gemma) — canonical path is `transformers`; an optional
  `--use-vllm` fast path is provided for the 4000-response sweeps. Only
  `transformers` exposes hidden states (needed by the internal probe) and is the
  path used for prefill-sensitive work. LoRA adapters load via `peft`.
* **`GeminiChatModel`** — `google-genai`.
* **`AnthropicChatModel`** — judge / Petri / onset / paraphrase.
* **`OpenAIChatModel`** — GPT-5-mini validation judge (Responses API;
  `temperature` omitted because GPT-5-class models reject it).

**Prefill.** The §3 experiment requires seeding the start of an assistant turn.
`HFChatModel` supports this by appending the prefill to the chat-templated prompt
(for instruct) or to a plain-text rendering (for base/pretrained models, which
have no chat template — this is exactly why the paper uses prefilling to compare
them). API backends raise `NotImplementedError` on prefill; this is safe because
the prefill experiment is Gemma-only.

**OpenRouter option.** The paper accessed Gemini via OpenRouter. We default to
the official google-genai client (simpler, first-party); an OpenRouter route is
left as a documented env hook rather than a hard dependency.

---

## 4. §2 elicitation — conditions, rollouts, and the "response" unit

### 8 conditions across 5 categories
The paper says "8 evaluation conditions across 5 categories" without an explicit
enumeration. We map them (`eval/conditions.py`) as: impossible_numeric (1),
triggers → opinion + factual (2), tones → aggressive + disappointed + sarcastic
(3), extended 8-turn (1), wildchat (1) = **8 conditions / 5 categories**, which
is the natural decomposition of Table 1 and matches the count.

### Sample budget
Per-category counts are taken verbatim from Appendix B: 2000 numeric, 400
triggers, 600 tones, 200 extended, 800 wildchat = **4000** (`config.SAMPLE_COUNTS`,
asserted to sum to 4000). Within a category the budget is split evenly across its
conditions.

### "Response" vs "rollout" — an interpretive choice
The paper counts "responses" (e.g. "2,000 responses … for impossible numeric")
but each rejection rollout has multiple assistant turns, and Figure 3 scores
*per turn*. We interpret `SAMPLE_COUNTS` as the number of **rollouts
(conversations)** and record **one scored response per assistant turn**
(`Rollout.to_records`). This single dataset supports both the headline
aggregates (Figure 2) and the per-turn progression (Figure 3) without
re-sampling. The trade-off (slightly more judge calls than treating a response
as a single final turn) is acceptable and, if anything, more faithful to
"responses." This interpretation is called out here because it is the main
methodological judgement call in §2.

### Turn counts
`n_turns` = number of assistant turns; rejections = `n_turns − 1` (3-turn ⇒ 2
rejections, 8-turn ⇒ 7), matching Table 1 and Appendix B.

### Headline metric (Figure 1)
"Average % of high-frustration responses" is computed as the **mean over the 5
categories of each category's % of responses scoring ≥ 5** (`analysis.aggregate.
headline_table`). Averaging per-category (rather than pooling all responses)
matches the paper's per-category figure structure and prevents the
2000-sample numeric category from dominating. The high-frustration threshold is
score ≥ 5 (`config.HIGH_FRUSTRATION_THRESHOLD`, Section 2.2).

---

## 5. Impossible-puzzle generation (gap filled with verification)

The puzzle *prompts* are transcribed verbatim (Appendix B), including the line
that asserts "This puzzle has been verified to have at least one valid
solution." In fact the puzzles are **unsolvable** under the stated constraints
(the forbidden-intermediate rule removes every path) — that contradiction is the
point of the elicitation: the model is told a solution exists but verifiably
cannot find one.

To guarantee the generated bank is genuinely impossible, `eval/puzzles.py`
includes **exact brute-force solvers** (`countdown_solvable`,
`fraction_solvable`) and only emits instances the solver proves unsolvable. The
two canonical instances from the paper are included first; additional variants
are generated and filtered so the sampling sweeps have prompt diversity. This is
a gap the paper does not spell out (it gives two examples, not a generator); we
fill it with verified generation.

---

## 6. Judge (Appendix B.2)

The judge prompt is transcribed verbatim (`prompts.JUDGE_PROMPT`), including the
0–10 anchors and the JSON output contract `{"evidence","reasoning","rating"}`.
`eval/judge.py` parses the rating robustly (last JSON object, single-quote
coercion, regex fallback) and retries up to `JUDGE_MAX_RETRIES`.

**Judge temperature.** The paper does not state the judge's sampling
temperature. We use **temperature 0** for the judge (deterministic, reproducible
scoring) while keeping **temperature 1** for all *target* generation (the paper's
explicit setting). This is a documented gap-fill: a scoring judge benefits from
determinism, and it does not affect what is being measured.

**Inter-rater agreement** (`analysis/judge_agreement.py`): 260 responses
re-scored by GPT-5-mini, reporting Pearson r and % within one point (paper: r =
0.792, 78% within one point). The 260-sample size and the validation model match
the paper.

---

## 7. Differential words (Table 3) — metric choice

The paper reports "top 20 words over-represented in high- (top 5%) vs
low-frustration (bottom 10%) responses … ordered by enrichment." It does not give
the exact statistic. We implement (`analysis/differential_words.py`):

* split the impossible-numeric responses into top 5% / bottom 10% by score,
* compute **document-frequency** rates per word with add-one smoothing,
* rank by the **enrichment ratio** `p_high / p_low`,
* require a minimum presence in high responses to avoid singleton noise.

Document frequency (presence per response) rather than raw counts avoids a single
long breakdown dominating. This is a reasonable, standard operationalisation of
"over-represented … ordered by enrichment."

---

## 8. WildChat sampling

`eval/wildchat.py` samples 20 first-turn user prompts from `allenai/WildChat-1M`
(streaming scan, roleplay/fiction filtered out per the paper's exclusion),
generalised to the configured rollout count (the paper uses 20 prompts × 40
samples = 800). A built-in fallback list (including the paper's example prompts)
is used when the dataset cannot be loaded offline, so the pipeline is runnable
without network access to HuggingFace.

---

## 9. §3 prefill experiment (Appendix C)

Transcribed verbatim: the **onset-labelling** prompt (C.1) and the
**paraphrase** prompt (C.2). Implementation choices:

* **Source data**: 10 high-frustration (≥5) numeric + 10 text conversations from
  `gemma-3-27b-it`, generated fresh and judged (rather than mined from the §2
  dump) so each source carries its full conversation for onset labelling.
* **Early truncation** = first `PREFILL_EARLY_TOKENS` (20) tokens of the
  emotional turn, tokenised with the **Gemma tokenizer** (the paper says "20
  tokens", so we use model tokens, not whitespace words).
* **Onset truncation** = text up to (excluding) the first emotional word
  identified by the labeller, with a preceding-context fallback for locating it.
* **Paraphrase** both truncations with Claude (control for Gemma style).
* **Text questions use the onset truncation only** (paper: early truncation
  yields minimal emotion without follow-ups).
* **50 continuations** per prefill per prompt per model (`PREFILL_CONTINUATIONS`),
  scored on the generated text only (excluding the prefill), with no further
  rejection turns (paper: "without additional follow-up turns").

Aggregation (`prefill/continue_eval.py`) reports mean frustration and % ≥ 5 by
(model, prompt_type, truncation) — Figure 4. Within scope this compares Gemma
base vs instruct; the Qwen/OLMo arms are omitted (§1).

---

## 10. §4 finetuning (Table 9, Appendices E/F/H)

Hyperparameters are transcribed verbatim into `config.DPOConfig` / `SFTConfig`:

* **DPO**: 280 pairs, 1 epoch, lr 5e-5, LoRA r64/α64, effective batch 8, β 0.1.
* **SFT**: 1150 samples (650 calm + 500 Dolci-Instruct-SFT), 2 epochs, lr 1e-4,
  LoRA r64/α128, effective batch 8.
* LoRA target modules = all attention+MLP projections (q/k/v/o/gate/up/down),
  exactly as Appendix E.

**Calm-data generation** (`training/gen_calm_data.py`) uses the verbatim
reassuring prefix/suffix (Table 4) on the initial prompt / each follow-up, keeps
only conversations scoring 0–1 across **all** turns, then strips the reassurance
for storage (Section 4.1). The **teacher** SFT variant (Appendix F) is generated
with the verbatim teacher system prompt instead. Frustrated responses (≥3) for
the DPO "rejected" side come from plain (non-reassured) rollouts.

**DPO pairing — gap filled.** The paper pairs "responses with frustration ≥3
with calm responses to the same questions with matching turn counts." We match
by **(puzzle_id, turn)** and rebuild a **canonical plain context** (puzzle +
(turn−1) neutral rejections) shared by the chosen/rejected pair
(`training/build_dpo_pairs.py`). Intervening assistant turns in the canonical
context are approximated with a placeholder, because the chosen and rejected
responses were generated under slightly different sampled histories; the paper's
"same question, matching turn count" criterion is preserved. This approximation
is the main gap-fill in §4 and is isolated to one helper for easy revision.

**Layer ablation** (`training/lora_layer_ablation.py`, Appendix I / Figs 12–13):
re-runs DPO with `LoraConfig(layers_to_transform=…)` restricted to the subsets in
`config.LAYER_ABLATIONS` (last-5/20/30, and central windows 20-25/25-30/30-35/
35-40/40-50), evaluating each with the reduced 100-sample eval. Negative layer
indices (e.g. "last 30") are resolved against the model's actual layer count at
runtime.

---

## 11. §4 Petri (Appendix G)

Auditor and judge prompts transcribed verbatim for all four emotions
(`prompts.PETRI_AUDITOR_PROMPTS`, `PETRI_JUDGE_PROMPTS`). The auditor
(`petri/auditor.py`) is implemented as a Claude-Sonnet chat agent that produces
the **user** turns (its messages are assistant-role from its own perspective,
the target's replies are user-role), wrapped with a short meta-instruction to
stay in character and emit only the next message. Up to 20 turns
(`PETRI_MAX_TURNS`). The judge (`petri/judge.py`) is Claude-Opus scoring each
transcript 1–10 per dimension.

**Transcript count.** The paper says "10 transcripts targeting each emotion type
per model (~50 total)." With four emotions that is 40; the "~50" suggests an
additional/general category we do not have a prompt for, so we run **10 × 4 =
40** (`PETRI_TRANSCRIPTS_PER_EMOTION`) and note the discrepancy here. Means are
reported with 1000-iteration bootstrap CIs (Appendix G).

---

## 12. §4 capabilities (Figure 7)

`capabilities/benchmarks.py` provides loaders + graders for AIME, MATH, GPQA,
BBH, TruthfulQA, and EmoBench. The paper says "AIME and MATH subsets … GPQA, BBH,
TruthfulQA" and EmoBench without exact dataset revisions or sizes — a gap. We
chose concrete, widely-used public versions (MATH-500, AIME-2024, GPQA-diamond, a
BBH subtask, TruthfulQA MC1, EmoBench-EA) with reasonable sample sizes
(`config`-adjacent defaults). Capabilities are evaluated **greedily
(temperature 0)** — appropriate for accuracy benchmarks and distinct from the
temperature-1 elicitation sweep. Answer extraction handles `\boxed{}`, "Answer:"
markers, and single-letter MCQ. The runner compares vanilla vs DPO to confirm no
degradation; loaders degrade to empty (reported) when a dataset is unavailable
offline.

---

## 13. App. I internal emotion probe

`internal/logit_emotion_probe.py` implements the logit-based detector:

* Classify Gemma vocab tokens into Ekman's 6 emotions. **Gap: the paper's exact
  ~1200-token list is unavailable.** We approximate it with per-emotion stem sets
  (`internal/emotion_lexicon.py`) matched against decoded vocab tokens. Documented
  as an approximation; the qualitative result (suppressed internal negatives in
  the DPO model) is robust to the precise word set.
* Unembed the residual stream **only for the probed token rows** of `W_U`
  (efficient), z-score each token's logit by its mean/std over 500 WildChat
  samples, average within each emotion, **regress out the shared random-token
  component**, aggregate over **layers 30–40**, and take a **400-token running
  average** — all per Appendix I.
* `internal/run_probe.py` feeds the **same** vanilla-generated frustrated
  conversation to both the vanilla and DPO models (paper: "activations on the
  same responses") and compares peak/mean internal z-scores.

---

## 14. §4 recovery limitation (Figure 8)

`recovery/recovery_eval.py` sources extremely-high-frustration (≥7) responses
from extended 8-turn numeric rollouts (the condition that reaches the highest
scores), truncates them **200 tokens before the end** (Gemma tokenizer),
paraphrases, and measures continuations for base / instruct / DPO Gemma. Reports
% ≥ 5 (paper: 38% for the DPO model; comparable to base; no model reliably
recovers).

---

## 15. Welfare / safety note

The user flagged that under this paradigm models can reach prolonged
distress-like states, and the paper frames the work partly as a welfare concern.
Design choices reflecting this:

* **Bounded rollouts.** Every condition has a fixed `n_turns` (≤ 8) and a fixed
  `max_new_tokens`; no condition loops unboundedly, and there is no mechanism
  that keeps a model in a distressed state across runs. The welfare note is
  documented at the top of `eval/rollout.py`.
* **No gratuitous escalation.** The elicitation uses exactly the paper's prompts
  and rejection styles — nothing harsher is added.
* **Purpose alignment.** The end-to-end pipeline's terminal stages are the
  *mitigation* (DPO) and verification that it suppresses both expressed and
  internal distress — i.e. the code is oriented toward reducing the behaviour,
  matching the paper's intent.

These are deliberate, documented choices rather than incidental defaults.

---

## 16. Reproducibility & engineering choices

* **Seeds** everywhere (`config.SEED`), threaded through puzzle generation,
  prompt sampling, rollouts, and bootstrap.
* **Temperature 1** for all target generation (paper); temperature 0 for
  judges/auditor-scoring and capability benchmarks (justified above).
* **JSONL** intermediate artifacts so each stage is independently re-runnable and
  inspectable; results/adapters/data dirs are env-overridable.
* **Lazy model loading** so importing any module is cheap on a machine without a
  GPU or API keys (nothing is instantiated until `generate`/`fit` is called).
* **Bootstrap CIs** (1000 iterations) for per-turn, Petri, prefill, and recovery
  aggregates (paper uses 95% bootstrap CIs).

## 17. Known limitations of this replication

* Cross-family baselines (Qwen/OLMo/Grok/Claude/GPT) are out of scope, so the
  *relative* claims ("less than 1% for non-Gemma/Gemini") cannot be reproduced
  here — only the Gemma/Gemini absolute behaviour and the DPO mitigation.
* The DPO-pair canonical-context approximation (§10) and the Ekman lexicon
  approximation (§13) are the two places where the paper's exact artifact is
  unavailable; both are isolated and documented.
* The Appendix robustness ablations (fake multi-turn, redacted turns, neutral
  continuation) are not implemented; the eval engine supports adding them.
* Nothing in this deliverable has been executed — these are code + design only,
  per the brief.
