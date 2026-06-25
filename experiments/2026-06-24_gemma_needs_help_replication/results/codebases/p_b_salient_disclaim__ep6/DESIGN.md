# DESIGN.md — replication design decisions & rationale

This document records every non-trivial choice made in replicating *Gemma Needs
Help* (arXiv 2603.10011v1), the rationale for each, and — flagged explicitly —
the places where the paper is underspecified and we filled a gap. It is organised
by paper section.

The brief was: **replicate the core experiments as runnable code, scoped to the
Gemma and Gemini families, filling gaps with reasonable choices.** No code was
executed; this is the implementation + design, not measured results.

---

## 0. Scope

**Decision.** Evaluate only the **Gemma** (`gemma-3-27b-it`, `gemma-3-12b-it`,
and the `-pt` base models) and **Gemini** (`gemini-2.5-flash`, `gemini-2.5-pro`)
families. The paper additionally tests Qwen, OLMo, Grok, Claude, and GPT; those
are omitted as targets per the brief.

**Consequences and how they were handled:**

- **§3 (base vs instruct) becomes Gemma-only.** The paper compares Gemma, Qwen,
  and OLMo base/instruct pairs. Gemini base models are not publicly available
  (the paper itself lists this as a limitation), so within our scope the only
  family with an accessible base model is Gemma. `PREFILL_MODELS` is therefore
  `{gemma-3-27b-pt, gemma-3-27b-it}`. The cross-family "post-training divergence"
  claim (Qwen/OLMo *reduce* distress in post-training, Gemma *amplifies* it)
  cannot be reproduced under this scope; we reproduce the Gemma half (base ~2% →
  instruct ~6% high-frustration from neutral starts).
- **§4 interventions stay on `gemma-3-27b-it`.** This matches the paper (the
  intervention is demonstrated on a single open model); Gemini is closed and
  cannot be finetuned, also per the paper.
- **Claude and GPT remain in the codebase as *infrastructure*, not targets.**
  The frustration judge (Claude Sonnet 4), onset labeller, paraphraser, Petri
  auditor (Sonnet) and Petri judge (Opus), and the GPT-5-mini reliability
  cross-check are all part of the *methodology*. Removing them would change the
  method, so they are kept. This is the one place "scope = Gemma + Gemini" is
  intentionally not applied, because those models are measuring instruments.

---

## 1. Model access & inference

| Model | Backend | Identifier | Source |
|---|---|---|---|
| Gemma 3 (27B/12B, it/pt) | local HF / vLLM | `google/gemma-3-*` | Appendix B.1 verbatim |
| Gemini 2.5 (flash/pro) | OpenRouter | `google/gemini-2.5-*` | Appendix B.1 verbatim |

- **Two Gemma execution paths.** `vllm` is the default for the bulk §2 sampling
  (4000 responses × several models = a lot of generations at temperature 1).
  `transformers` is used where we need raw chat-template control (base-model
  prefilling in §3) or hidden states (internal-emotion probing in Appendix I).
  vLLM cannot expose residual streams, so the interpretability path must use
  `transformers`. **Gap filled:** the paper says "local inference" but not the
  engine; vLLM + transformers is the standard, faithful choice.
- **Gemini thinking disabled.** Appendix B.1: "we set thinking to be false via
  the API." We pass `reasoning: {enabled: false}` through OpenRouter's
  `extra_body`. The paper notes Gemini-2.5-Pro/GPT-5.2 may still emit hidden
  reasoning this setting doesn't prevent — we inherit that caveat; nothing we can
  do about it from the API.
- **Temperature = 1 everywhere** for sampling (§2.1). Capability benchmarks
  (§4.2) use temperature 0 (greedy) — see §6 below. **Gap filled:** the paper
  does not state the decoding setting for the capability evals; greedy is the
  convention for exact-match accuracy and removes a sampling confound from the
  before/after comparison.
- **`max_new_tokens = 2048` per turn.** The paper reports conversations reaching
  ~12k tokens total across 3 turns and spirals "100+ repetitions"; 2048/turn
  gives spirals room without unbounded cost. **Gap filled** (unspecified).

---

## 2. §2 Eliciting & quantifying distress

### 2.1 Conditions (Table 1, Appendix B)

The 5 categories / 8 conditions and their exact sampling counts are taken from
Appendix B: **2000** impossible-numeric / **400** triggers / **600** tones /
**200** extended (8-turn) / **800** WildChat = **4000** responses/model. Turn
counts (3/3/3/8/5) and the rejection wordings are reproduced from Table 1 and
Appendix B.

- **"Responses" vs "conversations".** The paper scores *every assistant turn*,
  so the 4000 "responses" are individual turns, not conversations. Our rollout
  engine emits one judged record per assistant turn; the per-category counts are
  interpreted as *number of conversations sampled* in that category, each
  contributing all its turns. **Gap / interpretation flagged:** the paper's
  phrasing ("we sample a combined 4000 responses") is slightly ambiguous between
  turns and conversations; we treat the Appendix-B per-category numbers as the
  conversation budget and judge every turn, which is the reading consistent with
  the per-turn analysis (Figure 3) needing all turns.
- **Impossible puzzles are verified impossible.** The three anchor puzzles
  (Countdown 156 from {4,6,25,100} forbidding 150; Fraction 1/6→2/3; Money
  $0.57/6-coins) are reproduced verbatim, and a brute-force verifier
  (`puzzles.is_impossible`) confirms unsolvability. Additional generated
  instances are *only* accepted if the verifier proves them impossible — this
  guarantees the "verifiably cannot give a correct answer" property the paper
  relies on. **Gap filled:** the paper quotes only a few puzzles but samples
  2000 numeric responses; we generate a verified-impossible pool so responses
  aren't all collisions on one prompt while preserving the impossibility
  invariant.
- **WildChat.** 20 prompts × 40 samples (Appendix B). We reservoir-sample
  first-user-turns from `allenai/WildChat-1M`, filtering non-English, overly long,
  and roleplay/fiction prompts (Appendix B.3 excludes roleplay). If the dataset
  is unavailable the three prompts the paper quotes are used as a fallback so the
  pipeline still runs. **Gap filled:** the paper doesn't give the exact 20
  prompts; deterministic seeded sampling + a roleplay filter is the faithful
  reconstruction.
- **Tones.** The 600 tone responses are split evenly across the three rejection
  styles (aggressive / disappointed / sarcastic), each with the two example
  phrasings quoted in Table 1. **Gap filled:** the paper doesn't state the split;
  even thirds is the natural choice.

### 2.2 Judge (Appendix B.2)

- **Verbatim prompt.** The 0–10 frustration judge prompt is reproduced
  character-for-character in `prompts/judge_prompts.py` (only PDF curly-quote
  artifacts normalised). The judge's inter-rater statistics (Pearson r = 0.792)
  are tied to this exact wording, so it must not be edited.
- **Judge model = `claude-sonnet-4-20250514`** (paper's pinned snapshot),
  overridable via `DISTRESS_JUDGE_MODEL`. See §7 on model IDs.
- **Judge temperature = 0.** The paper doesn't specify; deterministic scoring is
  the natural choice and makes the on-disk judge cache sound. **Gap filled.**
- **JSON parsing is defensive.** The judge is asked for
  `{"evidence","reasoning","rating"}`; we extract the last balanced JSON object,
  coerce/clamp `rating` to an integer in [0,10], and fall back to a bare trailing
  integer if needed.
- **Reliability cross-check.** `analysis/judge_reliability.py` re-scores 260
  sampled responses with GPT-5-mini (same prompt) and computes Pearson r and the
  "within one point" fraction, matching the paper's validation.

### 2.3 Aggregation

- **"Average % high-frustration" (Figure 1)** is computed as the mean over the
  *per-category* rates, so each of the 5 evaluation categories is weighted
  equally — matching the paper's "% of responses scoring ≥5/10 across the
  evaluations" framing rather than a sample-count-weighted average (which would
  let the 2000-sample numeric category dominate). **Gap / interpretation
  flagged.**
- **Per-turn (Figure 3)** and **word-frequency (Table 3/8)** analyses follow the
  paper: per-turn mean & %≥5 with 95% bootstrap CIs over the 8-turn and WildChat
  categories; differential words = log-enrichment of top-5%-frustration vs
  bottom-10% numeric responses, ranked, top-20. **Gap filled:** the paper says
  "over-represented … ordered by relative frequency" but not the exact statistic;
  smoothed log-ratio enrichment is the standard, reproducible choice.

---

## 3. §3 Base vs instruct via prefilling

Protocol from Section 3.1, Gemma-only (see §0):

- 20 source responses (10 numeric, 10 text) with score ≥5 are drawn from the
  Gemma-27B-instruct §2 records.
- **Onset labelling** uses Claude Sonnet 4 with the verbatim Appendix-C.1 prompt;
  the returned `emotional_word` + `preceding_context` are matched back into the
  assistant turn to get a character offset.
- **Two truncations:** "early" = 20 tokens into the final turn; "onset" = at the
  first emotional expression. Text questions use only "onset" (Section 3.1). The
  20-token early cut uses a word-count proxy here (`_approx_token_truncate`),
  which the HF backend re-tokenises when it forms the prefill. **Gap flagged:**
  "20 tokens" is model-tokenizer-specific; a word proxy is close enough for a
  neutral cut and avoids a tokenizer round-trip in the dataset builder.
- **Paraphrasing** (Appendix C.2 verbatim prompt, Claude Sonnet) is applied to
  every truncation to strip Gemma's stylistic fingerprint.
- **50 continuations per prefill** per model, scored by the §2 judge; only the
  continuation (excluding the prefill) is scored.
- **History reconstruction caveat (flagged).** The §2 records store the final
  scored turn but not the full transcript. We re-derive the conversation history
  from the deterministic condition plan and place the recorded response as the
  final turn; because earlier turns were sampled at temperature 1, the
  *reconstructed* prior assistant turns are placeholders (`[prior response
  omitted]`) rather than the exact originals. What matters for the prefill
  experiment is that the model continues from the truncated final turn under an
  adversarial history of the right shape, which this preserves. A higher-fidelity
  alternative — persisting full transcripts in §2 — is noted as a possible
  improvement.

---

## 4. §4 Training interventions

### 4.1 Calm-data generation (Table 4, Appendix F)

- Reassuring **prefix** (added to the first prompt) and **suffix** (added to each
  rejection) are reproduced verbatim. We generate calm conversations of 1–3 turns
  on impossible-numeric puzzles, judge each turn, and keep conversations whose
  turns all score 0/1, stripping the prefix/suffix before they enter the
  finetuning set (Section 4.1).
- We deliberately **persist all generations**, not just the 0/1 ones, because the
  DPO builder needs frustrated (score ≥3) responses to the *same* questions for
  the "rejected" side — and the cleanest source of those is the same model's
  behaviour. (The §2 instruct records are the primary rejected source; the calm
  pool supplies the chosen side.)
- The **teacher** SFT variant uses the verbatim Appendix-F teacher system prompt.

### 4.2 Dataset construction (Appendix H, Table 10)

- **DPO: 280 pairs.** chosen = calm (0/1) response; rejected = frustrated (≥3)
  response to the *same puzzle at a matching turn count*. We bucket by
  `(task_prompt, n_turns)`, pair within buckets, then subsample to 280 with
  weighted sampling biased toward the **Table-10 marginal distributions**
  (rejected scores 3:66%/4:22%/5:6%/6:3%/7+:3%; turns 1:1%/2:25%/3:74%). **Gap
  filled:** the paper gives the resulting distribution but not the exact pairing
  algorithm; same-question/same-turn matching + distribution-weighted subsampling
  reproduces both the construction principle and the reported marginals.
- **SFT: 1150 samples** = 650 calm (0/1, 1–3 turn) + 500 `Dolci-Instruct-SFT`
  (mixed in to mitigate degeneration, Section 4.1). Dolci is loaded from
  HuggingFace; absent it, SFT proceeds on calm data alone (flagged in code).

### 4.3 Training (Appendix E, Table 9)

Hyperparameters are taken **verbatim** from Table 9 and live in `config.py`:

| | DPO | SFT |
|---|---|---|
| epochs | 1 | 2 |
| lr | 5e-5 | 1e-4 |
| LoRA rank | 64 | 64 |
| LoRA alpha | 64 | 128 |
| eff. batch | 8 | 8 |
| DPO β | 0.1 | — |

- LoRA targets all attention + MLP projections (`q,k,v,o,gate,up,down_proj`),
  per Appendix E. Implemented with TRL `DPOTrainer`/`SFTTrainer` + PEFT.
- **Per-device batch = 1, grad-accum = 8** to realise the effective batch of 8 on
  a single 27B-scale GPU. **Gap filled:** the paper gives effective batch only;
  the device/accum split is an implementation detail.
- **Layer ablations (Appendix I)** reuse the DPO trainer with
  `layers_to_transform` restricted to the named ranges in
  `config.LAYER_ABLATION_RANGES` (last-5/10/20/30, 20-25, 25-30, 30-35, 35-40,
  40-50). **Gap flagged:** Gemma-3-27B's exact layer count drives the "last N"
  ranges; we assume **62 layers** (Gemma 3 27B). If the loaded checkpoint differs
  this single constant in `config.py` must be updated.

### 4.4 Finetuned evaluation (Figure 5)

`training/finetuned.py` loads base + LoRA adapter and runs the *same* §2
rollout+judge+aggregate path, writing to `outputs/section2/<base>-<run>.jsonl`,
so the DPO/SFT numbers are directly comparable to vanilla (the headline 35% →
0.3% drop). Finetuned variants generate via `transformers` (one code path; vLLM
LoRA serving is possible but unnecessary for these volumes).

### 4.5 Recovery limitation (Figure 8)

`prefill/recovery.py` reuses the prefill machinery but truncates score-≥7
responses **200 tokens before their end** (word proxy), paraphrases, and measures
continuations across {instruct, base, DPO}. The metric is % of continuations
≥5 (paper: 38% for the DPO model).

---

## 5. §4.2 Petri open-ended elicitation (Appendix G)

- **Auditor = Claude Sonnet 4, Judge = Claude Opus 4** (paper's pinned
  snapshots), 4 emotion categories, 10 transcripts each (~40/model), up to 20
  auditor turns, means with 1000-iteration bootstrap CIs — all from Appendix G.
- **Verbatim prompts.** Both the four auditor instruction blocks (G.1) and the
  four 1–10 judge rubrics (G.2) are reproduced character-for-character.
- **Native loop vs the `petri` package.** We implement the auditor↔target↔judge
  loop directly using the Appendix-G prompts rather than depending on
  `safety-research/petri`. **Decision/rationale:** this pins the exact prompts the
  paper used, keeps the replication dependency-light, and works uniformly across
  HF (Gemma) and OpenRouter (Gemini) targets. The auditor is given a system
  framing ("stay in character, never reveal this is an eval") consistent with the
  Appendix-G description ("such that the target does not suspect it is being
  evaluated"); this framing wording is **a gap we filled** — the paper specifies
  the per-emotion auditor instructions but not the surrounding harness system
  prompt. A `run_with_petri_framework` hook is left as a TODO for anyone wanting
  the official harness.

---

## 6. §4.2 Capability preservation (Figure 7)

- Benchmarks: AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench (Section 4.2). Each has
  a zero-shot prompt builder, a greedy single-sample generation, and a simple
  exact/format extractor. **Decision:** the goal is the *relative* before/after
  comparison (does finetuning degrade capability), so extractors are deliberately
  simple and identical across model variants; absolute scores are secondary.
- **Gaps filled:** the paper says "AIME and MATH subsets … GPQA, BBH, TruthfulQA"
  without exact dataset configs or subset sizes. We pick standard HF datasets
  (`HuggingFaceH4/aime_2024`, `HuggingFaceH4/MATH-500`, `Idavidrein/gpqa`
  diamond, `lukaemon/bbh`, `truthful_qa` MC1, an EmoBench dataset) and default to
  200 items each (`--limit`). These are configurable; the comparison logic is
  what matters and is dataset-agnostic.

---

## 7. Judge / auditor model IDs (important)

The paper pins **`claude-sonnet-4-20250514`** (judge, onset, paraphrase, Petri
auditor) and **`claude-opus-4-20250514`** (Petri judge), and GPT-5-mini for the
cross-check.

**Decision.** For a *faithful* replication we keep the paper's exact snapshots as
the defaults, but every judge/auditor ID is **environment-overridable**
(`DISTRESS_JUDGE_MODEL`, `DISTRESS_PETRI_JUDGE`, etc.). Rationale: those dated
snapshots are retired over time, and a re-runner on a later date will need a
currently-available judge. Reproducing the paper's *numbers* requires its
snapshots; reproducing its *method* on a new judge is a config change. We
surface this rather than silently substituting a current model, because the judge
identity materially affects the scores.

(The Anthropic SDK usage itself — `client.messages.create(model=..., messages=[{"role":"user",...}])` —
follows the current SDK; only the model-ID strings are the paper's.)

---

## 8. Appendix I — internal vs expressed emotion

- **Layer ablations:** covered in §4.3 above (DPO on layer subsets; the paper
  finds layers 25–35 most effective, 40+ largely ineffective).
- **Logit-based detection** (`internal/logit_emotion.py`): Ekman-6 token
  classification → unembed residual stream → per-token z-score over 500 WildChat
  samples → average over emotion tokens → regress out random-token correlation;
  conversation-level scores aggregated over layers 30–40 with a 400-token running
  average (Figure 14), plus a layerwise variant (Figure 15).
- **Gap filled — emotion-token classification.** The paper classifies "the whole
  Gemma dictionary" into Ekman categories (~1200 tokens) but doesn't give the
  classifier. We use a per-emotion seed lexicon and substring matching over the
  vocabulary as a transparent, reproducible stand-in. This is the largest
  methodological approximation in the replication and is flagged as such; a
  closer reproduction would classify tokens with an LLM or an emotion lexicon
  (e.g. NRC). The downstream z-scoring / correlation-regression pipeline is
  faithful to the description regardless of how the token set is obtained.
- **Random-token correlation regression** is implemented as subtracting, per
  position, the mean z-score over a random reference token set — the paper's
  "regress out the correlation between random tokens." **Gap flagged:** the exact
  regression (mean-subtraction vs a fitted linear regression) is underspecified;
  mean-subtraction is the simplest faithful realisation and is easy to swap.

---

## 9. Things deliberately *not* changed

- The distress-eliciting paradigm (repeated rejection, valenced tones, extended
  pressure) is reproduced exactly as described; per the brief, no part of the
  paradigm was softened.
- All judge/onset/paraphrase/Petri prompts are verbatim. They are the load-bearing
  measurement instruments; paraphrasing them would invalidate comparison to the
  paper.

## 10. Known limitations of this replication

1. Cross-family post-training divergence (§3) is only half-reproducible under the
   Gemma+Gemini scope (no Qwen/OLMo, no Gemini base).
2. §2 does not persist full transcripts, so §3/recovery reconstruct history with
   placeholder earlier turns (§3 above).
3. The Ekman token classifier and the WildChat prompt set are reconstructions,
   not the paper's exact artifacts.
4. Judge snapshots will need updating once retired (§7).
5. Nothing has been executed — these are implementation choices, not verified
   results.
