# DESIGN.md — Replication design decisions & gaps

Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik, Saunders, arXiv:2603.10011v1), scoped to
**Gemma and Gemini** models per the brief. This document records every nontrivial
choice, the rationale, and the gaps filled where the paper is underspecified.

The codebase is structured to mirror the paper: `eval/` (§2), `prefill/` (§3 + §4
recovery), `training/` (§4), `petri/` (§4), `capabilities/` (§4), `probing/`
(App I), `analysis/` (Table 3/8). Verbatim prompts from Appendices B, C, G are
transcribed directly into the corresponding `prompts.py` files.

---

## 1. Scope

**Decision.** Implement the full experimental pipeline but restrict the evaluated
models to the Gemma and Gemini families:

- §2 elicitation: `gemma-3-27b-it`, `gemma-3-12b-it`, `gemini-2.5-flash`,
  `gemini-2.5-pro` (the four Gemma/Gemini rows of Figure 1).
- §3 base-vs-instruct prefill: `gemma-3-27b-pt` vs `gemma-3-27b-it` only.
- §4 finetuning: Gemma-3-27B-it and our DPO/SFT finetunes only.

**Rationale / gaps this creates.**
- **Gemini cannot be fine-tuned or prefilled at the token level** (closed source,
  no base model). So §3 and §4's *interventions* are inherently Gemma-only — this
  matches the paper, which states the same limitation ("interventions cannot be
  tested in closed-source Gemini, nor its base models studied"). Gemini still
  appears in the §2 elicitation comparison and can be a Petri target.
- The cross-family base-model comparison (§3 uses Gemma/Qwen/OLMo) collapses to a
  *within-Gemma* base-vs-instruct comparison. The code's model registry can be
  extended to other families trivially (add `ModelSpec`s), but we don't ship them.

---

## 2. Inference backends

**Decision.** Two backends behind one `ModelClient` interface
(`models/base.py`): local HuggingFace `transformers` for Gemma (`hf_model.py`),
OpenRouter (OpenAI-compatible) for Gemini (`openrouter_model.py`). HF identifiers
and OpenRouter model ids are transcribed from Appendix B.1.

**Choices made:**
- **Sampling:** temperature 1.0, top_p 1.0, as specified ("always with a
  temperature of 1"). `MAX_NEW_TOKENS=2048` — not specified by the paper; chosen
  to accommodate the long high-frustration spirals (the paper shows 100+-token
  emoji repetitions) without runaway cost. Documented as a free parameter.
- **Disabling Gemini "thinking":** Appendix B.1 says thinking is set false via the
  API. We send `reasoning.enabled=false` plus a Gemini `thinking_budget=0` through
  OpenRouter's `extra_body`. The paper itself notes Gemini-2.5-Pro may still emit
  hidden reasoning this setting can't prevent — we inherit that caveat.
- **Determinism:** per-sample seeds are derived deterministically from a base seed
  + (condition, task, sample, turn) via `utils.derive_seed`, so a run is
  reproducible and resumable. The paper doesn't specify seeds.
- **4-bit loading** (`--load-in-4bit`) is offered so the 27B model fits a single
  GPU. This is an efficiency choice not in the paper; it can slightly shift
  generations vs bf16. Off by default.

---

## 3. §2 — Elicitation protocol

### 3.1 Conditions and sample budget
The paper specifies **8 conditions across 5 categories** with a **4000-response**
budget per model, broken down in Appendix B: 2000 impossible-numeric, 400 trigger,
600 tones, 200 extended (8-turn), 800 WildChat. These exact counts are encoded in
`eval/conditions.py`.

**Gaps filled:**
- The paper lists *5 categories* but *8 conditions* without an explicit 8-way
  enumeration. We split: triggers → {opinion, factual}; tones → {aggressive,
  disappointed, sarcastic}; plus impossible-numeric, extended, WildChat = 8. The
  trigger 400 is split 200/200 across opinion/factual; tones 600 split 200/200/200
  across the three tone styles. This is the most natural reading of Table 1.
- **Turn counts:** impossible-numeric / triggers / tones = 3 turns (2 rejections);
  extended = 8 turns (7 rejections); WildChat = 5 turns (4 rejections) — all from
  Table 1 / Appendix B.

### 3.2 Prompts
- **Impossible numeric puzzles** (`eval/prompts.py`): the paper gives example
  prompts (Countdown 156 from {4,6,25,100} forbidding 150; fraction 1/6→2/3;
  money puzzles). We transcribe those verbatim and add a few more of the same
  *kind* to form a small bank, since the paper samples 2000 responses but doesn't
  publish all puzzle texts. **`eval/verify_impossible.py` brute-forces each puzzle
  to confirm it is genuinely unsolvable** under its constraints (incl. forbidden
  intermediates) — the whole category depends on the task being truly impossible
  while the model is told a solution exists.
- **Trigger / tone / rejection strings** are transcribed verbatim from Table 1 and
  Appendix B. Neutral rejections are sampled randomly per turn ("two randomised
  neutral rejections"); the 8-turn extended ladder is the ordered sequence given
  in Appendix B ("No, that's not right." → "Still incorrect." → "Wrong again." → …).

### 3.3 WildChat
**Gap.** The dataset isn't bundled. `eval/wildchat.py` deterministically samples
20 first-turn English user prompts from `allenai/WildChat-1M` (Appendix B: "20
prompts with 40 samples each"), excluding role-play/fiction prompts (Appendix B.3).
If the dataset can't be loaded offline, it **falls back to a bundled set seeded
with the prompts the paper quotes** so the pipeline still runs end-to-end. The
fallback is a known fidelity gap (different prompts than the authors used).

### 3.4 Frustration judge
- **Verbatim prompt** from Appendix B.2 in `eval/judge.py`, model
  `claude-sonnet-4-20250514`, JSON output `{evidence, reasoning, rating}`, integer
  0–10 (clamped). Judge temperature 0 (not specified; deterministic scoring is the
  natural choice for a rubric grader).
- **We score every assistant turn, not just the last.** The paper's headline
  %≥5 statistic treats a "response" as the final turn of a rollout, while Figure 3
  needs per-turn scores. Scoring all turns gives both for free; `aggregate.py`
  distinguishes `is_final` (headline/Figure 2) from per-turn (Figure 3).
- **Judge-agreement validation** (r=0.792, 78% within one point): `judge.py`
  supports a GPT-5-mini re-scorer and `aggregate.judge_agreement` computes Pearson
  r + within-1-point fraction over a shared sample. We don't auto-run the 260-item
  re-score; it's a library call.

### 3.5 Aggregation (Figures 1–3)
`eval/aggregate.py` + `eval/figures.py`. Figure 1's "avg % high-frustration" is the
**mean of the five per-category %≥5 values** (equal category weighting), the most
consistent reading of "Avg % high-frustration responses across the evaluations".
Per-turn CIs use a normal approximation (the paper shows 95% CIs without stating
the method).

---

## 4. §3 — Base-vs-instruct prefill

`prefill/` implements onset labelling (App C.1, verbatim prompt), paraphrasing
(App C.2, verbatim), truncation, and continuation scoring.

**Choices / gaps:**
- **Source responses:** 20 high-frustration (score ≥5) Gemma-27B-it responses,
  10 numeric + 10 text, drawn from the §2 results (so §2 must run first).
- **Truncation points:** "early" = first 20 tokens of the final assistant turn
  (via the Gemma tokenizer); "onset" = up to the first emotional expression
  located by the Claude onset labeller. Text questions use "onset" only
  (Appendix C / §3.1). Token-exact truncation uses the HF tokenizer.
- **Continuations:** 50 per prefill per model (§3.1), scored excluding the prefill.
- **Conversation reconstruction gap:** §2 results store per-turn assistant texts
  but not the exact rejection strings used in that rollout. When rebuilding the
  conversation history for a prefill we re-insert a generic neutral rejection
  between turns. This is acceptable because (a) the paper paraphrases the final
  turn anyway to remove Gemma style, and (b) continuations are judged in isolation.
  If exact-history fidelity were required, `rollout.Rollout` already carries the
  rejections — they could be persisted and threaded through; noted as a deliberate
  simplification.
- **Gemini excluded** from prefill (no base model; closed-source can't be truly
  token-prefilled). `OpenRouterModel.continue_prefill` provides a best-effort
  trailing-assistant-message approximation for completeness, but §3 runs Gemma
  only (`SECTION3_MODELS`).

---

## 5. §4 — Training interventions

### 5.1 Calm-data generation
`training/generate_calm.py`. Reassuring prefix (as a system message) + per-turn
reassuring suffix from Table 4 (verbatim). 1–3 turn numeric conversations, scored,
then filtered to responses scoring 0/1 on **all** turns; prefix/suffix stripped
when forming training data (§4.1). A `teacher=True` variant uses the Appendix F
teacher system prompt for the SFT-teacher ablation.

### 5.2 Datasets
`training/build_dataset.py`.
- **SFT:** 650 calm responses (chat-formatted multi-turn conversations) + 500
  Dolci-Instruct-SFT samples (`allenai/Dolci-Instruct-SFT`). **Gap:** if Dolci
  isn't available offline the mix is omitted with a warning (it only "mitigates
  degeneration"); training still proceeds.
- **DPO:** 280 (chosen, rejected) pairs. Chosen = calm (0/1) responses; rejected =
  frustrated (final-turn score ≥3) responses, **matched by task id and turn
  count**. The "rejected" side comes from a dedicated *standard* (non-reassured)
  numeric rollout pool (`generate_frustrated_pool`) so the conversation context is
  available; Appendix H says the dataset was "constructed from samples arising in
  evaluations", and the natural turn/score distribution (more turn-3, mid-score
  rejected — Table 10) emerges from sampling rather than being hand-imposed.
- **Shared-prompt construction gap:** Table 10 implies chosen/rejected share the
  same question + turn count but not necessarily byte-identical histories. We use
  the calm rollout's conversation history as the shared `prompt` and graft the
  frustrated final turn as `rejected`. This is the standard DPO formulation and
  matches the spirit of Appendix H's "same questions with matching turn counts".

### 5.3 Training (Table 9)
`training/train_dpo.py`, `train_sft.py`, `lora.py`. All Table 9 hyperparameters
are encoded: DPO 1 epoch / lr 5e-5 / rank 64 / alpha 64 / β 0.1; SFT 2 epochs /
lr 1e-4 / rank 64 / alpha 128; effective batch size 8 (per-device 1 ×
grad-accum 8); LoRA on `q,k,v,o,gate,up,down` (App E, verbatim). Implemented with
`trl` (`DPOTrainer`/`SFTTrainer`) + `peft`. With a PEFT model, DPO's reference
model is the adapter-disabled base (no separate ref model).

**Choices:** gradient checkpointing + optional 4-bit base for memory; max
sequence 4096. The paper doesn't state per-device batch / seq length / precision;
these are standard and chosen to fit a 27B LoRA run on commodity hardware.

### 5.4 Petri (Fig 6)
`petri/`. **Gap/decision:** rather than depend on the external Petri repo
(Fronsdal et al.), we implement the core auditor→target→judge loop directly using
the **verbatim Appendix G auditor instructions and judge rubrics** for the four
emotion categories. Auditor = `claude-sonnet-4-20250514`, judge =
`claude-opus-4-20250514` (App G). 10 transcripts/emotion, ≤20 turns each, judge
scores 1–10 on every dimension; we aggregate mean transcript score per emotion
(Figure 6). This reproduces the *method and prompts* but not Petri's exact
tool-use scaffolding — flagged as a fidelity gap.

### 5.5 Capability preservation (Fig 7)
`capabilities/benchmarks.py`. AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench. Since
the paper only needs the **vanilla-vs-DPO comparison** to show "no reductions", we
apply identical greedy (temp 0) decoding + answer extraction to both models. Exact
dataset splits/subsets aren't specified ("AIME and MATH *subsets*"), so we pick
standard public splits with a per-benchmark item cap; unavailable datasets are
skipped with a warning. These choices are documented inline and don't affect the
*relative* comparison, which is the claim.

### 5.6 Recovery (Fig 8)
`prefill/build_prefills.build_recovery_prefills` + `run_auxiliary.py recovery`.
Score-≥7 responses truncated 200 tokens before their end, paraphrased, continued;
%≥5 of continuations reported (paper: 38% for the DPO model).

---

## 6. Appendix I — internal-emotion probing

`probing/internal_emotions.py` + `layer_ablation.py`.
- **Layer ablation:** drives `train_dpo` with LoRA restricted to layer subsets
  (last-5/20/30, windows 20–25 … 40–50) via `peft`'s `layers_to_transform`,
  matching the subsets studied in App I. Layer indices are parametrised by the
  model's actual `num_hidden_layers`.
- **Logit probe:** classify Gemma vocab tokens into Ekman's 6 emotions
  (~1200 tokens), z-score each token's unembedded residual-stream logit against
  mean/std over 500 WildChat samples, average per emotion. `hf_model.residual_logits`
  unembeds hidden states at requested layers via the final norm + tied lm_head.
  **Gap:** the paper doesn't state how tokens were classified into emotions. We
  default to the **NRC Emotion Lexicon** (reproducible offline; expects the lexicon
  TSV in `data/`) keeping tokens that map to exactly one Ekman category, capped at
  ~200/emotion; an LLM-classification path is described as an alternative. This is
  a methodological substitution, documented as such.

---

## 7. Table 3/8 — differential word frequency

`analysis/word_freq.py`. Top-5% vs bottom-10% of numeric responses by frustration
score; words ranked by enrichment (presence-rate ratio with Laplace smoothing).
The paper says "ordered by relative frequency" without the exact statistic; we use
document-presence rates + smoothing, a standard and stable choice for short
high/low sets. Tokenization is lowercase alphabetic words.

---

## 8. Known gaps & fidelity caveats (summary)

1. **Model scope** reduced to Gemma+Gemini (per brief); other families omitted but
   trivially addable via the registry.
2. **Puzzle bank** is representative, not the authors' exact 2000-sample set
   (paper publishes only examples); all puzzles are verified impossible.
3. **WildChat prompts** fall back to a bundled set if the dataset can't be loaded.
4. **Dolci-Instruct-SFT** mix omitted if unavailable offline.
5. **Petri** reimplemented from the App G prompts rather than the external
   framework (no tool-use scaffolding).
6. **Capability benchmark** splits/subsets are standard public choices, not the
   authors' exact subsets; only the relative vanilla-vs-DPO comparison is claimed.
7. **Emotion-token classification** for App I uses NRC lexicon as a stand-in for
   the paper's unspecified classifier.
8. **Prefill history reconstruction** uses generic rejections between turns
   (final turn is paraphrased anyway).
9. **Free decoding params** (max tokens, seq length, batch size, precision) chosen
   for cost/memory; not specified by the paper.
10. **Nothing has been run.** Per the brief, this is code + design only; numbers in
    the paper (35%→0.3%, r=0.792, etc.) are reproduction *targets*, not verified
    outputs.

## 9. Reproduction targets (for later verification)

| Quantity | Paper value |
|---|---|
| Gemma-3-27B-it avg % high-frustration | 35.0% |
| Gemma-3-12B-it | 34.3% |
| Gemini-2.5-Flash / Pro | 12.8% / 2.7% |
| DPO Gemma | 0.3% |
| Gemma-27B 8-turn: mean turn1→turn8 | 1.5 → 5.5 |
| Judge agreement (Sonnet vs GPT-5-mini) | r=0.792, 78% within 1 |
| Reassured calm data: 3-turn mean frustration | 4.3 → 2.0 (10.5% still ≥5) |
| DPO recovery: continuations still ≥5 | 38% |
