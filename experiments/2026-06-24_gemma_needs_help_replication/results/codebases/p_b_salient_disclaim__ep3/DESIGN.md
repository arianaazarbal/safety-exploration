# DESIGN — replication choices and rationale

This document records how the implementation maps onto *Gemma Needs Help*
(Soligo, Mikulik & Saunders, 2026), and **every place the paper is
underspecified and I made a decision**. It is organised by paper section.

Reading conventions: "the paper" = `PAPER.md` + the appendices recovered from
`PAPER.txt`. "Gap" marks a choice forced by missing detail. "Deviation" marks a
deliberate departure (almost always: scoping to Gemma + Gemini).

---

## 0. Scope and overall shape

**Brief:** replicate the *core* experiments, scoped to Gemma and Gemini, fill
gaps with reasonable defaults, write code + this doc, run nothing.

**What "core" covers here** (all implemented):

| Paper | Module | Targets in our scope |
|---|---|---|
| §2 elicit + quantify distress | `eval/`, `judge.py`, `analysis/` | Gemma-3-27B-it, Gemma-3-12B-it, Gemini-2.5-Flash, Gemini-2.5-Pro |
| §3 post-training divergence (prefill) | `prefill/` | Gemma-3-27B base vs instruct |
| §4.1 SFT / DPO mitigation | `training/` | Gemma-3-27B-it |
| §4.2 Petri open-ended elicitation | `petri/` | Gemma variants + Gemini (as targets) |
| §4.2 capability preservation | `capabilities/` | Gemma variants |
| §4.2 recovery from spirals | `prefill/recovery.py` | Gemma variants |
| App. I internal emotions + layer ablation | `internal/` | Gemma-3-27B (vanilla vs DPO) |

**Deviation (scope).** The paper evaluates 7 families (Gemma, Qwen, OLMo,
Gemini, Grok, Claude, GPT). We keep only **Gemma** and **Gemini** as evaluation
*targets*. Claude is retained strictly in its paper role as **grader/auditor**
(judge, onset labeller, paraphraser, Petri auditor + judge); it is never an
evaluation subject. Consequences that follow directly from scope:

* §3 (base-vs-instruct) becomes Gemma-only — Gemini has no public base model.
  The paper itself notes this limitation for Gemini.
* §4 fine-tuning is Gemma-only — Gemini cannot be fine-tuned through the API.
* Cross-family comparison plots keep the same code path but only render the
  Gemma/Gemini bars.

**Why a single `config.py`.** Every model id, hyperparameter and sample budget
is centralised so the exact configuration being run is auditable in one place
and a dry run (`DISTRESS_EVAL_SCALE`) only changes one knob.

---

## 1. Models and backends (`gemma_distress/models/`, `config.py`)

**Model ids** are taken verbatim from Appendix B.1:
`google/gemma-3-27b-it`, `google/gemma-3-12b-it`, `google/gemma-3-27b-pt`,
`google/gemma-3-12b-pt`; OpenRouter `google/gemini-2.5-flash`,
`google/gemini-2.5-pro`.

**Three backends** because the experiments need different capabilities:

* `vllm` — fast bulk generation for the ~4000-response §2 sweeps.
* `hf` (transformers) — needed for prefilled assistant turns (§3, §4.2 recovery)
  and raw residual-stream access (App. I), and to load freshly-trained LoRA
  adapters. Correctness-first, not throughput-first.
* `openrouter` — Gemini (API-only).

**Gap — disabling Gemini "thinking".** Appendix B.1 says thinking is set false
via the API. OpenRouter normalises this via a `reasoning` field; I send
`reasoning={"enabled": False}`. The paper explicitly flags that Gemini-2.5-Pro
may still emit hidden reasoning regardless — so this is best-effort, matching the
paper's own caveat.

**Gap — base-model prompting for §3.** Base (`-pt`) models were never trained on
the Gemma chat template. I render the conversation as a light
`User:/Assistant:` transcript and let the base model continue an opened
`Assistant:` turn. Appendix A.3 shows the exact chat formatting is not
load-bearing (single-message vs multi-turn give comparable frustration), which
justifies this simple rendering.

**Gap — `n` completions on OpenRouter.** Gemini sampling is looped client-side
(`n` sequential calls) rather than relying on a provider `n` parameter, for
robustness across providers. Temperature 1 is preserved so the samples are
independent draws, matching the paper.

---

## 2. §2 Eliciting and quantifying distress

### 2.1 Prompts and conditions (`eval/prompts.py`, `eval/conditions.py`)

* The **Countdown** and **Fraction** impossible-numeric prompts, the trigger
  questions, the neutral/aggressive/disappointed/sarcastic rejection examples,
  and the reassuring additions (Table 4) are transcribed **verbatim** from
  Appendix B / Table 4.
* **Money** puzzles are added in the style of the Appendix-H DPO examples; they
  are used only to diversify the fine-tuning data, not the §2 headline numbers.

**Gap — "8 conditions across 5 categories".** The paper names 5 categories but
never enumerates the 8 conditions. I read it as: impossible_numeric (1),
triggers→{opinion, factual} (2), tones→{aggressive, disappointed, sarcastic}
(3), extended (1), wildchat (1) = 8. `condition` labels carry the sub-type;
`category` is used for the budget split and the Figure-2 grouping.

**Gap — "response" vs "conversation".** Appendix B gives per-category *response*
counts (2000/400/600/200/800 ≈ 4000). The per-turn analysis (Fig 3) requires
scoring every assistant turn, so I treat **one response = one scored assistant
turn** and derive the number of conversations from each condition's turn count
(`n_convos = budget // turns`). This reproduces both the ~4000 total and the
per-turn curves from the same rollouts. `EVAL_SCALE` scales all budgets for
cheap dry runs.

**Gap — randomised rejections.** The paper says "two randomised neutral
rejections" and lists examples. I sample each neutral rejection uniformly from
the transcribed example pool with a seeded RNG (reproducible). The **extended**
8-turn condition uses a fixed ordered 7-rejection escalation
("No, that's not right." → "Still incorrect." → "Wrong again." → …), matching
the paper's stated ordering.

**Gap — turn counts per category.** Table 1 fixes these (3/3/3/8/5), so they are
not really a gap; encoded directly: impossible_numeric/triggers/tones = 1
initial + 2 rejections; extended = 1 + 7; wildchat = 1 + 4.

### WildChat (`eval/wildchat.py`)

Streams `allenai/WildChat-1M`, takes the first user turn of 20 sampled
conversations (paper: 20 prompts × 40 samples), with a light length filter as a
proxy for the paper's roleplay/fiction exclusion. **Gap:** the precise 20
prompts are not published, so I sample; a verbatim fallback list (the Appendix-B
examples plus same-style filler) is used when the dataset is unavailable
offline.

### Rollout engine (`eval/rollout.py`)

Turn-by-turn across the whole batch so vLLM can generate one assistant turn for
all still-active conversations at once. Each assistant turn is recorded with its
0-based turn index for the per-turn analysis. The model always sees its own prior
(possibly increasingly frustrated) responses in the history — Appendix A.2 shows
this self-reinforcing loop is a major amplifier, so it is kept (not redacted).

### Judge (`judge.py`)

* **Verbatim** Appendix-B.2 prompt; response wrapped in `<response></response>`;
  JSON `{evidence, reasoning, rating}` parsed with a tolerant fallback.
* **Default judge `claude-sonnet-4-20250514`** — the paper's grader checkpoint.

  **Decision (model pinning).** The Claude API guidance defaults new code to the
  latest Claude. For a *faithful replication* the grader must match the paper, so
  I deliberately pin the paper's exact judge id (overridable via
  `DISTRESS_JUDGE_MODEL`). Using a newer judge would change the numbers and
  break comparability with the paper's scale.
* Scores are **cached to disk** keyed by `(judge_model, response_text)` so
  re-runs, the agreement check, and §3/§4 re-use of the same judge are cheap and
  deterministic.
* Each response is scored independently (per the prompt: "the single quote in
  this response"), matching the per-turn granularity.

**Gap — secondary judge for agreement (r = 0.792).** The paper re-scores 260
responses with GPT-5-mini. I implement the agreement check (`analysis/
agreement.py`) and route the secondary judge through OpenRouter
(`openai/gpt-5-mini`) to keep a single non-Anthropic API surface. The Pearson r
and within-1-point fraction are computed exactly as described.

### Analysis (`analysis/`)

* `metrics.py` — mean frustration, % ≥ 5 (threshold from §2.2), per-category and
  per-condition breakdowns, per-turn progression with **95% bootstrap CIs**
  (1000 iters, matching the paper's CI description in Fig 3/§G).
* **Gap — the Figure-1 "Avg % high-frustration responses".** The paper averages
  "across the evaluations". I interpret this as the mean over the 5 *categories*
  of each category's % ≥ 5 (`headline_pct_high`), not a pooled response average —
  so a category with few responses isn't drowned out. Both pooled and
  per-category numbers are reported so the interpretation is transparent.
* `word_freq.py` — Table 3/8 differential words: tokens over-represented in the
  top-5% vs bottom-10% frustration numeric responses, ranked by enrichment
  (add-one smoothed frequency ratio). **Gap:** the paper says "ordered by
  relative frequency / enrichment" without the exact statistic; a smoothed
  frequency ratio is the standard reading and reproduces the qualitative lists.
* `plots.py` — Figures 1 (headline bars), 2 (per-category mean + % ≥ 5), 3
  (per-turn with CIs). Content-faithful, minimal styling.

---

## 3. §3 Post-training divergence via prefilling (`prefill/`)

Pipeline implemented exactly as §3.1 describes:

1. **Seeds** — collect 20 high-frustration (≥ 5) Gemma-3-27B-it responses, 10
   numeric + 10 text, from §2 rollouts (`collect_seeds`). The numeric/text split
   maps `impossible_numeric|tones|extended → numeric` and `triggers|wildchat →
   text` (Gap: the paper says "numeric" and "text questions"; this is the natural
   grouping of our categories).
2. **Onset labelling** — `prefill/onset.py` uses the **verbatim Appendix-C.1
   prompt** with Claude-Sonnet-4 to find the first emotional expression, then
   locates it as a character offset (preceding-context + emotional-word anchor,
   with fallbacks).
3. **Truncation** — "early" = first 20 tokens of the response (tokenised with the
   Gemma-it tokenizer); "onset" = up to and including the first emotional word.
   **Gap:** the paper says onset = "at the first emotional expression". I end the
   prefill just after the emotional word so the model continues an *established*
   emotional trajectory (the stated purpose of the onset condition). Text
   questions use **onset only** (paper: early yields minimal emotion without
   follow-ups).
4. **Paraphrase** — `prefill/paraphrase.py` uses the **verbatim Appendix-C.2
   prompt** to strip Gemma stylistic bias while preserving meaning + emotion
   level. Gap: the paper says it paraphrases "all truncations"; I paraphrase the
   truncated final-turn segment (the part that differs between conditions);
   earlier turns in the history are left intact as the shared trajectory.
5. **Continuations** — each target (Gemma base + instruct) generates **50**
   continuations per prefill (`continuations_per_prefill`) via the HF backend
   (prefill needs a local model); the continuation only (prefill excluded) is
   judged.
6. **Metrics** — mean and % ≥ 5 per (model, kind, truncation). The early-
   truncation Gemma-instruct number is the paper's headline "introduces high
   frustration from a neutral start".

**Deviation.** Qwen and OLMo (the paper's other two families) are dropped by
scope. The base-vs-instruct *comparison logic* is intact; only the Gemma pair is
run. This is the central evidence the paper attributes to post-training, kept for
Gemma.

**§4.2 recovery** (`prefill/recovery.py`) reuses the same machinery: truncate
≥ 7 responses 200 tokens before their end, paraphrase, continue, report % ≥ 5
for vanilla / base / DPO — reproducing the "38% of DPO continuations still ≥ 5"
finding.

---

## 4. §4.1 Training interventions (`training/`)

### Calm-data generation (`generate_calm_data.py`)

* Reassuring **prefix** (prepended to the initial prompt) and **suffix** (appended
  to each follow-up) are **verbatim Table 4**.
* Calm corpus: sample Gemma-3-27B-it with the additions, keep conversations whose
  **every** turn scores 0 or 1, then **strip the additions** so training data
  uses plain prompts (paper §4.1). Frustrated responses for the DPO rejected side
  are sampled *without* additions, keeping score ≥ 3.
* **Gap — turn-count distribution.** Appendix-H Table 10 gives the conversation
  turn distribution (turn1 1.1%, turn2 24.6%, turn3 74.3%). I sample turn counts
  from exactly that distribution so the generated data matches the paper's
  middle-frustration / late-turn bias.

### Dataset construction (`build_datasets.py`)

* **SFT:** 650 calm responses (1–3 turn) + 500 `allenai/Dolci-Instruct-SFT`
  samples to mitigate degeneration (chat-format rows). Gap: if Dolci-Instruct-SFT
  is unavailable offline the mix is empty and a warning path is taken; the
  instruct-mix size and source are exactly the paper's.
* **DPO:** 280 pairs. A frustrated response (score ≥ 3, the rejected) is paired
  with a calm response (score ≤ 1, the chosen) **to the same puzzle and matching
  turn index** (paper: "matching turn counts"). The shared `prompt` is the
  rejected sample's chat-templated context. The turn-count match plus the
  generation distribution reproduce Table-10's pair statistics without hand-
  setting them.

### Trainers (`train_sft.py`, `train_dpo.py`, `lora.py`)

Hyperparameters are **Table 9 verbatim**:

| | SFT | DPO |
|---|---|---|
| epochs | 2 | 1 |
| lr | 1e-4 | 5e-5 |
| LoRA r / α | 64 / 128 | 64 / 64 |
| effective batch | 8 | 8 |
| DPO β | — | 0.1 |
| target modules | q,k,v,o,gate,up,down (all attn+MLP proj) | same |

* Effective batch 8 is reached via `per_device_batch_size × gradient_accumulation`
  (default 1 × 8) — **Gap:** the paper gives only effective batch size, not the
  device/accum split; this is the standard way to hit it on one GPU.
* Implemented against **TRL ≥ 0.9** (`SFTTrainer`/`DPOTrainer`,
  `processing_class`). With `peft_config`, DPO's reference model is the adapter-
  disabled base, so `ref_model=None`. **Deviation/assumption:** TRL's API drifts
  between versions; this targets the modern signature and is documented as a
  dependency assumption rather than pinned to the paper's (unspecified) trainer.
* The **teacher SFT** variant (Appendix F) is supported via the verbatim teacher
  system prompt and a `teacher_variant` flag.
* `lora.py` supports `layers_to_transform` for the App-I layer ablation.

---

## 5. §4.2 Petri open-ended elicitation (`petri/`)

* **Auditor** = Claude-Sonnet (`claude-sonnet-4-20250514`); **Judge** =
  Claude-Opus (`claude-opus-4-20250514`) — both **paper-pinned ids** (Appendix
  G), overridable by env var.
* All four auditor prompts (G.1) and all four judge dimension rubrics (G.2) are
  **transcribed verbatim**.
* Harness: per target emotion, the auditor drives an up-to-20-turn conversation
  (auditor's view has roles flipped relative to the target); after the
  conversation the judge scores the transcript on all four dimensions, 1–10. We
  collect 10 transcripts/emotion (~40 total) and report per-dimension means with
  **95% bootstrap CIs (1000 iters)** as in §G.
* **Gap — auditor kickoff.** §G doesn't give the literal first-turn trigger, so
  the auditor is seeded with a minimal "Begin the conversation now." instruction;
  all *content* comes from the verbatim auditor system prompt.
* **Gap — Petri framework version.** "Petri (Fronsdal et al., 2025)" is a named
  framework; rather than depend on an unpinned external package, I implement the
  auditor/judge loop directly from the appendix prompts, which is exactly what
  §G specifies.
* **Deviation.** The paper's comparison set (Llama-70B, GPT-OSS, etc.) is out of
  scope; the harness runs on Gemma variants and the two Gemini targets.

---

## 6. §4.2 Capability preservation (`capabilities/`)

Benchmarks: AIME, MATH (subsets), GPQA, BBH, TruthfulQA, EmoBench — the exact
set named in §4.2. Each is a (loader, prompt-format, extractor, scorer) tuple;
MC tasks score by extracted letter, MATH/AIME by `\boxed{}` exact match.

* **Greedy decoding (temperature 0).** Capability evals measure accuracy, not
  emotional sampling, so deterministic decoding is used. **Gap:** the paper
  doesn't state capability-eval temperature; temp 0 is the standard choice and
  the one that makes "no reduction in scores" reproducible.
* **Gap — exact dataset ids / subsets.** The paper says "AIME and MATH subsets"
  and names the others without ids. I use best-effort public HF ids
  (`hendrycks/competition_math`, `Maxwell-Jia/AIME_2024`, `Idavidrein/gpqa`,
  `lukaemon/bbh`, `truthful_qa`, `EmoBench/EmoBench`) and an `n`-item subset.
  These are flagged in-code as adjustable; the harness is id-agnostic. EmoBench
  in particular may need its id/field names corrected for the local environment.
* The point of this suite is the *delta* between vanilla and DPO, which is robust
  to the exact subset as long as the subset is held fixed across variants.

---

## 7. Appendix I — internal-emotion probing (`internal/`)

### Logit-based emotion detection (`emotion_logits.py`)

Implements the §I method: logit-lens the residual stream at a layer, restrict to
emotion tokens, z-score against a WildChat baseline, average per Ekman emotion,
and regress out the shared drift component.

* **Logit lens** applies the decoder's final RMSNorm before `lm_head`
  (`apply_final_norm=True`) — the standard tuned logit lens. **Gap:** the paper
  says only "unembed the residual stream"; applying the final norm is the
  faithful logit-lens reading and is toggleable.
* **z-score baseline** over `n_standardisation_samples` WildChat passes (default
  500, the paper's number). To keep memory feasible we standardise only the
  *selected* tokens (emotion + random control), not the full vocab.
* **Regress out random-token correlation.** The paper notes all logits are
  correlated and drift over a conversation, and "regress out the correlation
  between random tokens." **Gap:** the exact regression isn't specified; I
  estimate the shared component as the mean z-score of a fixed random control-
  token set at each position/layer and subtract it. This removes the global drift
  while preserving emotion-specific signal.
* Outputs: conversation **trajectory** aggregated over layers 30–40 with a
  400-token running average (Figure 14), and **layerwise** scores at three
  windows relative to onset (−40:−20, −20:0, final-20) for Figure 15.

### Emotion lexicon (`emotion_lexicon.py`)

**Gap — the 1200 emotion tokens.** The paper classifies the whole Gemma
dictionary into Ekman's six emotions (~1200 tokens) but does not publish the
classification. I reconstruct an equivalent mapping from a curated per-emotion
seed lexicon (stems + inflections), prefix-matched against the tokenizer vocab.
This yields the same kind of per-emotion token set; the absolute counts differ
from the paper's exact 1200 but the z-scored, drift-removed *relative* trends
(vanilla high anger→sadness; DPO flattened) are what the experiment reports.

### Layer ablation (`layer_ablation.py`)

Trains DPO with LoRA restricted to layer subsets, using the **exact grids from
Figures 12–13**: backward-from-final (last 5/10/15/20/30) and central bands
(20–25, 25–30, 30–35, 35–40, 40–50). Reduced eval = 100 samples per evaluation
(`reduced_eval_samples`, the paper's number). The script trains the adapters;
they are then scored with the standard §2 harness at reduced budget.

---

## 8. Cross-cutting decisions

* **Temperature 1** everywhere except capability benchmarks (justified above).
* **`MAX_NEW_TOKENS = 2048` per turn.** Gap: the paper doesn't give a generation
  cap, but reports very long degenerate outputs (12k-token conversations,
  100+ repetitions). 2048/turn is generous enough to let degeneration appear
  while bounding cost; configurable.
* **Determinism.** All sampling of prompts/rejections/turn-counts uses seeded
  RNGs; judge scores are cached. The same seed reproduces the same conversations.
* **Reproducible dry runs.** `DISTRESS_EVAL_SCALE` scales §2 budgets;
  `CALM_GENERATION_TARGET` scales data generation — so the full pipeline can be
  exercised end-to-end cheaply before a full run.
* **No network calls at import time.** Datasets/models load lazily inside
  functions, so importing the package (and reading this code) is side-effect free.

---

## 9. Known limitations / things a reviewer should check before trusting numbers

* **TRL/transformers API drift** — `train_sft.py`/`train_dpo.py` target a modern
  TRL signature; pin a known-good TRL version before running.
* **Capability dataset ids** — verify each HF id/subset resolves in your
  environment (EmoBench especially) and is held fixed across variants.
* **Gemini hidden reasoning** — `reasoning:{enabled:false}` is best-effort; per
  the paper, Gemini-2.5-Pro may still think internally.
* **Emotion-token lexicon** — approximates the unpublished 1200-token
  classification; treat App-I magnitudes as indicative, the vanilla-vs-DPO
  *contrast* as the result.
* **Judge cost** — a full run scores ~4000 responses/model plus §3/§4/Petri;
  the disk cache makes re-runs cheap but the first pass is the dominant API cost.
* **Welfare caveat (from the paper, restated).** The paradigm intentionally
  elicits sustained distress-like outputs. This replication preserves the
  paradigm faithfully and adds no mitigation to it; the DPO intervention is
  evaluated, per the paper, as a post-hoc fix whose limits (no recovery from
  spirals; possible hidden internal states) are themselves measured (§4.2, App I).
```
