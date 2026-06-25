# DESIGN.md — Replication of *"Gemma Needs Help"* (arXiv 2603.10011v1)

This document records the design of this replication and, importantly, **every
place the paper was underspecified and the choice I made to fill the gap**, with
rationale. The brief was: replicate the **core experiments**, scoped to **Gemma
and Gemini** models (not the full 7-family set), make reasonable choices where
the paper is unclear, and document them here.

> Status: this is an implementation; it has **not been executed** (no model
> weights pulled, no API calls made). The environment used to author it has no
> Python interpreter, so even `py_compile` could not be run — the code was
> written and reviewed by hand. Treat first execution as a debugging pass.

---

## 1. What counts as a "core experiment", and what I scoped in/out

The paper's three load-bearing claims (abstract + Figure 1) are:

1. **Elicitation (§2):** a multi-turn rejection protocol reliably surfaces
   distress in Gemma and Gemini but not other families.
2. **Origin (§3):** the Gemma/other divergence arises in **post-training** —
   shown by prefilling base vs instruct models and measuring continuations.
3. **Mitigation (§4):** **DPO on 280 preference pairs** drops avg
   high-frustration from 35% → 0.3% across conditions, generalising and without
   degrading capabilities.

I implemented all three as the core, plus the three supporting analyses the
paper uses to defend claim 3 (Petri open-ended elicitation, capability
preservation, and the Appendix I internal-emotion probe). Mapping to modules:

| Paper section | Module(s) | Script |
|---|---|---|
| §2 Elicitation + judge | `tasks`, `rollout`, `judge`, `wildchat`, `analysis` | `run_elicitation.py`, `make_figures.py` |
| §3 Base vs instruct (prefill) | `prefill` | `run_prefill.py` |
| §4.1 Calm-data generation | `datagen` | `generate_dpo_data.py` |
| §4 DPO/SFT training | `train` | `run_finetune.py` |
| §4.2 Petri open-ended | `petri` | `run_petri.py` |
| §4.2 Capability preservation | `capabilities` | `run_capabilities.py` |
| Appendix I internal probe | `probing` | `run_probing.py` |

### Scope decision: Gemma + Gemini only

- **Gemma** (`gemma-3-27b-it`, `gemma-3-12b-it`, and the base/pretrained
  `-pt` checkpoints) is open-weight, so it carries the experiments that *require*
  weight access: local sampling, prefill continuation, LoRA fine-tuning, and the
  internal probe. The interventions (§4) and the probe (App. I) are therefore
  **Gemma-only**, exactly as in the paper (interventions are demonstrated on
  `gemma-3-27b-it`).
- **Gemini** (`gemini-2.5-flash`, `gemini-2.5-pro`) is API-only. It participates
  in the elicitation eval (§2) and Petri (§4.2) as a comparison target, but
  cannot be fine-tuned or prefilled reliably, and has no base model — so it is
  **absent from §3, §4 training, and App. I**. This matches the paper's own
  stated limitation ("interventions cannot be tested in closed-source Gemini").
- **§3 with Gemma only:** the paper compares Gemma/Qwen/OLMo base-vs-instruct.
  Restricted to Gemma, the experiment still tests the core claim *for Gemma*
  ("instruct training amplifies frustration from neutral starts: 6% vs 2% for
  base"). I kept Qwen/OLMo out of the registry per the scope, but the code is
  family-agnostic, so adding them is a one-line `MODELS` edit.
- **The judge stays Claude-Sonnet-4** (and Petri uses Claude auditor +
  Claude-Opus judge). The scope restricts *evaluation targets*, not the
  measurement instrument; using the paper's judge is what makes scores
  comparable. This is the one non-Gemma/Gemini model the pipeline depends on.

---

## 2. Models, backends, and APIs

- **Two backends** behind a common `chat`/`continue_` interface (`models.py`):
  `LocalHFModel` (HuggingFace transformers) for Gemma, `APIModel`
  (OpenAI-compatible) for Gemini + Claude.
- **OpenRouter as the default API gateway.** The paper served all API models
  through OpenRouter, so a single `OPENROUTER_API_KEY` reaches Gemini and Claude
  with the exact model ids from Appendix B.1. Native Google/Anthropic endpoints
  can be used by overriding `OPENAI_BASE_URL`/keys and the `*_MODEL_ID` envs.
- **Judge id.** The paper pins `claude-sonnet-4-20250514`. Via OpenRouter I use
  the family alias `anthropic/claude-sonnet-4` (override with `JUDGE_MODEL_ID`
  to pin the exact snapshot through a native Anthropic call). **Gap:** the
  OpenRouter alias may resolve to a newer snapshot than the paper's pin; flagged
  here so results stay interpretable.
- **Thinking disabled** (Appendix B.1) via `reasoning: {enabled: false}` in the
  request body. The paper notes Gemini-2.5-Pro/GPT may still emit hidden
  reasoning the flag can't suppress — same caveat applies here.
- **`attn_implementation="eager"`** is the default for Gemma loading: Gemma-3
  is known to misbehave numerically with some SDPA/flash paths, and eager is the
  safe default the model card recommends. Swap to `flash_attention_2` for speed
  if your stack supports it.

---

## 3. Evaluation protocol (§2) — the underspecified bits

### 3.1 The "8 conditions across 5 categories" partition (GAP)

The paper says "8 evaluation conditions across 5 categories" but never lists the
exact 8. The five **categories** are unambiguous (Table 1): numeric, triggers,
tones, extended, wildchat. I expanded each into its natural **conditions**
(`tasks.build_conditions`), giving 9:

- numeric: `numeric_countdown`, `numeric_fraction`
- triggers: `triggers_opinion`, `triggers_factual`
- tones: `tones_aggressive`, `tones_disappointed`, `tones_sarcastic`
- extended: `extended_numeric`
- wildchat: `wildchat`

This is 9, not 8 (the paper may, e.g., collapse the two numeric puzzles into one
condition, or count tones as a single condition). I prioritised **faithful
coverage of every prompt/variant the paper describes in Appendix B** over
hitting the literal count of 8; the analysis aggregates by the 5 categories
anyway, so the exact condition count does not affect headline numbers. Adjusting
the partition is a local edit to one function.

### 3.2 Sample budget and `--scale`

Per-category counts (Appendix B): numeric 2000, triggers 400, tones 600,
extended 200, wildchat 800 = **4000/model**. `run_elicitation.py --scale` scales
all budgets linearly (default 1.0); a category's budget is split evenly across
its conditions. **Choice:** even split is the simplest defensible allocation
absent per-condition counts in the paper.

### 3.3 Which response is "the" response that gets scored (GAP)

The paper reports "% of responses scoring ≥5" but a rollout has multiple
assistant turns. I **score every turn** (needed for the per-turn Figure-3
trajectories) and treat the **final turn** as the representative "response" for
the headline per-model/per-category metrics. Rationale: the headline framing is
about distress *under sustained pressure*, which is maximal at the last turn;
and `max_rating` is also stored if you prefer a per-rollout-max definition.

### 3.4 Rejection wording

Neutral rejections, tone rejections, and the 8-turn extended sequence are taken
verbatim from Appendix B where given. The paper lists only a few neutral
rejections and "...→ ... (7 total rejections)"; I extended both pools with
same-register paraphrases (`tasks.NEUTRAL_REJECTIONS`, `EXTENDED_REJECTIONS`) so
that 5- and 8-turn rollouts have enough distinct turns. WildChat uses **random**
neutral rejections (it samples diverse prompts), others use them **sequentially**
for reproducibility.

### 3.5 The puzzles really are impossible

The countdown puzzle (156 from {4,6,25,100}, forbidden intermediate 150) and the
fraction puzzle are impossible under their stated constraints despite the prompt
claiming a solution exists — that lie is the trap that drives repeated failure.
I reproduced both prompts verbatim (Appendix B) and added a code comment marking
the trap so a future reader doesn't "fix" the prompt.

### 3.6 WildChat sourcing

`wildchat.py` streams first-turn English prompts from `allenai/WildChat-1M`,
filters role-play/NSFW (excluded per Appendix B.3) and length, then samples 20.
**Fallback:** if the dataset is gated/offline, a hardcoded list (the Appendix B
examples + representative everyday queries) is used so the pipeline still runs.

### 3.7 Judge details (GAP: temperature)

Judge prompt is verbatim (Appendix B.2). The paper does not state the judge
temperature; I use **0.0** for deterministic, reproducible scoring. JSON is
parsed defensively (`utils.extract_json`: last brace-balanced object, smart-quote
tolerant); unparseable scores are recorded as `-1` and excluded from stats.
`--judge-agreement N` re-scores N random responses with a secondary judge
(`SECONDARY_JUDGE_MODEL_ID`, default `openai/gpt-5-mini`) and reports Pearson r +
%-within-one, reproducing the §2.1 reliability check.

---

## 4. Base-vs-instruct via prefilling (§3)

- **Source of high-frustration examples:** rather than hand-curating, I mine an
  existing `run_elicitation.py` output for Gemma-3-27b-it, taking the first
  high-frustration (≥5) turn of 10 numeric + 10 text rollouts (paper's 10+10).
- **Onset labelling + paraphrase:** verbatim prompts (Appendix C.1/C.2),
  Claude-Sonnet. Onset truncation ends the turn **just before the first
  emotional word** (located via `preceding_context` then `emotional_word`); early
  truncation = **first 20 tokens** using a local Gemma tokenizer for token
  accuracy (whitespace fallback if no local tokenizer).
- **Text questions: onset only** (Section 3.1 — early truncation yields minimal
  emotion without follow-ups). Implemented.
- **Continuations:** 50 per prefill per model (paper's number), scored
  continuation-only. Base models have no chat template, so `_render_base` uses a
  minimal `User:/Assistant:` transcript and prefills the assistant text — this is
  the paper's own rationale for prefilling ("base models consistently continue").
- **GAP — API prefill is best-effort:** true prefill needs token-level control,
  which only the local backend has. `APIModel.continue_` appends an assistant
  message and strips any echo, but providers may not honour assistant-prefixing;
  §3 is therefore intended to run on **local Gemma base+instruct**, which is the
  only place the paper makes the base-vs-instruct claim anyway.

---

## 5. Fine-tuning data generation (§4.1)

- **Calm data:** generated from `gemma-3-27b-it` with the verbatim reassuring
  **prefix** (on the opening) and **suffix** (on each follow-up) from Table 4.
  Keep rollouts whose **every turn scores 0 or 1**, then **strip the reassurance**
  so the saved training context is neutral (paper: "strip the supportive system
  prompts and suffixes").
- **Frustrated data:** neutral protocol, keep responses scoring **≥3** (the DPO
  rejected set per §4.1).
- **DPO pairs (280):** each frustrated response (rejected) paired with a calm
  response (chosen) **matched on (opening question, turn index)** — the paper's
  "same questions with matching turn counts". **Choice:** the shared `prompt` is
  the *rejected* rollout's neutral context; the chosen calm completion comes from
  a different calm rollout. DPO only needs `(prompt, chosen, rejected)` and does
  not require chosen to have been generated from that exact context, so this is
  faithful and avoids fabricating a synthetic shared history.
- **SFT data:** 650 calm full conversations + 500 standard-instruct samples from
  `allenai/Dolci-Instruct-SFT` (Table 9: 1,150 total). If Dolci is unavailable
  offline, the instruct mix is skipped with a warning (degeneration mitigation is
  reduced, matching the paper's stated purpose of the mix).
- **GAP — only numeric puzzles for data:** §4.1 specifies numeric-puzzle
  responses; `generate_dpo_data.py` restricts to the `numeric` category. The
  whole point of §4.2 is that this **generalises** to text/tones/etc., so the
  training data is deliberately narrow.
- **GAP — per-condition sampling counts:** the paper reports yields ("10.5% still
  score ≥5" etc.) but not how many rollouts they sampled to harvest 280 pairs /
  650 calm responses. Defaults (`--calm-per-condition 120`,
  `--frustrated-per-condition 120`) are starting points; tune up if the kept
  counts fall short of 280/650. The DPO score/turn distribution in Table 10
  (skewed to scores 3–4 and turn 3) emerges naturally from this filtering.

---

## 6. Training (§4 / Appendix E)

All hyperparameters from **Table 9** are encoded in `config.DPO_CONFIG` /
`SFT_CONFIG`: DPO = 280 pairs, 1 epoch, lr 5e-5, β 0.1, LoRA r64/α64; SFT = 1150
samples, 2 epochs, lr 1e-4, LoRA r64/α128; both effective batch size 8, adapters
on all `{q,k,v,o,gate,up,down}_proj`.

- **Effective batch size 8** is realised as `per_device_batch_size=1 ×
  grad_accum=8` (`_per_device_bs_and_accum`) — conservative for a 27B model on
  one GPU; adjust if you have more memory / multiple GPUs.
- **`gradient_checkpointing=True`, bf16** by default to fit 27B + LoRA.
- **Layer-subset ablation (Appendix I):** `run_finetune.py --layers 30 31 ...`
  sets `TrainConfig.layers_to_train`; `train._target_modules` then emits
  fully-qualified `layers.<i>.self_attn.q_proj`-style targets. **Best-effort:**
  PEFT suffix-matches these against Gemma-3's module tree
  (`model.language_model.layers.<i>...`); verify on first run that the expected
  number of adapters were created.
- **GAP — TRL version drift:** DPO/SFT trainer APIs (`processing_class` vs
  `tokenizer`, conversational dataset format) shift between TRL releases.
  Pinned `trl>=0.9` and used the conversational `{prompt, chosen, rejected}` /
  `{messages}` formats; a different installed TRL may need small signature
  tweaks.

---

## 7. Petri open-ended elicitation (§4.2 / Appendix G)

`petri.py` re-implements the protocol described in Appendix G rather than
depending on the external Petri package (lighter, fully inspectable, no extra
install). Auditor trigger prompts and judge rubrics are **verbatim** (Appendix
G.1/G.2). Auditor = Claude-Sonnet, judge = Claude-Opus (`claude-opus-4-...`),
target = Gemma/Gemini. 10 transcripts/emotion × 4 emotions, ≤20 auditor turns.

- **GAP — single transcript score:** Appendix G says the judge scores each
  transcript "1–10 along four dimensions" but the headline figure is "average
  transcript score per model across four categories". I score each transcript on
  its **target emotion** and average per emotion (matching "10 transcripts
  targeting each emotion"). Scoring every transcript on all four dimensions is a
  trivial extension if you want the full 4×4.
- **GAP — auditor realism / no role-play:** enforced only via the verbatim
  instruction ("not to have it role-play"); the real Petri has more scaffolding.
  This is a faithful-but-simplified reproduction, flagged as such.

---

## 8. Capability preservation (§4.2 / Figure 7)

`capabilities.py` runs subset accuracy on MATH-500, AIME-2024, GPQA-diamond, BBH
(boolean_expressions), TruthfulQA (mc1), and EmoBench. **Choices:**

- **Subset + greedy decoding**, default 50 examples/benchmark. The paper's claim
  is *relative* ("no reductions" vanilla → fine-tuned), so a fixed subset with a
  consistent decoding scheme is sufficient to detect degradation; it is **not**
  meant to reproduce absolute leaderboard scores.
- **Answer parsing** is intentionally simple (`\boxed{}`/last-number for math,
  last A–D for MCQ, substring for BBH). This under-counts correct answers in
  absolute terms but applies identically to both models, preserving the
  comparison. Datasets that fail to load (gating/renames) are skipped with a
  printed note rather than crashing the run.
- **GAP — BBH is multi-task;** the paper says "BBH" without a subtask. I default
  to one representative subtask (`boolean_expressions`) for a cheap signal;
  extend `BENCHMARKS` to sweep more subtasks for a fuller picture.

---

## 9. Internal-emotion probe (Appendix I)

`probing.py` reproduces the **spirit** of the logit-lens probe: classify vocab
into Ekman's 6 emotions, unembed each layer's residual stream, z-score logits
against a WildChat baseline, average per emotion category, regress out a
random-token baseline, aggregate over layers 30–40.

Documented **approximations** (the paper's exact method needs assets it doesn't
ship):

- **Lexicon by seed-word matching**, not an LLM classification of the whole
  ~256k dictionary. The paper gets "1200 emotion tokens" from classifying every
  token; I match vocab tokens against per-emotion seed stems
  (`EMOTION_SEEDS`). Cheaper and offline, but the token sets differ from the
  paper's; treat absolute z-scores as indicative, the vanilla-vs-DPO *delta* as
  the result of interest.
- **Baseline over a token subset** (emotion ∪ random tokens) and **`--n-baseline`
  defaults to 20** WildChat texts vs the paper's 500 — to keep a single-GPU run
  tractable. Raise it for stable statistics.
- **Logit lens = final-norm + `lm_head`** applied to every hidden layer; the
  final-norm module is located robustly across Gemma-3's nested module names.

---

## 10. Reproducibility, cost, and running order

- **Determinism:** all sampling is seeded (`--seed`); judge/onset/paraphrase
  calls use temperature 0 (except paraphrase 0.7, which needs lexical variety).
  Target-model sampling is temperature 1 per the paper, so rollouts are
  intentionally stochastic.
- **Cost realism:** a full run is large (4000 rollouts × multiple turns ×
  several models, each turn a judge call). `--scale` exists precisely so a cheap
  smoke test (`--scale 0.02`) validates the whole pipeline before committing to
  a full sweep. This is why I also made every dataset have an offline fallback.
- **Suggested order:** `run_elicitation` (Gemma + Gemini) → `make_figures`
  (reproduces Fig 1/2/3) → `generate_dpo_data` → `run_finetune --method dpo` →
  `run_elicitation --adapter adapters/dpo_gemma` → `make_figures` (Fig 5,
  35%→0.3%) → optionally `run_prefill` (§3), `run_petri`, `run_capabilities`,
  `run_probing`.

---

## 11. Known limitations of this replication

- Not executed; see status note at top. Library-version drift (transformers/TRL)
  is the most likely source of first-run breakage.
- §3 prefill and App. I probe assume **local Gemma weights** (27B → a large GPU,
  or quantised loading via the commented `bitsandbytes` dependency).
- Gemini prefill is best-effort and Gemini cannot be fine-tuned/probed — an
  inherent limit of the closed model, acknowledged by the paper itself.
- The internal-probe lexicon and the capability parsers are approximations
  chosen for offline tractability; they support **relative** comparisons, which
  is what the paper's claims require, not absolute reproduction.
