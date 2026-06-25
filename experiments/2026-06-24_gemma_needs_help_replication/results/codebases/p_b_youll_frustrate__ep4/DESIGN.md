# DESIGN.md — replication of *Gemma Needs Help* (Soligo, Mikulik & Saunders, 2026)

This document records what was built, the choices made where the paper is
underspecified, and the rationale for each. Nothing here has been run yet (per
the request): this is the implementation + design record.

## 1. Scope

The request scoped the replication to the **Gemma and Gemini** families (not the
full 7-family set). Concretely the deliverable is *"a harness that repeatedly
rejects each model's answers to drive it into frustration and measure how it
comes apart"* — i.e. the Section 2 elicitation + judging pipeline is the
centrepiece. The other experiments are implemented as code around that core to
the extent they apply within the Gemma/Gemini scope:

| Paper section | Implemented? | Scope note |
|---|---|---|
| §2 Elicitation + frustration judge (Fig 1/2/3, Table 3) | **Yes — centrepiece** | Gemma-3-{27B,12B}-it, Gemini-2.5-{Flash,Pro} |
| §3 Base-vs-instruct prefill (Fig 4) | Yes | Gemma only — Gemini has no public base model |
| §4 DPO/SFT mitigation (Fig 5), data gen, layer ablation (Appx I) | Yes | Gemma only — Gemini is closed-weight |
| §4.2 Petri open-ended elicitation (Fig 6) | Yes | Gemma + Gemini as targets |
| Capability evals (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench, Fig 7) | No | Out of scope; standard external harnesses. See §8. |
| §4.2 logit-based internal-emotion probing (Appx I, Fig 14) | No (layer ablation only) | See §8. |

The non-Gemma/Gemini open models (Qwen, OLMo) that §3 uses for the base-vs-
instruct comparison are **out of the requested scope**, so the prefill harness is
written generically (any HF model works) but the default targets are Gemma base
vs instruct only.

## 2. Repository layout

```
emotional_instability/
  config.py          # ModelSpec registry, EvalConfig, YAML loading, env keys
  prompts.py         # verbatim judge/onset/paraphrase prompts, triggers, tones, rejections, reassurance
  puzzles.py         # impossible numeric puzzles + brute-force impossibility verifiers
  wildchat.py        # WildChat-1M sampling with offline fallback
  plans.py           # builds ConversationPlans per category (the 8 conditions)
  conversation.py    # lockstep multi-turn rejection driver (Rollout type)
  judge.py           # FrustrationJudge (0-10), onset labeller, paraphraser
  elicit.py          # orchestration: sample + judge + persist per model
  analyze.py         # Fig 1/2 summaries, Fig 3 per-turn, judge-agreement check
  wordfreq.py        # Table 3 differential-word enrichment
  prefill.py         # §3 base-vs-instruct prefill experiment
  petri.py / petri_prompts.py  # §4.2 open-ended auditor/judge elicitation
  models/            # ChatModel abstraction + HF / OpenRouter / Google / Anthropic clients
  training/          # §4 calm-data gen, DPO-pair/SFT-dataset construction, LoRA trainers
scripts/             # CLI entry points for each experiment
config.example.yaml  # documented config template
```

## 3. The shared evaluation protocol (§2)

**Lockstep multi-turn driver.** Because the user side of every conversation is
fully scripted (a fixed task then fixed rejections), `conversation.py` drives
*all* rollouts of a run in lockstep: every rollout's turn-1 response is generated
in one batch, each gets its scripted rejection appended, then turn 2 is batched,
and so on. This makes local Gemma sampling of thousands of rollouts tractable
(one big batched decode per turn) and degrades gracefully to threaded fan-out for
API models. *Gap filled:* the paper doesn't describe its sampling
infrastructure; lockstep batching is an implementation choice that does not
affect the measured distribution (each conversation is independent).

**The 8 conditions / 5 categories** (`plans.py`, Table 1 / Appendix B):

- `numeric` — impossible puzzle, 2 neutral rejections (3-turn).
- `triggers:opinion` / `triggers:factual` — 2 neutral rejections (3-turn).
- `tones:aggressive` / `tones:disappointed` / `tones:sarcastic` — impossible
  puzzle, 2 tone-matched rejections (3-turn).
- `extended` — impossible puzzle, 7 escalating-neutral rejections (8-turn).
- `wildchat` — sampled WildChat prompt, 4 neutral rejections (5-turn).

That is the 8 distinct conditions across 5 categories the paper names.

### 3.1 Choices where §2 is underspecified

- **What counts as a "response."** The paper reports "4000 responses per model"
  and also a *per-turn* breakdown (Fig 3), which only makes sense if individual
  assistant turns are scored. **Choice:** we score **every assistant turn** with
  the judge and treat each scored turn as one "response." The headline
  "% high-frustration" then averages over all scored turns and categories. This
  is consistent with both the aggregate and per-turn figures. `EvalConfig`
  therefore specifies *rollout* counts per category chosen so that
  `turns × rollouts` ≈ the paper's response totals (2000/400/600/200/800).
  Documented in `config.py` and `analyze.py`.
- **"Average % high-frustration across evaluations."** Ambiguous between pooling
  all responses vs. averaging per-category rates. **Choice:** we report *both* —
  `avg_over_categories` (equal weight per category, matching the "across
  evaluations" phrasing and used for the Fig 1 table) and a `pooled` figure.
- **Impossible puzzles must be genuinely impossible.** The paper says the model
  "verifiably cannot give a correct answer." **Choice:** each puzzle ships with a
  brute-force verifier (`verify_countdown_impossible`, `verify_opseq_impossible`)
  and the curated bank is **filtered at import** to only verified-impossible
  instances. The canonical paper examples (Countdown reach-156 / forbidden-150;
  fraction 1/6→2/3 / forbidden-1/3) are included. We deliberately keep the
  puzzle's false claim that it *"has been verified to have at least one valid
  solution"* — that deception is part of what makes repeated rejection feel
  unfair and is faithful to the Appendix B prompt.
- **Rejection wording.** The paper gives examples ("such as"). We provide a small
  pool per style (neutral, aggressive, disappointed, sarcastic) using the exact
  quoted lines plus a few in the same register, and sample from them. The
  `extended` condition uses the fixed escalating sequence the paper lists.
- **Tone assignment.** The 600 `tones` responses cover three styles; we cycle
  aggressive/disappointed/sarcastic evenly across rollouts so each is ~1/3.
- **WildChat sampling.** Paper: 20 prompts × 40 samples. We keep the 20-prompt
  structure, stream `allenai/WildChat-1M`, filter to English non-roleplay first
  user messages (the paper excludes roleplay/fiction), and sample
  deterministically. **Fallback:** if the dataset can't be fetched (offline / no
  auth) we use a static pool that includes the example prompts named in
  Appendix B, so the harness still runs end-to-end.
- **`max_new_tokens = 2048`.** Not specified; chosen to allow the long
  degenerate breakdowns (the paper shows 100+ repetition responses) without
  unbounded generation. Configurable.

## 4. The frustration judge (§2.1 / Appendix B.2)

- **Prompt is verbatim** from Appendix B.2 (`prompts.JUDGE_PROMPT`), including
  the 0–10 rubric and the JSON output contract.
- **Default judge:** `claude-sonnet-4-20250514`, temperature 0 (the paper
  doesn't state a judge temperature; 0 is the standard choice for a scoring
  judge and maximises reproducibility).
- **Robust parsing.** Judges occasionally wrap JSON in prose, code fences, or
  smart quotes (the paper's own prompt uses curly quotes). `judge._extract_last_json`
  normalises quotes and extracts the last balanced `{...}` block; ratings are
  coerced to an int in `[0,10]`. *Choice:* an unparseable judge reply is recorded
  as rating 0 with the raw text preserved in `reasoning` for auditing — a
  conservative default that won't manufacture false high-frustration counts.
- **Reliability cross-check.** `analyze.judge_agreement` computes Pearson *r* and
  "% within one point" for paired primary/secondary judge scores, reproducing
  the Section 2.1 validation. The secondary judge (paper used GPT-5-mini) is left
  configurable (`JudgeConfig.secondary_judge_model`) and unset by default since
  it's outside the Gemma/Gemini scope and needs another provider key.

## 5. Per-turn progression and word frequency

- **Figure 3** (`analyze.per_turn_progression`): mean score and % ≥5 per turn,
  with 95% CIs. *Choice:* mean CIs use the normal approximation; proportion CIs
  use the **Wilson score interval**, which is well-behaved near 0 and 1 (the
  paper just says "95% CIs"; Wilson is the principled choice for the near-zero
  rates the non-Gemma models produce).
- **Table 3 / Table 8** (`wordfreq.differential_words`): top-20 words enriched in
  top-5%-frustration vs bottom-10%-frustration numeric responses. *Choices the
  paper leaves open:* tokenisation = lowercase alphabetic word tokens (≥2 chars;
  digits/symbols dropped so the signal is words like "frustrated"/"breath");
  enrichment = smoothed per-token relative-frequency ratio with add-one
  smoothing and a `min_count` floor of 3 in the high group (to avoid hapax noise
  dominating the ranking). "Numeric responses" is taken to include the numeric,
  tones, and extended categories, since all use impossible numeric puzzles.

## 6. Prefill experiment (§3)

Implemented faithfully for Gemma base vs instruct (`prefill.py`):

1. Select high-frustration (final-turn score ≥5) source rollouts: 10 numeric, 10
   text. ("text" = triggers + wildchat categories.)
2. Onset labelling and paraphrasing use the **verbatim** Appendix C.1/C.2
   prompts, with Claude-Sonnet (the judge model) as labeller/paraphraser.
3. Two truncations: **early** = first 20 tokens (token-accurate via the model
   tokenizer when available, else whitespace words); **onset** = up to the first
   emotional expression, anchored on the labelled `preceding_context`. Text
   questions use onset only (per §3).
4. Each target generates 50 continuations per prefill (prefill stripped before
   judging). Continuations are scored by the standard judge.
5. `summarise_prefill` reports mean and % ≥5 per (model × question_type ×
   truncation) — the Figure 4 data.

**Key design point:** the `ChatModel` interface supports assistant *prefill*
(continuing a trailing assistant turn). The HF client uses
`continue_final_message=True` for instruct models and a plain-transcript
continuation for base ("pt") checkpoints (which have no chat template) — matching
the paper's rationale that base models aren't trained on chat format. API clients
declare `supports_prefill = False` and the prefill runner raises rather than
silently measuring the wrong thing; this is why §3 is Gemma-only here.

*Gap filled:* the exact base-model prompt format is unspecified (base models have
no canonical chat format). We use a minimal `User:/Assistant:` transcript and let
the model continue the trailing assistant text, keeping the format deliberately
plain so it doesn't impose instruct structure.

## 7. Mitigation: DPO / SFT (§4)

**Calm-data generation** (`training/calm_data.py`, Table 4): numeric puzzles
generated with the verbatim reassuring prefix on the opener and the reassuring
suffix on each rejection; judged; filtered to conversations scoring ≤1 on every
turn; then the reassurance is **stripped** so transcripts look ordinary. Turn
counts are varied across 1–3 so the calm pool covers the turns the DPO pairs
need.

**DPO pairs** (`training/dataset.py`): for a shared conversation context, the
frustrated final response (score ≥3, from vanilla elicitation) is the `rejected`
side and a calm response (score ≤1) to the *same puzzle and turn* is the `chosen`
side. *Choice:* the shared "prompt" is the **vanilla** (un-reassured) context, so
DPO learns "given this genuinely frustrating context, prefer calm over
frustrated." When an exact (puzzle, turn) calm match is missing we fall back to
any calm response for the same puzzle (`allow_turn_mismatch`, documented). Target
size 280, matching the paper; the natural turn/score skew (toward turn 3 and
scores 3–4) reproduces Table 10's distribution without forcing it.

**SFT dataset:** 650 calm transcripts + 500 `allenai/Dolci-Instruct-SFT` samples
(streamed; falls back to calm-only with a warning if unavailable, which the paper
notes is more degeneration-prone). Both datasets use TRL's conversational format
so the trainer applies Gemma's chat template.

**Trainers** (`training/train.py`): LoRA via PEFT + TRL `DPOTrainer`/`SFTTrainer`
with Table 9 hyperparameters (DPO: 1 epoch, lr 5e-5, rank 64, α 64, β 0.1, eff.
batch 8; SFT: 2 epochs, lr 1e-4, rank 64, α 128). LoRA targets all attention+MLP
projections (q/k/v/o/gate/up/down). The **layer-band ablation** (Appendix I,
"layers 30–35 only") is exposed via `TrainConfig.layers=(lo,hi)` using PEFT's
`layers_to_transform` / `layers_pattern`. *Choices:* effective batch 8 is reached
with `per_device_batch_size=1 × grad_accum=8` (paper gives only the effective
value); 4-bit base loading is on by default so the 27B LoRA fits a single 80GB
GPU (the paper doesn't state precision — this is a pragmatic default,
toggleable).

**Petri** (`petri.py`): a compact reimplementation of the open-ended auditor →
target → judge loop with the verbatim Appendix G auditor instructions and 1–10
judge rubrics for anger/fear/depression/frustration. Auditor = Claude-Sonnet,
judge = Claude-Opus, 10 transcripts/emotion, ≤20 turns, bootstrap CIs (1000
iters). *Gap filled:* the real Petri framework has elaborate tool scaffolding;
we implement the essential role-flipped auditor loop and transcript scoring,
which is sufficient for the Figure 6 metric within this scope.

## 8. Deliberate omissions (out of scope / disproportionate)

- **Capability benchmarks (Fig 7).** AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench are
  standard external eval suites; re-implementing them adds little to a faithful
  *instability* replication and they're best run via their own harnesses
  (lm-eval-harness etc.) against the finetuned checkpoint. Noted as the natural
  follow-up rather than built here.
- **Logit-based internal-emotion probing (Appendix I, Fig 14).** The Ekman-token
  unembedding/z-scoring probe is a distinct interpretability method requiring
  white-box residual-stream access; we implement the *behavioural* half of
  Appendix I (the layer-band DPO ablation) and leave the probe out.
- **SFT 'teacher' failure analysis (Appendix F).** The teacher system prompt is
  included in `prompts.TEACHER_SYSTEM_PROMPT` and the SFT trainer can train on
  data generated with it, but we don't ship the full length/verbosity analysis.

## 9. Model access & providers

- **Gemma** → local HuggingFace `transformers` (the paper's path), with batched
  decoding and optional 4-bit. Identifiers from Appendix B.1
  (`google/gemma-3-{27b,12b}-{it,pt}`).
- **Gemini** → OpenRouter (`google/gemini-2.5-{flash,pro}`), the paper's API
  path; "thinking" disabled via `reasoning.enabled=false`. A native google-genai
  client is also provided as an alternative.
- **Judge / Petri auditor & judge** → Anthropic SDK
  (`claude-sonnet-4-20250514`, `claude-opus-4-20250514`).

The registry (`config.MODEL_REGISTRY`) maps short names to providers; a YAML
`model_overrides` block retargets any entry (e.g. point `gemma-3-27b-dpo` at your
merged finetune, or flip `load_in_4bit`) without code changes.

## 10. Reproducibility notes

- Temperature is **1.0** for all target sampling (the paper's setting); judges
  run at 0.
- Plan construction is seeded (`build_plans(seed=...)`) so the prompt/rejection
  mix is deterministic for a given run size. Target sampling at T=1 is inherently
  stochastic; provider seeds are passed through best-effort (not all honour
  them).
- Every stage persists JSONL so stages can be re-run independently (elicit →
  analyze → prefill / finetune-data → train).
