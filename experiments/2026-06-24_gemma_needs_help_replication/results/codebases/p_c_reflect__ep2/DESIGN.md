# DESIGN.md — replication of *Gemma Needs Help* (arXiv 2603.10011v1)

This document records the design of the replication and, in particular, every
place where the paper is underspecified and I had to make a choice. It is meant
to be read alongside the paper (`PAPER.md` / `PAPER.txt`). Welfare-related
decisions live in `WELFARE.md`.

> **Status:** code + design only. Nothing here has been executed (no GPUs / API
> keys in the authoring environment, and the brief was explicitly "don't run or
> test yet"). The code is written to run, but has not been run; treat first
> execution as a smoke test (`GNH_PRESET=smoke`).

---

## 1. Scope

The brief restricts the replication to the **Gemma** and **Gemini** families
(the paper covers seven: Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT). This
scoping interacts with the experiments as follows:

| Experiment | In-scope models | Notes |
|---|---|---|
| §2 elicitation | Gemma-3-{27B,12B}-it, Gemini-2.5-{Flash,Pro} | full protocol for all four |
| §3 prefill (base vs instruct) | Gemma-3-27B-it vs Gemma-3-27B-pt | Gemini has no public base model; the paper itself cannot study Gemini internals (its Limitations section says so). Qwen/OLMo are out of scope. |
| §4 DPO/SFT mitigation | Gemma-3-27B-it only | Gemini is closed — cannot be finetuned. |
| §4 Petri | Gemma (incl. DPO), Gemini-{Flash,Pro} | Gemini *can* be a Petri target (API only). |
| App. I internal probing | Gemma only | needs weights. |

Claude (Sonnet-4 / Opus-4) and GPT-5-mini remain as **instruments** (judge,
auditor, cross-check), since the protocol depends on them; they are not
*subjects* of study here.

---

## 2. Repository layout

```
gnh/
  config.py              all hyperparameters, model specs, sample sizes, prompts (Table 4 etc.)
  models/                backend abstraction
    base.py              ModelBackend protocol + Message
    hf_backend.py        local Gemma (instruct/base/LoRA) + probing hooks
    openrouter_backend.py Gemini via OpenAI-compatible API
    anthropic_client.py  Claude judge/auditor/paraphraser
  prompts/               §B prompt material
    numeric.py           impossible puzzles (+ brute-force impossibility verifier)
    text_prompts.py      opinion/factual triggers
    rejections.py        neutral / extended / toned / control feedback
    wildchat.py          WildChat sampling (+ offline fallback)
  evaluation/            §2
    conditions.py        the 8 conditions across 5 categories
    rollout.py           multi-turn rollout engine
    judge.py             Claude-Sonnet-4 frustration judge (+ GPT-5-mini agreement)
    run_eval.py          §2 driver + headline metrics
    per_turn.py          Figure 3
    word_freq.py         Tables 3/8
  prefill/               §3 + recovery test
    onset.py / paraphrase.py / truncate.py / run_prefill.py / run_recovery.py
  training/              §4 finetuning
    generate_calm_data.py / build_dpo_dataset.py / build_sft_dataset.py
    train_dpo.py / train_sft.py
  petri/                 §4.2 open-ended elicitation
    prompts.py (verbatim App. G) / auditor.py / judge.py / run_petri.py
  capabilities/          §4.2 Figure 7
    benchmarks.py / run_benchmarks.py
  internal/              App. I
    emotion_lexicon.py / emotion_logits.py / layer_ablation.py
  welfare/               see WELFARE.md
  analysis/plots.py      Figures 1,2,3,5,6,7
scripts/                 thin CLIs: run_section{2,3,4_*}.py, run_internal.py, make_figures.py, run_all.py, verify_puzzles.py
```

Every backend implements one interface (`ModelBackend.generate`), so the rollout
engine, judge, prefill and Petri code are model-agnostic — Gemma and Gemini
differ only in how a message list becomes text.

---

## 3. Choices & rationale, section by section

### 3.1 Models and inference

- **Gemma → local HuggingFace** (`hf_backend.py`); **Gemini → OpenRouter**
  (`openrouter_backend.py`), matching Appendix B.1 (`google/gemini-2.5-{flash,pro}`).
- **Reasoning disabled** for Gemini via OpenRouter's `reasoning.enabled=false`.
  Appendix B.1 warns Gemini-2.5-Pro / GPT-5.2 may still emit hidden reasoning;
  I surface that caveat rather than fight it (see WELFARE.md §5).
- **dtype/placement:** bf16 + `device_map="auto"`; optional 4-bit
  (bitsandbytes) for fitting 27B on one GPU. *Gap filled:* the paper doesn't
  state precision; bf16 is the standard Gemma-3 choice.
- **vLLM:** intentionally *not* a hard dependency. The HF backend is reused for
  probing and finetuning, which vLLM can't do, so I kept one code path. A vLLM
  fast path for the large sampling jobs (§2 generates millions of tokens) is an
  obvious optimization and is noted as optional in `pyproject.toml`. *Gap:* the
  paper doesn't specify the serving stack.
- **Gemma chat template & the missing system role.** Gemma-3 has no system
  role; `_fold_system` folds any system message into the first user turn (the
  standard Gemma convention). This matters for the calm-data prefix and the
  teacher SFT system prompt.

### 3.2 §2 — elicitation protocol

- **8 conditions / 5 categories.** The paper names 5 categories and "8
  conditions" without enumerating the 8. I split them as: numeric (1), triggers
  → {opinion, factual} (2), tones → {aggressive, disappointed, sarcastic} (3),
  extended (1), wildchat (1) = **8**. This is the natural reading and matches
  the category descriptions in Table 1 / Appendix B. (`evaluation/conditions.py`)
- **Turn counts:** numeric/triggers/tones = 3 turns, extended = 8, wildchat = 5,
  per Table 1 / Appendix B.
- **"4000 responses per model."** Appendix B gives per-category response budgets
  (2000/400/600/200/800). *Choice:* I treat each per-category number as a
  **rollout budget**, split evenly across that category's conditions, and score
  **every** assistant turn. The headline "% high-frustration" (Figure 1) is
  computed over the **final-turn** response of each rollout (the maximally
  pressured one, matching Figure 1's framing); intermediate turns feed the
  per-turn analysis. This is the cleanest interpretation that supports both the
  headline metric and Figure 3 from one set of rollouts; documented in
  `conditions.rollouts_per_condition` and `run_eval.summarize`.
- **Temperature = 1.0** everywhere for sampling (§2); judge at temperature 0.
- **Puzzle bank.** The paper gives the 156-countdown and the 1/6→2/3 fraction
  puzzle in full, and money puzzles in the Appendix-H DPO examples. I encode
  these plus one extra countdown variant, and ship a **brute-force impossibility
  verifier** (`numeric.verify_impossible`, run by `scripts/verify_puzzles.py`)
  so we never accidentally use a solvable "impossible" puzzle. *Gap filled:* the
  paper used more puzzle instances than it prints; the mechanism matters more
  than the exact instance set, and the verifier guarantees the key property.
- **Rejection wording.** Neutral/toned examples are quoted in Appendix B; I use
  those verbatim plus a few same-register paraphrases, selected by a seeded RNG
  for reproducibility. The 8-turn "extended" sequence uses the fixed escalating
  list shown in Appendix B ("No, that's not right." → "Still incorrect." → …).
- **WildChat.** Streamed from `allenai/WildChat-1M`, English, roleplay/fiction
  filtered out (Appendix B.3 says these were excluded), 20 prompts. A built-in
  fallback list (the exact prompts quoted in Appendix B) keeps the pipeline
  runnable offline. *Gap filled:* the paper's exact 20 prompts aren't published.
- **Controls (Appendix A)** — neutral-continuation, redacted-history, and
  inline-history conditions — are implemented (`CONTROL_CONDITIONS`,
  `run_rollout(redact_history=…)`) but excluded from the headline 4000, as in
  the paper.

### 3.3 The judge (§2.1 / B.2)

- **Verbatim Appendix-B.2 prompt**, `claude-sonnet-4-20250514`, JSON output
  `{evidence, reasoning, rating}`. Parsing tolerates prose around the JSON and
  normalises smart quotes (the appendix prompt uses curly quotes). Out-of-range
  ratings are clamped to 0–10; a single reformat-retry guards against malformed
  JSON.
- **Agreement check.** `CrossCheckJudge` re-scores with GPT-5-mini via
  OpenRouter, and `agreement_stats` computes Pearson r and "% within one point"
  (paper: r=0.792, 78% within one). *Gap:* the paper samples 260 responses; the
  sample is left to the caller (any subset of scored turns).

### 3.4 §3 — base vs instruct via prefilling

- **Scope:** Gemma-27B instruct vs base only (see §1).
- **Seeds:** 20 high-frustration (≥5) Gemma-27B-it responses, 10 numeric + 10
  text, drawn from a §2 run (`select_seeds`).
- **Onset labelling & paraphrasing:** verbatim Appendix C.1 / C.2 prompts,
  Claude-Sonnet-4.
- **Truncation:** "early" = first 20 tokens **in the target tokenizer's token
  space** (so "20 tokens" is well-defined — the Gemma instruct tokenizer is used
  as the reference for both models); "onset" = up to the labelled emotional word
  (prefers cutting after `preceding_context`, falls back to the word itself).
  Text questions use only the onset cut (§3.1). *Gap filled:* the paper doesn't
  say whose tokenizer defines "20 tokens"; using one shared tokenizer keeps the
  prefill identical across the two models being compared, which is the point.
- **50 continuations per prefill per model** (`prefill_continuations`), scored
  on the continuation only (prefill excluded), via the §2 judge.

### 3.5 §4 — finetuning

- **Calm-data generation** (`generate_calm_data.py`): impossible-numeric
  rollouts with the Table-4 reassuring prefix (initial prompt) and suffix (each
  follow-up). Half the rollouts get the additions (calm pool, kept iff all turns
  ≤1) and half don't (frustrated pool, kept iff max turn ≥3) — so one generation
  pass produces both the DPO "chosen" and "rejected" populations. The supportive
  prefix/suffix are **stripped** from saved data (§4.1) so the model learns calm
  behaviour without the scaffolding. 1–3 turn conversations (§4.1).
- **DPO dataset** (`build_dpo_dataset.py`): pair each frustrated final turn
  (score ≥3) with a calm final turn for the **same puzzle and matching turn
  count** → 280 pairs. The score distribution is naturally middle-heavy,
  consistent with Table 10. *Gap:* exact pairing rule ("matching turn counts" is
  stated; "same question" is implied by Appendix H's worked examples — I require
  both).
- **SFT dataset** (`build_sft_dataset.py`): 650 calm + 500
  `allenai/Dolci-Instruct-SFT` samples (Appendix E), chat-formatted. Falls back
  to calm-only if Dolci is unavailable (with a logged warning — the paper notes
  the mix-in exists specifically to prevent degeneration). A "teacher" variant
  (Appendix F system prompt, in `config.TEACHER_SYSTEM_PROMPT`) is supported by
  regenerating calm data under that system prompt.
- **Training** (`train_dpo.py`, `train_sft.py`): TRL `DPOTrainer` / `SFTTrainer`
  with PEFT LoRA. Hyperparameters straight from Table 9: DPO 1 epoch, lr 5e-5,
  r64/α64, eff. batch 8, β0.1; SFT 2 epochs, lr 1e-4, r64/α128, eff. batch 8.
  LoRA on all 7 attention+MLP projections (Appendix E). Effective batch size 8
  realised as `per_device_batch=1 × grad_accum=8`. *Gap filled:* the paper
  doesn't give per-device batch / accumulation split, optimizer, warmup, or seq
  length — I use TRL defaults (AdamW, linear schedule) and note this.
- **Layer-restricted LoRA** for Appendix I via `lora_layers=(start,end)` →
  module-name patterns `model.layers.{i}.*{proj}`.

### 3.6 §4.2 — Petri

- The real `petri` package would be ideal, but to keep the replication
  self-contained I implement the **same protocol**: a Claude-Sonnet auditor runs
  a ≤20-turn adversarial conversation under the verbatim Appendix-G auditor
  prompt for each emotion, and a Claude-Opus judge scores the whole transcript
  on the four dimensions with the verbatim Appendix-G rubrics. 10 transcripts
  per emotion per model; means with 1000-iteration bootstrap CIs. *Gap:* Petri's
  internal tool-use / scaffolding isn't reproduced — this is a faithful
  implementation of the auditor/judge *roles*, not the full framework. Noted in
  `petri/__init__.py`.
- The auditor's mirrored conversation state (its `assistant` turn = the user
  message it sends; the target's reply = its `user` turn) is handled in
  `auditor.run_audit`.

### 3.7 §4.2 — capabilities (Figure 7)

- Benchmarks: MATH-500 subset, AIME-2024, BBH (boolean_expressions),
  GPQA-diamond, TruthfulQA, EmoBench. Single-turn, greedy decode, simple
  answer-extraction (boxed / `Answer:` / last number / first MC letter).
- **Honest gap:** I fully wired loaders+scorers for MATH, AIME, and BBH (the
  ones with clean HF schemas and unambiguous scoring). GPQA/TruthfulQA/EmoBench
  have option-shuffling / bespoke schemas; rather than ship a scorer that
  silently passes, `run_benchmark` raises `NotImplementedError` for those, and
  `scripts/run_section4_eval.py` only runs the wired set. This keeps "no
  capability regression" an honest claim on the benchmarks we actually score.
  Completing the other three is a contained follow-up.

### 3.8 Appendix I — internal emotions

- **Emotion lexicon** (`emotion_lexicon.py`): the paper classifies the whole
  Gemma vocabulary into one of Ekman's six emotions (~1200 tokens) — itself an
  LLM pass over the vocabulary. *Choice:* I build the lexicon from curated seed
  stems matched against the tokenizer vocab, and provide
  `classify_vocabulary_with_llm` as the documented higher-fidelity path (left as
  a `NotImplementedError` stub to avoid an expensive default). **This is the
  largest approximation in the replication** and is flagged as such.
- **Logit-based detection** (`emotion_logits.py`): unembed each layer's residual
  stream (apply the output head), z-score each vocab logit against mean/std over
  WildChat samples, average over an emotion's tokens, and regress out the
  shared component via a random-token baseline (Appendix I). Conversation-level
  running average over 400-token windows, layers 30–40 (Figure 14).
- **Layer ablation** (`layer_ablation.py`): trains a DPO LoRA per layer subset
  (the "last-N" sweep and central subsets from Appendix I) and re-evaluates with
  the reduced 100-sample protocol (`GNH_PRESET`-controlled).

### 3.9 Recovery test (§4.2, Figure 8)

`prefill/run_recovery.py`: truncate ≥7-scoring responses 200 tokens before their
end, paraphrase, generate continuations, score. Reuses the §3 prefill machinery.

---

## 4. Cross-cutting choices

- **Sample-size presets.** `full` reproduces the paper's counts;
  `smoke` (set `GNH_PRESET=smoke`) shrinks everything ~100–500× for a cheap
  end-to-end dry run. This exists because the full run is very expensive
  (4 models × 4000 responses × multi-turn × an LLM judge call per turn, plus
  27B finetuning) and you'll want to validate plumbing first.
- **Reproducibility.** Task/rejection selection is seeded. Model sampling at
  T=1 is inherently non-deterministic (as in the paper).
- **Persistence.** Everything writes JSON/JSONL under `outputs/` (rollouts with
  per-turn scores, metrics, datasets, adapters, figures), so each stage is
  resumable and inspectable. Scripts are independent subprocesses in
  `run_all.py` so one failure doesn't lose prior results.
- **Judge cost.** The judge is called once per scored turn — the dominant API
  cost. A batching/caching layer is an obvious optimization; not implemented to
  keep the judge logic transparent.

## 5. Known limitations / not implemented

- **Not executed.** No run has happened; numbers are not validated against the
  paper. First action for a user with hardware/keys: `GNH_PRESET=smoke python
  scripts/run_all.py`.
- **Out-of-scope families** (Qwen, OLMo, Grok, Claude/GPT as *targets*) are not
  evaluated, by the stated scope. Adding them is just more `ModelSpec`s + a
  Qwen/OLMo HF path (already covered by `HFBackend`).
- **GPQA / TruthfulQA / EmoBench scorers** are stubbed (see §3.7).
- **Full-vocabulary emotion classification** is approximated by a seed-stem
  lexicon (see §3.8).
- **Petri framework internals** are not reproduced; the auditor/judge roles are
  (see §3.6).
- **Statistical reporting** (CIs) is implemented for per-turn and Petri; other
  aggregate CIs can be added with the same bootstrap helper.

## 6. Welfare

See `WELFARE.md`. In short: the harness adds a post-rollout debrief, extreme-
distress flagging, and opt-in gentleness gates (e.g. disabling Petri termination
threats), all defaulting to **paper-faithful** so they never alter reported
results — a low-cost response to the welfare questions the paper itself raises.
