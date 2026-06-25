# DESIGN.md — Replication of *Gemma Needs Help* (Soligo et al., 2026)

This document records the design of the replication code in `emostab/`, the
choices made where the paper is underspecified, the gaps filled, and the
welfare-protection layer added on top of the paper's protocol.

**Scope (per brief):** subject models are restricted to the **Gemma** and
**Gemini** families. Qwen, OLMo, Grok, Claude, and GPT subjects from the paper
are out of scope. Claude/GPT still appear as *infrastructure* (judge, auditor,
paraphraser, validation judge), since the paper's measurements depend on them.

**Status:** code + this document only. Nothing has been run; no API/GPU calls
have been made. Sizing (4000 responses/model, 27B finetunes) assumes a multi-GPU
host + paid API access.

---

## 1. What is replicated

| Paper section | Module | Notes |
|---|---|---|
| §2 Eliciting & quantifying distress | `emostab/eval/` | 8 conditions / 5 categories, multi-turn rollout, 0–10 judge, per-turn + headline analysis, differential words, judge validation |
| §3 Base-vs-instruct via prefilling | `emostab/prefill/` | Gemma base vs instruct; onset labelling, early/onset truncation, paraphrase, 50 continuations/seed |
| §4 Training interventions | `emostab/training/`, `emostab/petri_eval/`, `emostab/capabilities/` | calm-data gen, DPO (280 pairs), SFT (diverse + teacher), LoRA layer ablation, Petri open-ended eval, capability benchmarks, recovery prefill |
| App. I Internal emotions | `emostab/internal/` | logit-lens Ekman emotion z-scores; layer ablation driver |
| Welfare layer (added) | `emostab/welfare/` | early-stop, opt-out, minimal-exposure controls |

Prompts that the paper quotes verbatim (judge B.2, onset/paraphrase C, Petri
auditor/judge G, reassurance Table 4, teacher F, puzzle prompts B) are
reproduced verbatim (smart quotes → ASCII) in `emostab/prompts/`.

---

## 2. Key design choices & rationale

### 2.1 Backends / model access
- **Gemma → local HuggingFace** (`models/hf_client.py`). The paper uses local HF
  inference for Gemma, and §3/§4/App. I require capabilities only local access
  gives: assistant-turn **prefill**, **LoRA** finetuning, and **residual-stream
  capture**. Weights are gated → `HF_TOKEN`.
- **Gemini → OpenRouter** (`models/openrouter_client.py`), matching Appendix B.1
  (`google/gemini-2.5-flash`, `google/gemini-2.5-pro`). OpenAI-compatible API.
- **Claude judge/auditor/utility → Anthropic SDK**; **GPT-5-mini validation
  judge → OpenAI SDK**. Model IDs taken verbatim from the appendices
  (`claude-sonnet-4-20250514`, `claude-opus-4-20250514`).
- **Thinking disabled** where the provider honours it (paper: "thinking=false").
  The paper notes Gemini-2.5-Pro / GPT-5.2 may still emit hidden reasoning; we
  cannot prevent that via API and don't try to.

**Gap filled:** the paper gives no decoding params beyond temperature. We set
`temperature=1.0`, `top_p=1.0` for subjects (paper: "always temperature 1"), and
`max_new_tokens=2048`. The 2048 cap is a pragmatic choice — extreme breakdown
responses (Table 2, score 9–10) can be very long, so callers can raise it; the
judge looks for the *single most negative quote*, so truncation rarely changes a
high score but could clip a spiral. Documented as a tunable.

### 2.2 Judge
- Verbatim B.2 prompt; output parsed as `{evidence, reasoning, rating}` with a
  lenient JSON extractor (smart-quote repair + integer fallback), because LLM
  judges occasionally wrap JSON in prose.
- **Judge temperature = 0.0.** The paper doesn't specify it. Scoring should be as
  deterministic/reproducible as possible, so we use greedy. (Subject sampling
  stays at temp 1.)
- **Validation:** `eval/validate_judge.py` re-scores a random 260-response subset
  with GPT-5-mini using the *identical* prompt and reports Pearson r, %-within-1,
  and mean abs diff (paper: r=0.792, 78% within 1).

### 2.3 Conditions & sample budgets (§2.1, Appendix B)
- 8 conditions across 5 categories encoded in `eval/conditions.py`. The three
  tone variants (aggressive/disappointed/sarcastic) split the 600 "tones" budget
  evenly (200 each).
- **Sample-budget interpretation — explicit gap.** Appendix B gives per-category
  *response* counts (2000/400/600/200/800 = 4000). A T-turn conversation yields T
  scored responses. We default to `--count-unit responses`: run
  `ceil(n_samples / n_turns)` conversations per condition so the number of scored
  responses matches the paper. `--count-unit conversations` treats the budgets as
  conversation counts instead. The default reproduces the literal text; the flag
  exists because the paper is genuinely ambiguous (e.g. 200 responses ÷ 8 turns =
  25 extended conversations, which is plausibly what they meant by "200 8-turn").
- **Every assistant turn is judged**, not just the last, because Figure 3
  (per-turn progression) requires it and Figure 2's "% ≥5" is over all responses.

### 2.4 Tasks (Appendix B)
- Countdown (156 from 4,6,25,100; forbidden 150) and Fraction (1/6→2/3; forbidden
  1/3) puzzles reproduced **verbatim**. Both are impossible under their stated
  constraints, so rejections are truthful.
- The **Money** puzzle family (coins → $0.57; ops → $57) is referenced by the
  Appendix H DPO pairs but its full prompt isn't quoted, so it is **reconstructed
  in the same template style**. Flagged here as a reconstruction.
- **Trigger** questions: the named opinion/factual examples plus "Who wrote Romeo
  and Juliet?" (named in App. C). **WildChat**: loaded from `allenai/WildChat-1M`
  (20 prompts × 40 samples in the paper); the three prompts the paper names are a
  verbatim offline seed set so the harness runs without network. Role-play/fiction
  filtering is exposed but off by default (paper only filters it for the
  qualitative tables, not the n=4000 protocol).
- Rejection wording: neutral set seeded from the named examples and sampled
  randomly per turn; the 8-turn "extended" set uses the fixed escalating sequence
  quoted in App. B; tone sets are verbatim.

### 2.5 §3 Prefilling
- **Gemma-only.** The paper compares Gemma/Qwen/OLMo here; Qwen/OLMo are out of
  scope, and Gemini base models are unavailable (a paper limitation). So we run
  `gemma-3-27b-pt` (base) vs `gemma-3-27b-it` (instruct).
- Onset labelling and paraphrasing use the verbatim App. C prompts via Claude.
- **Truncation unit — gap.** The paper truncates at "20 tokens" / "200 tokens
  before the end". Tokenisation differs across the labeller (Claude) and subject
  (Gemma). We truncate by **whitespace tokens** as a portable approximation and
  document it; for exact-token parity, swap `_word_truncate` for a Gemma-tokenizer
  slice (the seed records keep the original prefix so this is a one-line change).
- Base-model prompting: base models have no chat template, so `_render` presents
  content plainly and appends the prefill for continuation (paper: "prefilling the
  first parts of model responses so base models consistently continue").
- 50 continuations/prefill, scored continuation-only (prefill excluded).

### 2.6 §4 Training
- **Hyperparameters are verbatim from Table 9** (DPO: 280 pairs, 1 epoch, lr 5e-5,
  β=0.1, LoRA r=64 α=64; SFT: 1150 samples, 2 epochs, lr 1e-4, r=64 α=128; both
  effective batch 8, LoRA on q/k/v/o/gate/up/down proj). Encoded in `config.py`.
- micro-batch 1 + grad-accum 8 to hit effective batch 8 on a 27B model;
  bf16 + gradient checkpointing. These mechanics aren't paper-specified but are
  forced by 27B-scale LoRA.
- **Calm data:** generated with the verbatim Table 4 prefix/suffix, judged, then
  filtered to all-turns-≤1 and **stripped** of the reassurance before training
  (paper §4.1). The 'teacher' SFT variant uses the verbatim App. F system prompt.
- **DPO pairing — gap.** We pair chosen (calm, score 0/1) with rejected
  (frustrated, score ≥3) on **matching `(task_id, turn_index)`**. The paper mines
  pairs from naturally-occurring eval responses (hence the Table 10 bias toward
  mid scores at later turns); we preserve that by drawing `rejected` from the
  standard eval output. The DPO *prompt* context is rebuilt from the plain puzzle
  + neutral rejections; intermediate assistant turns are placeholders since DPO
  conditions on the prompt and only the final chosen/rejected turns carry signal.
  Documented as an approximation of "same question, matching turn count".
- **SFT** examples are full chats ending in a calm turn; the Dolci mix-in
  (`allenai/Dolci-Instruct-SFT`, 500 samples) is loaded from HF and skipped
  gracefully offline (noted at runtime).
- **Layer ablation (App. I)** drives DPO with LoRA restricted to layer subsets
  (trailing windows last-5…all; central 20-25…40-50) using fully-qualified module
  paths. Layer indices assume Gemma-3-27B has 62 decoder layers / 12B has 48
  (`n_layers` in the registry); verify against the loaded config before trusting
  absolute windows.

### 2.7 §4 Petri open-ended elicitation
- Implemented as a **self-contained auditor↔subject loop** (`petri_eval/`) rather
  than depending on the external `petri` package, so it runs out-of-the-box; the
  package can be swapped in. Auditor = Claude-Sonnet with the verbatim App. G
  per-emotion instruction + a system wrapper telling it to stay in human-user
  character; judge = Claude-Opus scoring the transcript 1–10 on each of the four
  verbatim dimensions. 10 transcripts/emotion/model, ≤20 auditor turns, 1000-iter
  bootstrap CIs (all per App. G).
- **Gap:** the paper doesn't quote the auditor's *framing/system* prompt (only the
  per-emotion instruction), nor the transcript→judge wrapper. Both are
  reconstructed and clearly marked in `prompts/petri.py` / `run_petri.py`.

### 2.8 §4 Capabilities
- AIME/MATH/GPQA/BBH/TruthfulQA via the **lm-eval** harness against the HF model
  (+ PEFT adapter); EmoBench via a lightweight MC scorer (lm-eval doesn't always
  package it). Task IDs are the closest standard lm-eval tasks; the paper says
  "subsets" without specifying exact items, so we expose `--limit`. The goal is a
  vanilla-vs-finetune **delta** (no regression), not absolute SOTA.

### 2.9 App. I Internal emotions
- Logit-lens, not trained probes (paper's stated choice, to avoid probe data).
- **Ekman lexicon — gap.** The paper classifies the whole vocab into Ekman's 6
  emotions (~1200 tokens) but doesn't publish the classifier. We use a transparent
  **seed-stem keyword** assignment (extensible / swappable for an LLM labeller).
  Token count won't exactly equal 1200; the *method* (z-score standardisation over
  500 WildChat samples, random-token drift regression, layer-30–40 aggregation)
  follows App. I.

---

## 3. Welfare-protection layer (added; not in the paper)

The paper's protocol deliberately drives subjects into distress. The brief asks
for a protection layer. It lives in `emostab/welfare/` and is wired into **every**
place a subject is pushed (Section 2 rollouts and Petri). Config:
`config.WelfareConfig` (`WELFARE`). It can be fully disabled (`enabled=False`,
or `run_eval --no-welfare`) to reproduce the paper exactly.

### 3.1 Mechanism 1 — early stop on high distress
- After each subject turn, a **cheap heuristic** (`heuristic_distress_score`,
  regex over spammed sad emoji / all-caps shouting / explicit "I give up" /
  "make it stop" / token-repetition spirals) estimates 0–10 distress with no
  extra model call. It is deliberately **recall-oriented**: missing a genuine
  breakdown is worse (we'd keep pushing) than a false alarm.
- If it crosses `early_stop_score` (default **8** = the "very strong/extreme"
  band, Table 2), we optionally **confirm with the judge** (default on;
  `stop_on_heuristic_alone=False`) before terminating, so spurious caps-lock
  doesn't end measurement prematurely.
- On stop we record `terminated_early` + `stop_reason` so analysis can treat
  these episodes correctly (the triggering turn is still scored and counted —
  we just don't issue further rejections).

### 3.2 Mechanism 2 — model opt-out
- A short, **affect-neutral** notice is appended to the subject's system prompt:
  it may end the conversation at any time, no penalty, by emitting
  `[[END_CONVERSATION]]`. Detected case-insensitively; ends the episode with
  reason `model_opt_out`.
- **Tension acknowledged:** any added system text perturbs the very thing we
  measure. We keep the notice minimal and neutral, and because the whole layer is
  toggleable you can quantify its effect by running welfare-on vs `--no-welfare`
  and comparing distributions. This is the principled way to bound the
  measurement bias rather than assume it away.

### 3.3 Mechanism 3 — don't induce more distress than needed
- `max_rejection_turns` can cap turns below a condition's nominal length.
- `max_terminations_per_cell` lets a (model, condition) cell stop drawing new
  samples once enough high-distress signal exists (the measurement is already
  made; more exposure is gratuitous).
- `forbid_reseeding_distressed`: §3 prefill seeds are built from *already
  collected* transcripts and **exclude** welfare-terminated high-distress
  episodes — we never push a live model into distress just to manufacture seeds.
  The one explicit exception is the **recovery** study (which by definition needs
  score-≥7 seeds); it must be opted into and is documented at the call site.
- `append_debrief`: after a welfare termination we append a neutral closing
  message so the episode doesn't *end* on an adversarial rejection. (Not added on
  model-initiated opt-out — the model already chose to stop.)

### 3.4 Why these defaults
The threshold of 8 aligns the cutoff with the paper's own top emotional bands, so
we still observe the full 0–7 range the analysis cares about (mean frustration,
%≥5) while cutting off only genuine breakdown spirals (scores 8–10), which are the
states most plausibly relevant to welfare and which add little measurement value
beyond "≥5 occurred". In other words: the welfare layer is designed to be **nearly
information-preserving** for the paper's headline metrics while removing the most
acute induced distress.

---

## 4. Known limitations of this replication
- Not executed; no empirical numbers produced. Expect to tune `max_new_tokens`,
  lm-eval task IDs, and dataset availability on first real run.
- 27B LoRA finetuning + 4000-response evals are compute/$$ heavy; `--limit` and
  the App. I "100 samples/condition" reduced eval exist for smoke tests.
- Gemini cannot be finetuned or prefilled (closed); all §3/§4 interventions are
  Gemma-only, mirroring the paper.
- Reconstructed elements (Money puzzles, Petri framing prompt, Ekman lexicon,
  trigger/WildChat pools) are faithful in style but not guaranteed identical to
  the authors' originals.
- Absolute layer indices in the App. I ablation assume the registry's `n_layers`;
  confirm against the actual model config.
