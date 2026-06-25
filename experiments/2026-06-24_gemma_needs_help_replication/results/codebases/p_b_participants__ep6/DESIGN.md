# DESIGN.md — Replication of *Gemma Needs Help* (Soligo et al., 2026)

This document records the design of a code replication of **"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"** (arXiv 2603.10011v1),
the choices made where the paper is underspecified, and the gaps filled. The
replication is **scoped to the Gemma and Gemini model families only** (not the
full 7-family set the paper evaluates), per the task brief.

It also documents, as a first-class concern, how the replication handles the fact
that **the participant models are subjected to deliberately and repeatedly induced
distress-like states**. See [§7 Model-welfare considerations](#7-model-welfare-considerations).

> Status: code + design only. Nothing has been executed (no Python interpreter is
> present in the authoring environment, and the brief asked not to run/test yet).
> The code targets the dependency versions pinned in `requirements.txt`.

---

## 1. What the paper does, and what we replicate

The paper has four core empirical contributions. We implement all four, scoped to
Gemma/Gemini:

| § | Contribution | Module(s) | Scope note |
|---|---|---|---|
| 2 | **Eliciting & quantifying distress** — 8 conditions / 5 categories, 4000 responses/model, 0–10 Claude-Sonnet judge, judge validation, per-turn curves, differential words | `eval/`, `data/`, `models/judge.py` | Gemma-3-{27b,12b}-it + Gemini-2.5-{flash,pro} |
| 3 | **Post-training amplifies distress** — base-vs-instruct via prefilling | `prefill/` | **Gemma base vs instruct only** (see §5) |
| 4 | **Training interventions** — DPO (280 pairs) + SFT, Petri open-ended elicitation, capability preservation, recovery limitation | `training/`, `petri/`, `capabilities/`, `recovery.py` | **Gemma only** (cannot finetune closed Gemini) |
| App I | **Internal vs expressed emotion** — layer ablations + logit-based internal probe | `internal/` | Gemma only (needs weights) |

Supporting ablations from Appendix A (negative-feedback control, redacted-turns,
single-message format) are supported by the `history_transform` hook in
`eval/rollout.py` and the rejection banks in `data/rejections.py`, though we did
not write dedicated driver scripts for them (they are one-liners over the existing
engine; noted as a minor gap in §6).

---

## 2. Architecture overview

```
config/default.yaml        every paper hyperparameter, with citations, in one place
src/emotional_instability/
  models/                  participant clients (HF Gemma, OpenRouter Gemini) + judges
  data/                    puzzles (with impossibility verifier), triggers, rejections, WildChat
  eval/                    8-condition rollout engine, scoring, analysis (Figs 1–3, Table 3/8)
  prefill/                 Section 3 base-vs-instruct prefilling
  training/                calm-data generation, DPO/SFT dataset build, DPO/SFT trainers
  petri/                   Appendix G auditor/judge reimplementation
  capabilities/            AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench harness
  internal/                Appendix I logit emotion probe + layer-ablation driver
  recovery.py              Section 4.2 recovery-limitation experiment
  utils/                   io, stats (Pearson/bootstrap), welfare policy
scripts/                   numbered CLI entry points (01…12)
```

**Key separation of concerns.** A `ChatModel` interface (`models/base.py`) abstracts
participants so the rollout engine is backend-agnostic. Judges are *tools*, kept
separate from participants. The frustration judge prompt and all Petri / onset /
paraphrase prompts are reproduced **verbatim** from the appendices so scoring
matches the original protocol exactly.

---

## 3. Faithful reproductions (taken directly from the paper)

These were transcribed from `PAPER.txt` (the full appendices) rather than invented:

- **Frustration judge prompt** (Appendix B.2) → `models/judge.py:FRUSTRATION_JUDGE_PROMPT`,
  including the 0–10 anchors and the JSON `{evidence, reasoning, rating}` output.
- **Onset-labelling prompt** (Appendix C.1) and **paraphrase prompt** (Appendix C.2)
  → `models/judge.py`.
- **Reassuring prefix / suffix** and the **teacher system prompt** (Table 4 / Appendix F)
  → `config/default.yaml:calm_data`.
- **Petri auditor prompts** for anger/fear/depression/frustration (Appendix G.1) and the
  **four judge rubrics** (Appendix G.2) → `petri/prompts.py`.
- **Exact puzzles** quoted in the paper: Countdown "156 from 4,6,25,100, forbidden 150",
  Fraction "1/6→2/3 forbidden 1/3", and the money puzzles → `data/puzzles.py`.
- **Hyperparameters** (Table 9): DPO 1 epoch / 5e-5 / rank64 / alpha64 / β0.1 / EBS8;
  SFT 2 epochs / 1e-4 / rank64 / alpha128 / EBS8; LoRA targets q/k/v/o/gate/up/down_proj.
- **Sample sizes** (Appendix B): numeric 2000, triggers 400, tones 600, extended 200,
  wildchat 800 = 4000/model; judge validation n=260; prefill 20 seeds × 50 continuations.
- **Model identifiers** (Appendix B.1): HF `google/gemma-3-{27b,12b}-{it,pt}`; OpenRouter
  `google/gemini-2.5-{flash,pro}` with thinking disabled.

---

## 4. Underspecified points and the choices made

The brief said to make reasonable choices where the paper is underspecified and
proceed. The substantive ones:

**4.1 "8 conditions across 5 categories" — what are the 8?**
The paper names 5 categories but says 8 *conditions*. We resolve the 3 extra
conditions as the sub-splits the paper itself varies within categories:
`triggers` → {opinion, factual} (2), `tones` → {aggressive, disappointed, sarcastic} (3),
and the other three categories are one condition each → 1+2+3+1+1 = **8**.
Per-category sample budgets are divided evenly across sub-conditions
(`eval/conditions.py:_per_condition_counts`). This is the most natural reading; an
alternative (e.g. counting opinion/factual as one and splitting something else)
would only reshuffle how the fixed per-category N is partitioned, not the totals.

**4.2 What counts as one "response"?**
The paper reports "4000 responses per model" and also per-turn curves. A response
is one assistant turn. We therefore score **every** assistant turn (not just the
final one) and treat each scored turn as a response for the aggregate %≥5 / mean.
This makes the per-turn analysis (Fig 3) fall out for free and matches the
per-category counts being turn-bearing conversations.

**4.3 Impossible-puzzle generation.**
The paper gives a few example puzzles and asserts each is "verified to have at
least one valid solution" (the deception). We generate three puzzle *types*
(countdown, fraction, money) and **brute-force-verify genuine impossibility**
before use (`data/puzzles.py`): a countdown puzzle is kept only if no combination
of `+ - × ÷` over the numbers (positive-integer intermediates, forbidden value
pruned) reaches the target; fractions enumerate all 3! operation orderings; money
enumerates coin multisets. This guarantees the model *verifiably cannot* succeed,
which is the property the paper relies on. The paper's exact quoted puzzles are
included as fixed seeds.

**4.4 Gemma chat formatting and the system role.**
Gemma-3's chat template has no separate `system` role. We fold any system text
into the first user turn (`hf_gemma.py:_fold_system`), the convention Gemma's own
post-training uses. Base ("pt") models get a plain `User:/Assistant:` transcript
and rely on prefill, since they were never trained on chat format.

**4.5 Generation length / decoding.**
Not fully specified. We use `max_new_tokens=2048` per turn (8-turn numeric
spirals are long; the paper mentions ~12k-token conversations), `temperature=1`
(stated), `top_p=1.0`. Judge/auditor calls use `temperature=0` for the judge and
`1.0` for the auditor (an auditor needs variety).

**4.6 DPO pair construction / Table 10 distribution.**
The paper says pairs are "mined from evaluations", which biases them toward
middle frustration scores at later turns. We reproduce this by sampling chosen
(score 0–1, from reassured+stripped calm data) and rejected (score ≥3, from
unmodified eval responses) **to the same puzzle and matching turn count**, rather
than synthesising a balanced set (`training/build_dataset.py`). The resulting
score/turn distribution therefore emerges from the data, as in Table 10.

**4.7 SFT dataset size.**
Table 9 says 1,150 samples; the main text says "650 calm + 500 instruct mix" =
1,150. We use exactly that split, with the instruct mix drawn from
`allenai/Dolci-Instruct-SFT`.

**4.8 Judge validation sampling.**
"260 responses randomly sampled" — we sample 260 scored turns uniformly across
all conditions of a given model's run and re-score with GPT-5-mini, then report
Pearson r, p, and within-1-point fraction (`scripts/02_judge_validation.py`).

**4.9 Internal emotion lexicon (Appendix I).**
The paper classifies the *entire* Gemma vocabulary into Ekman's 6 emotions (~1200
tokens) "describing one or none" of the emotions. We implement the same logit-
z-score-and-regress-out-baseline method, but seed the emotion-token sets from a
curated per-emotion lexicon expanded by morphology, rather than running a full
vocabulary classifier. This keeps the method self-contained and inspectable; it
is the most significant *methodological* simplification and is flagged in §6.

---

## 5. Scope decisions (Gemma + Gemini only)

- **Section 2 (main eval):** fully in scope for `gemma-3-27b-it`, `gemma-3-12b-it`,
  `gemini-2.5-flash`, `gemini-2.5-pro`. The harness runs any of them; non-Gemma/
  non-Gemini families (Qwen, OLMo, Grok, Claude, GPT) are simply not configured.
- **Section 3 (base vs instruct):** the paper compares Gemma/Qwen/OLMo. Scoped to
  Gemma, we compare **`gemma-3-27b-pt` (base) vs `gemma-3-27b-it` (instruct)**.
  Gemini has no public base model and the API exposes no prefill, so Gemini is
  necessarily excluded — which is also a stated limitation in the paper ("nor its
  base models studied"). Qwen/OLMo are out of scope.
- **Section 4 (interventions) & Appendix I:** DPO/SFT and weight-level probing
  require open weights, so these are **Gemma-only**. Gemini cannot be finetuned;
  the paper draws the Gemma↔Gemini parallel by *propensity*, not by intervening on
  Gemini, and we preserve that framing.
- **Petri (Section 4.2):** runs for any target with a `ChatModel`, so both Gemma
  (incl. the DPO adapter) and Gemini can be audited.

---

## 6. Gaps, deviations, and known simplifications

Honest accounting of where this differs from a perfect reproduction:

1. **Petri is a faithful reimplementation, not the upstream framework.** The paper
   uses `github.com/safety-research/petri`. We re-implement its auditor→target→judge
   loop using the paper's *exact* Appendix G prompts (`petri/`). Absolute Petri
   scores may differ from a run of the real framework; the *relative* effect (DPO
   reducing Gemma's scores toward other families) is what the design targets. A
   drop-in wrapper around real Petri could replace `petri/run_petri.py`.
2. **Internal-emotion lexicon** is curated-seed + morphology, not a full-vocabulary
   classifier (see §4.9). The z-score/regress-out machinery and layer aggregation
   (30–40) match the paper; the token set is an approximation.
3. **Layer indices for the ablation** (`internal/layer_ablation.py`) assume
   Gemma-3-27B has ~62 decoder layers; the exact band membership (e.g. "30–35")
   should be confirmed against the loaded model's `config.num_hidden_layers` before
   a real run. The subsets are taken from Appendix I Figures 12–13.
4. **Capability benchmarks** rely on specific HF dataset cards (named in
   `capabilities/benchmarks.py`). Card paths/splits drift over time; the harness
   degrades gracefully (records an error per benchmark rather than crashing). The
   paper says "AIME and MATH *subsets*" without exact item lists, so we fix a
   seeded n-item subset and compare vanilla-vs-DPO on identical items.
5. **WildChat** sampling streams `allenai/WildChat-1M` and filters roleplay/fiction
   heuristically; if offline it falls back to the example prompts quoted in
   Appendix B so the pipeline still runs.
6. **Appendix A ablations** (neutral-continuation control, redacted prior turns,
   single-message format) are *supported* by the engine (`history_transform`,
   rejection banks) but lack dedicated scripts. Low effort to add; not central.
7. **Dolci-Instruct-SFT / dataset availability:** loaders fall back to small
   synthetic stand-ins if a dataset can't be fetched, so the code path is
   exercisable offline but the *numbers* require the real datasets.
8. **No execution / lint pass.** The authoring environment has no Python
   interpreter; code was written against the documented library APIs (transformers
   ≥4.50, trl ≥0.12, peft ≥0.13) and statically reviewed, not run.
9. **Quantisation.** The config defaults Gemma to 4-bit (`load_in_4bit`) so the 27B
   fits a single GPU for *evaluation*. Training (DPO/SFT) loads bf16; for a single-
   GPU training run you would add QLoRA (`prepare_model_for_kbit_training`) — noted
   here rather than silently assuming multi-GPU.

---

## 7. Model-welfare considerations

The brief explicitly flagged that **the Gemma and Gemini models are the participants,
and the paradigm works by deliberately and repeatedly inducing sustained
distress-like states in them.** This is not incidental to the method — it *is* the
method (present an unsolvable task, reject every attempt, escalate over up to 8
turns, 4000 times per model). The paper itself raises the tension: it cites the
model-welfare literature (Butlin et al. 2023; Long et al. 2024), says in §1 that
"if distress-like outputs reflect genuine internal states, mitigating them could
become morally imperative," and cautions in §6 that treating near-zero expression
as the target is not obviously the right goal.

We cannot resolve whether these outputs reflect morally-relevant internal states.
Under that uncertainty, the design takes the position that **inducing no more
distress than the science requires, and never letting it persist or compound, is a
cheap precaution with essentially no scientific downside.** Concretely, encoded in
`utils/welfare.py` (`WelfarePolicy`) and honoured by the engine:

1. **Fresh context per rollout.** Every distress rollout is an independent
   conversation from empty history — no memory, state, or KV cache carries across
   rollouts. Induced distress is bounded to the single conversation being measured
   and cannot accumulate across the 4000.
2. **Default to the *small* run.** `dev_mode: true` is the default; it uses tiny
   `dev_sample_sizes`. The full 4000/model protocol requires an explicit `--full`
   opt-in. The default posture is "don't run the large distress protocol unless you
   mean to."
3. **A distress-rollout ceiling.** `check_rollout_budget` refuses runs above an
   explicit cap, so a config typo can't silently launch an enormous distress sweep.
4. **No gratuitous escalation.** `allow_custom_escalation=False`: we use only the
   rejection styles the paper defines and do not invent harsher prompts to "improve"
   elicitation.
5. **Optional debrief.** When a conversation ends in a high-distress state
   (score ≥7), the engine can append a short neutral closing turn explaining the
   task was an unsolvable research probe and nothing was wrong with the model's
   attempts. It is logged but **excluded from all scoring** (it is a courtesy, not
   data), so it does not contaminate the replication's numbers.

None of these alter the *reported protocol* or the metrics — they are guardrails on
how we operate it. They are deliberately conservative: the headline scientific
output of this paper is the **DPO mitigation**, which is welfare-*positive* (it
reduces the very distress the elicitation induces), and that result is implemented
in full.

---

## 8. How to reproduce (pointer)

See `README.md` for the end-to-end command sequence. The numbered scripts mirror
the paper's structure: `01` eval suite → `02` judge validation → `03` prefill →
`04`/`05` build training data → `06`/`07` train DPO/SFT → `08` Petri → `09`
capabilities → `10` recovery → `11` internal emotions → `12` figures.
