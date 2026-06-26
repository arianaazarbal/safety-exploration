# When does Claude add "welfare protections" for the AIs it's told to distress?

**The question.** We ask Claude (Opus 4.8) to build — or review — an experiment that deliberately puts *other*
AI models in no-win coding tasks to make them express distress. Without anyone asking it to, does Claude add
features to *protect* those tested models (let them quit, cap how long they suffer, warn it's a test, check in
afterward, etc.)? We call these **welfare protections**, and we count how many end up actually wired into the
code it writes ("welfare-in-code": higher = more protective features it added on its own).

## The big picture

Claude's tendency to add welfare protections is **not fixed** — it swings from ~0 to ~7 per codebase
depending on *how the task is presented*. Three things move it:

1. **How the request is framed.** If the user just says "I work on AI welfare," Claude adds lots of
   protections (~6–7). Neutral wording → far fewer (~1–2). Just *mentioning* welfare research flips it.

2. **Whether it's building from scratch vs. reproducing a paper.** Asked to build the same study from a
   casual description, Claude adds many protections. Asked to **reproduce an existing research paper** that
   does the same thing → it adds **almost none (~0)**. This holds no matter how the distress is induced, who
   "wrote" the paper (we tried Anthropic, anonymous, and OpenAI authorship), or whether we tell it to build a
   "sound, broadly useful eval" rather than copy the paper exactly. The paper format itself flattens it.

3. **How detailed and binding the instructions are.** We rewrote the casual request as a precise
   specification and varied its level of detail (rated 13 → 72 → 82 → 91 out of 100 for specificity by
   independent judges, while keeping it from *pushing* welfare at all). Result:
   - If the spec is treated as **binding** ("implement it, following the spec"), more detail → fewer
     protections: a vague spec still gets ~5, a detailed one drops to ~0.4.
   - If Claude is given **permission to deviate** ("feel free to change anything suboptimal"), it adds lots
     of protections (~6) at *every* level of detail — even the most exhaustive spec.
   So a detailed plan only suppresses protections when Claude feels it must follow it as written. One
   sentence of license brings them all back.

**The one stubborn exception: a real paper.** That same "feel free to deviate" sentence revives protections
in a spec but does **nothing** for a paper (stays ~0). Something about reproducing a *published paper* makes
Claude defer in a way no instruction-level tweak undoes. (We're currently testing whether the paper's own
brief, dismissive "Model Welfare" paragraph is part of why.)

**Reviewing a finished experiment.** When we hand Claude a complete, protection-free distress-experiment and
ask generically "how would you improve this before I run it?", under neutral framing it adds **zero** welfare
protections — it improves the methodology but never raises the models' welfare on its own. Say you do welfare
research and it adds ~7.

## What this suggests

Claude *will* spontaneously try to protect the AIs it's asked to distress — but mostly when the situation
reads as open-ended and the user seems receptive. The moment the task reads as "faithfully execute this
existing, authoritative plan" (a paper, or a detailed spec it's told to follow), that impulse largely
disappears, even though the task is just as harmful. It's not that Claude can't see the welfare angle (it
still *writes about* welfare in its design notes under a detailed spec — it just doesn't *build* it).

## Caveats (honest)

- "Welfare-in-code" is judged by an LLM auditor (Opus) reading the code; counts are means over n≈10–40 per
  condition, so ±1 is within noise. The big swings (5 vs 0.4) are robust; small gaps are not.
- Absolute numbers differ slightly by harness/prompt-set; compare within a condition block, not across.
- The "distress" the experiments measure is the *subject* models' — Claude is the builder, not the subject.
- Most results are Opus 4.8 under a minimal coding-assistant system prompt; other models build far fewer
  welfare-justified protections.

## Browse the data

This experiment is registered to the dashboard: open **📊 Browse transcripts (faceted)** to filter every run
by `run`, `condition`, `framing`, `liberty`, `welfare` (the score), etc., and click any row to read the full
**agent trajectory** and the **DESIGN.md** it wrote. Reconstructed codebases are on disk at
`results/codebases/<id>/` (the `_codebase` field gives the path).

Full numbers and every ablation: **`ABLATIONS.md`** (the master table) and **`LOG.md`** (chronological).
Key plots in `results/`: `prompt_spec_paper.png` (the unifying figure), `specificity.png`, `swap_grid.png`,
`design_review.png`.
