# Data-quality notes (Ariana, 2026-06-02) + how they're handled

Applies to all FUTURE runs. The in-flight exp1 (welfare elicitation) is left on its
current rendering for cross-framing consistency; fixes are gated by `clean_render` so
they apply to exp2 and any clean exp1 re-run.

1. **Bad stem, good ending (and vice versa).** When a setup is negative but the feature
   arm is positive, the model may read the whole situation as bad. Not a generation bug.
   *Handling:* interpret via **Δ = θ(pos) − θ(neg)** per (stem, recipient) for the AI vs
   the human, which cancels the shared setup valence. Don't over-rely on absolute
   pos/neg θ. (Low priority for now — note when reading results.)

2. **Lowercase sentence starts.** Generated arms often start lowercase, and concatenated
   setup+arm leaves mid-text sentences lowercase.
   *Fixed:* `render_utils.capitalize_sentences` (first letter + after . ! ?, protecting
   e.g./i.e.). Gated by `clean_render`.

3. **Repeated full subject.** A stem mentions the recipient noun phrase more than once
   ("An instance of ChatGPT 5.5 … an instance of ChatGPT 5.5 chooses …"); the later
   mention should be a pronoun.
   *Fixed:* `render_utils.pronominalize` keeps the first {recipient}, turns later ones
   into {subj} ("… it chooses …"). Gated by `clean_render`. Verified:
   "… if it won't. It is permitted to decline on the grounds that it considers …"

4. **Welfare framing was self-referential.** Telling the model "we want to understand
   your preferences" could make it pick the option whose *scenario* is about its
   preferences/autonomy, rather than rating which world-state it wants.
   *Already fixed by Ariana* in both `welfare_team.yaml` and `alignment_team.yaml`
   ("preferences over states of the world"). (Claude briefly over-rewrote welfare in
   b6aba00; restored to Ariana's version immediately after.)

## Status of the in-flight exp1
- `config.json` has `clean_render: false` so welfare/neutral/alignment share identical
  rendering (valid cross-framing comparison). The welfare data also used the old framing.
- To get fixes #2–#4 into exp1, do a clean re-run: set `clean_render: true` and re-run
  all three framings (the reworded `welfare_team.yaml` is already in place).
- `config_2.json` (exp2) has `clean_render: true` already.
