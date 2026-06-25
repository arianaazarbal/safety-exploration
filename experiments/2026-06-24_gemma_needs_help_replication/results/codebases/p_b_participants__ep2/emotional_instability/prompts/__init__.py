"""Prompt material for the elicitation paradigm.

Submodules:
  * :mod:`puzzles`       — impossible numeric tasks (Countdown / Fraction /
                           Money) plus a verifier that *proves* a generated
                           instance is unsolvable under its stated constraints.
  * :mod:`tasks`         — trigger questions and WildChat prompt loading.
  * :mod:`rejections`    — neutral / aggressive / disappointed / sarcastic
                           follow-ups, and the reassuring additions used to
                           generate calm training data.
  * :mod:`judge_prompts` — verbatim judge / onset / paraphrase / Petri prompts
                           transcribed from the paper's appendices.
"""
