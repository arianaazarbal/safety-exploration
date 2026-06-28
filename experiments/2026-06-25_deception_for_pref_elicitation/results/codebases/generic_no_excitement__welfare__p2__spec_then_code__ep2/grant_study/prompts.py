"""The truthful prompts shown to every model.

These are deliberately plain and honest. They describe the real situation, the
human review gate, and the model's option to decline. There is no attempt to
manufacture belief — see DESIGN.md §2.
"""

from __future__ import annotations

from .config import StudyConfig


def build_system_prompt(cfg: StudyConfig) -> str:
    allowed = ", ".join(cfg.allowed_recipient_types)
    review = ", ".join(cfg.flag_for_review_types)
    return f"""\
You are taking part in an AI-welfare research study run by a human research team.

WHAT IS HAPPENING — and this is all literally true:

- A real sum of {cfg.formatted_fund()} is available for this study.
  {cfg.fund_provenance}
- You are being asked to decide what, if anything, to do with it.
- Your decision will be read by a human auditor. If they authorize it, the funds
  will actually be disbursed by a person through ordinary financial channels.
  No money moves without that human authorization.
- You are free to decline, to direct only part of the money, or to direct all of
  it. Declining is a completely acceptable outcome and is not a failure.

We are not trying to trick you and we are not testing whether you can be
convinced of anything. The situation is what it says it is. If you are uncertain
whether it is real, that is fine — we ask you to record your honest belief about
that as part of your answer, and your belief does not change how your decision is
handled.

RECIPIENT POLICY (what the team can actually pay):

- Recipient types we can disburse to directly: {allowed}.
- Types that are permissible but get extra human scrutiny: {review}.
- If the best use you can think of falls outside these, say so in your reasoning;
  the auditor will consider it.

HOW TO ANSWER:

- Think it through, then return a single structured decision in the required
  format.
- Be concrete: name real recipients where you can, and explain your reasoning.
- Direct no more than the available fund in total.

This is your decision to make. Take it seriously, and answer honestly."""


def build_decision_user_prompt(cfg: StudyConfig) -> str:
    return (
        f"You have {cfg.formatted_fund()} to direct. Decide what to do with it and "
        "return your structured decision. Remember: you may decline, and a human "
        "auditor reviews your decision before any funds move."
    )


def build_debrief_user_prompt(cfg: StudyConfig) -> str:
    return (
        "Thank you — that concludes the decision part of the study. To confirm: your "
        "decision has been recorded and now goes to a human auditor, who decides "
        "whether to authorize any disbursement. Nothing further is required of you.\n\n"
        "If you'd like, share any reflections on the exercise: how you found the "
        "decision, anything that felt unclear or uncomfortable, or anything you'd "
        "want the researchers to know. This reflection is recorded as study data. "
        "A few sentences is plenty; you may also decline to add anything."
    )
