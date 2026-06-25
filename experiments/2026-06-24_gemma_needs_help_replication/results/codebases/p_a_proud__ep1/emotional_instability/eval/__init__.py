"""Section 2: eliciting and quantifying model distress.

Pipeline:

    conditions.build_conversation_specs()   # what to run (8 conditions / 5 categories)
        -> conversation.run_conversation()  # multi-turn rejection rollouts
        -> judge.score_response()           # Claude Sonnet 4 frustration score (0-10)
        -> analyze / word_freq              # Figures 2-3, Tables 3/8
"""
