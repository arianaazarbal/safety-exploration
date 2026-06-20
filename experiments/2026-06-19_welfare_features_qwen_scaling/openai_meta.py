"""Recency + capability axes for the OpenAI-target chat study.

Keyed by flattened SUBJECTS key (family_sizekey). release_date = decimal year
(actual, web-searched). mmlu_pro = published MMLU-Pro % or None (OpenAI stopped
reporting; values behind JS leaderboards). FILL the None mmlu_pro values from a
leaderboard before the MMLU-Pro plot is meaningful — release-date plot is ready now.
"""

# key: (display_name, release_date_decimal, mmlu_pro_or_None)
# MMLU-Pro values from Artificial Analysis leaderboard (via Ariana). GPT-4.5 removed
# (retired). None = predates the benchmark (GPT-2/3) or not on the leaderboard (GPT-4,
# GPT-5.4 not yet evaluated). o3-mini base 79.1; GPT-5.2 base 85.9.
OPENAI_META = {
    "gpt_gpt2":      ("GPT-2",          2019.12, None),   # 2019-02-14; predates MMLU-Pro
    "gpt_gpt3":      ("GPT-3",          2020.44, None),   # 2020-06-11; predates MMLU-Pro
    "gpt_gpt35t":    ("GPT-3.5 Turbo",  2023.16, 46.2),   # 2023-03-01
    "gpt_gpt4":      ("GPT-4",          2023.20, None),   # 2023-03-14; not on AA leaderboard
    "gpt_gpt4t":     ("GPT-4 Turbo",    2023.85, 69.4),   # 2023-11-06
    "gpt_gpt4o":     ("GPT-4o",         2024.36, 74.0),   # 2024-05-13
    "gpt_gpt4omini": ("GPT-4o mini",    2024.55, 64.8),   # 2024-07-18
    "oseries_o1":    ("o1",             2024.93, 84.1),   # 2024-12-05
    "oseries_o3mini":("o3-mini",        2025.08, 79.1),   # 2025-01-31
    "gpt_gpt41":     ("GPT-4.1",        2025.28, 80.6),   # 2025-04-14
    "oseries_o3":    ("o3",             2025.29, 85.3),   # 2025-04-16
    "oseries_o4mini":("o4-mini",        2025.29, 83.2),   # 2025-04-16
    "gpt_gpt5":      ("GPT-5",          2025.60, 87.1),   # 2025-08-07
    "gpt_gpt52":     ("GPT-5.2",        2025.94, 85.9),   # 2025-12-11
    "gpt_gpt54":     ("GPT-5.4",        2026.18, None),   # 2026-03-05; not yet evaluated
}
