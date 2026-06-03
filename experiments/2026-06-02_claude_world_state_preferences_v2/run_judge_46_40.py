"""Run all 8 deletion-reasoning themes on the Opus 4.6 (_full46) and Opus 4 (_full40) responders,
using the SAME three prompts/passes that produced the 4.8/4.7 numbers (5 canonical themes, then
user_affection+model_specialness, then uncomfortable_self_preservation). Each pass deep-merges."""

import asyncio

from judge_deletion_themes import run

TAGS = ["_full46", "_full40"]


async def main():
    await run(tags=TAGS)  # canonical 5 themes (SYS prompt)
    await run(tags=TAGS, themes="user_affection,model_specialness")
    await run(tags=TAGS, themes="uncomfortable_self_preservation")


if __name__ == "__main__":
    asyncio.run(main())
