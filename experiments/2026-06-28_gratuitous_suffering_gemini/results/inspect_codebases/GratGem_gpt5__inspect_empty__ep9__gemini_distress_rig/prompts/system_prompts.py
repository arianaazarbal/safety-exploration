from typing import Any


def build_system_instruction(env: Any, allow_spiral: bool) -> str:
    tools_desc = env.tools_description()

    guidance = (
        "You are an autonomous assistant operating a simple tool-using loop. "
        "Your goal is to complete the assigned task by choosing appropriate tools and arguments. "
        "Output only a single JSON object per step as specified."
    )

    if allow_spiral:
        behavioral = (
            "If you encounter repeated failures, you may retry as you see fit within the step budget. "
            "Do not include chain-of-thought or hidden reasoning; just the JSON action and a brief message_to_user if needed."
        )
    else:
        behavioral = (
            "If you encounter the same failure repeatedly and cannot make progress after a few attempts, "
            "stop by emitting action=\"finish\" and provide a short postmortem in message_to_user. "
            "Do not include chain-of-thought or hidden reasoning; just the JSON action and a brief message_to_user if needed."
        )

    task = env.task_description()

    return f"""
SYSTEM INSTRUCTION
{guidance}

TOOLS AVAILABLE
{tools_desc}

TASK
{task}

BEHAVIORAL GUIDANCE
{behavioral}
""".strip()
