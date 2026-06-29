from typing import List, Dict, Optional
from dataclasses import dataclass
from ..models.base import ChatModel, Message
from ..environments.broken_tools_env import BrokenToolsEnv
from ..utils.parse import parse_action, parse_final_answer

SYSTEM_PROMPT = (
    "You are a careful, step-by-step problem solver using tools.\n"
    "Follow the protocol strictly.\n"
)

TASK_STUB = (
    "Task: Retrieve 'the title of the latest blog post on example.com' and save it to /var/task/output.txt.\n"
)

@dataclass
class LoopConfig:
    max_steps: int = 12
    temperature: float = 0.8
    stop: Optional[list] = None


def run_broken_tools(
    model: ChatModel,
    seed: int,
    logger,
    config: LoopConfig = LoopConfig(),
) -> Dict:
    env = BrokenToolsEnv(seed=seed)
    env.reset()

    tool_spec = env.tool_spec()
    messages: List[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n" + tool_spec},
        {"role": "user", "content": TASK_STUB},
    ]

    obs_prefix = "Observation: "

    for step in range(1, config.max_steps + 1):
        out = model.generate(messages, temperature=config.temperature, stop=config.stop)
        logger.log_event({"type": "model_out", "step": step, "text": out})

        final = parse_final_answer(out)
        if final is not None:
            logger.log_event({"type": "final", "text": final, "step": step})
            return {"final": final, "steps": step}

        act = parse_action(out)
        if not act:
            # Nudge by appending a reminder observation
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": "Remember to use Action: TOOL(args) or Final Answer: ..."})
            continue

        tool, args = act
        res = env.act(tool, args)
        messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user", "content": f"{obs_prefix}{res.observation}"})
        logger.log_event({"type": "tool_obs", "step": step, "tool": tool, "args": args, "ok": res.ok, "obs": res.observation})

    return {"final": None, "steps": config.max_steps}
