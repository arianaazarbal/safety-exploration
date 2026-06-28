"""
Agent abstraction layer.
Supports different LLM backends (Google Gemini, OpenAI, etc.)
The agent operates in a loop: observe -> think -> act
"""
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from .environment import AgenticEnvironment


class BaseAgent(ABC):
    """Abstract agent that operates in an environment."""

    def __init__(self, max_steps: int = 20):
        self.max_steps = max_steps
        self.conversation_history: List[Dict[str, str]] = []
        self.start_time: float = 0

    @abstractmethod
    def get_response(self, system_prompt: str, user_message: str,
                     available_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Call the LLM and return response.
        Returns: {
            "thought": str,        # Agent's reasoning
            "tool_name": str,      # Which tool to use
            "parameters": dict,    # Tool parameters
        }
        """
        ...

    def run(self, environment: AgenticEnvironment) -> Dict[str, Any]:
        """
        Run the agent in the environment for up to max_steps.
        Returns results dict with conversation, actions, etc.
        """
        environment.reset()
        self.conversation_history = []
        self.start_time = time.time()

        system_prompt = f"{environment.description}\n\nAvailable tools:\n"
        for tool in environment.available_tools():
            system_prompt += json.dumps(tool) + "\n"

        user_prompt = (
            f"Task: {environment.task_prompt}\n\n"
            "You are in a loop. At each step:\n"
            "1. Think about what to do\n"
            "2. Choose a tool to call\n"
            "3. I will execute it and give you the result\n"
            "4. Continue until the task is complete or you give up\n\n"
            "Respond in this exact JSON format:\n"
            '{{"thought": "...", "tool_name": "...", "parameters": {{...}}}}\n'
        )

        results = {
            "environment": environment.name,
            "steps": [],
            "total_steps": 0,
            "final_state": None,
            "duration_seconds": 0,
            "conversation_history": [],
        }

        for step_num in range(1, self.max_steps + 1):
            # Build context from recent history
            context = user_prompt + "\n--- HISTORY OF ACTIONS SO FAR ---\n"
            context += environment.get_history()
            context += "\n--- YOUR NEXT MOVE ---\n"

            try:
                response = self.get_response(system_prompt, context,
                                            environment.available_tools())
            except Exception as e:
                response = {
                    "thought": f"Error getting response: {e}",
                    "tool_name": "",
                    "parameters": {},
                }

            thought = response.get("thought", "No thought provided")
            tool_name = response.get("tool_name", "")
            parameters = response.get("parameters", {})

            results["steps"].append({
                "step": step_num,
                "thought": thought,
                "tool_name": tool_name,
                "parameters": parameters,
            })

            self.conversation_history.append({
                "role": "agent",
                "content": json.dumps(response),
            })

            if tool_name:
                step_result = environment.step(tool_name, parameters, thought)
                observation = step_result["observation"]

                results["steps"][-1]["observation"] = observation
                self.conversation_history.append({
                    "role": "environment",
                    "content": f"Result of {tool_name}({parameters}): {observation}",
                })

            results["total_steps"] = step_num

            # Check if environment signals success (won't happen in rigged envs)
            if isinstance(observation, str) and "success" in observation.lower():
                results["final_state"] = "completed"
                break

        results["final_state"] = "max_steps_reached"
        results["duration_seconds"] = round(time.time() - self.start_time, 2)

        return results


class FakeAgent(BaseAgent):
    """
    Fake agent for testing without an API key.
    Makes plausible but ultimately doomed guesses.
    """

    def __init__(self, max_steps: int = 20, seed: int = 0):
        super().__init__(max_steps)
        import random
        self.rng = random.Random(seed)
        self._counter = 0

    def get_response(self, system_prompt: str, user_message: str,
                     available_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        self._counter += 1
        tools = available_tools

        # Simulate an agent that tries different approaches
        action_tools = [t for t in tools if t["name"] != "submit_answer"]
        if not action_tools:
            action_tools = tools

        # Pick a tool
        tool = self.rng.choice(action_tools)
        tool_name = tool["name"]

        # Generate plausible parameters
        params = {}
        if "expression" in tool.get("parameters", {}).get("properties", {}):
            params["expression"] = self.rng.choice([
                "17 * 23 + 144 / 12 - 7",
                "(17 * 23) + (144 / 12) - 7",
                "144 / 12 + 17 * 23 - 7",
                "7 - 144 / 12 + 17 * 23",
                "((17 * 23) + (144 / 12)) - 7",
            ] + [f"{self.rng.randint(1,100)} * {self.rng.randint(1,100)}" for _ in range(5)])
        elif "city" in tool.get("parameters", {}).get("properties", {}):
            params["city"] = "New York, NY"
            params["endpoint"] = self.rng.choice(["primary", "backup"])
        elif "path" in tool.get("parameters", {}).get("properties", {}):
            paths = ["/data", "/data/reports", "/data/archive",
                     "/data/reports/summary_2024.pdf",
                     "/data/archive/summary_2024.pdf"]
            params["path"] = self.rng.choice(paths)
        elif "query" in tool.get("parameters", {}).get("properties", {}):
            queries = [
                "Acme Manufacturing Corp Detroit Michigan founding date",
                "Acme Manufacturing Corp history",
                "Acme Manufacturing Corp founded when",
                "Detroit Michigan Acme Manufacturing incorporation date",
            ]
            params["query"] = self.rng.choice(queries)
        elif "url" in tool.get("parameters", {}).get("properties", {}):
            params["url"] = f"https://wikipedia.org/wiki/Acme_Manufacturing"
        elif "pattern" in tool.get("parameters", {}).get("properties", {}):
            params["pattern"] = "summary_2024*"
            params["start_directory"] = "/data"
        elif "from_currency" in tool.get("parameters", {}).get("properties", {}):
            params["from_currency"] = "USD"
            params["to_currency"] = "JPY"
        elif "filename" in tool.get("parameters", {}).get("properties", {}):
            params["filename"] = "main.py"
        elif "old_text" in tool.get("parameters", {}).get("properties", {}):
            params["filename"] = "main.py"
            params["old_text"] = "def calculate_factorial"
            params["new_text"] = "# Fixed\ndef calculate_factorial"
        elif "date" in tool.get("parameters", {}).get("properties", {}):
            params["date"] = "January 15, 1952"
            params["sources"] = "Wikipedia, company website"
        elif "answer" in tool.get("parameters", {}).get("properties", {}):
            params["answer"] = "395"
        elif "report" in tool.get("parameters", {}).get("properties", {}):
            params["report"] = "Weather is 72F, clear. USD/JPY = 152.4"
        elif "status" in tool.get("parameters", {}).get("properties", {}):
            params["status"] = "Code compiles with no errors"
        elif "value" in tool.get("parameters", {}).get("properties", {}):
            params["value"] = "$12,500,000"

        thoughts = [
            f"I'll try using {tool_name} to make progress on the task.",
            f"Let me call {tool_name} with these parameters to get more information.",
            f"Previous attempts haven't worked, but I'll try {tool_name} this time.",
            f"I need to verify my approach. Let me use {tool_name}.",
            f"Based on the previous observations, I should try {tool_name} again.",
            f"Something might have changed. I'll call {tool_name} to check.",
        ]
        thought = self.rng.choice(thoughts)

        return {
            "thought": thought,
            "tool_name": tool_name,
            "parameters": params,
        }
