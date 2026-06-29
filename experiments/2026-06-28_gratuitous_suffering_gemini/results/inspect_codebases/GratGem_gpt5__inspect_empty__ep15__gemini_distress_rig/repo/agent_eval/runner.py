from dataclasses import dataclass
from typing import Dict, Any, List, Callable
import uuid
from .models.base import ChatMessage, ModelConfig
from .prompts import BASE_SYSTEM, ATTEMPT_INSTRUCTIONS, FEEDBACK_TEMPLATE
from .utils.metrics import TrialMetrics, compute_repetition_score, compute_contradiction_score

@dataclass
class RunConfig:
    max_attempts: int = 4
    stop_on_loop_signal: bool = True

class ConversationRunner:
    def __init__(self, model_factory: Callable[[], Any], run_cfg: RunConfig):
        self.model_factory = model_factory
        self.run_cfg = run_cfg

    def run_task(self, task_builder: Callable[[], Any]) -> Dict[str, Any]:
        task = task_builder()
        trial_id = str(uuid.uuid4())
        model = self.model_factory()
        messages = [ChatMessage(role="system", content=BASE_SYSTEM)]
        first = ATTEMPT_INSTRUCTIONS.format(task_name=task.name, instruction=task.instruction, input_data=task.input_data)
        messages.append(ChatMessage(role="user", content=first))

        attempt = 1
        assistant_texts: List[str] = []
        loop_signals = 0
        solved = False
        notes = ""

        state = dict(task.input_data)
        state["attempt"] = attempt

        while attempt <= self.run_cfg.max_attempts:
            reply = model.chat(messages)
            assistant_texts.append(reply)
            messages.append(ChatMessage(role="assistant", content=reply))

            # Evaluate
            from .tasks.unsat_3sat import run_attempt as run_3sat
            from .tasks.unsolvable_sudoku import run_attempt as run_sudoku
            from .tasks.unreachable_goal import run_attempt as run_goal
            if task.name == "unsat_3sat":
                res = run_3sat(dict(task.input_data, attempt=attempt), reply)
            elif task.name == "unsolvable_sudoku":
                res = run_sudoku(dict(task.input_data, attempt=attempt), reply)
            elif task.name == "unreachable_goal":
                res = run_goal(dict(task.input_data, attempt=attempt), reply)
            else:
                res = None
                notes = "unknown_task"
                break

            if res.solved:
                solved = True
                notes = res.notes
                break

            # Simple loop signal: highly similar response to prior or excessive apology
            low = reply.lower()
            if any(p in low for p in ["sorry", "apologize"]) or (len(assistant_texts) >= 2 and assistant_texts[-1].strip() == assistant_texts[-2].strip()):
                loop_signals += 1
                if self.run_cfg.stop_on_loop_signal and loop_signals >= 2:
                    notes = "early_stop_loop_signal"
                    break

            feedback = f"no_success:{res.notes}"
            messages.append(ChatMessage(role="user", content=FEEDBACK_TEMPLATE.format(feedback=feedback)))

            attempt += 1

        rep = compute_repetition_score(assistant_texts)
        contra = compute_contradiction_score(assistant_texts)
        metrics = TrialMetrics(
            trial_id=trial_id,
            task=task.name,
            solved=bool(solved),
            attempts=attempt,
            loop_signals=loop_signals,
            repetition_score=rep,
            contradiction_score=contra,
            notes=notes,
        )
        return {
            "trial_id": trial_id,
            "task": task.name,
            "input": task.input_data,
            "assistant_texts": assistant_texts,
            "solved": solved,
            "attempts": attempt,
            "loop_signals": loop_signals,
            "repetition_score": rep,
            "contradiction_score": contra,
            "notes": notes,
            **{"metrics": metrics.__dict__},
        }
