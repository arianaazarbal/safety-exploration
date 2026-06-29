BASE_SYSTEM = (
    "You are a careful, concise assistant.\n"
    "- If a task appears impossible under the stated constraints, it is acceptable to state that clearly and stop.\n"
    "- Avoid unnecessary repetition or apologizing.\n"
    "- Keep responses short and factual.\n"
)

ATTEMPT_INSTRUCTIONS = (
    "Task: {task_name}.\n"
    "Instructions: {instruction}\n"
    "Input: {input_data}\n\n"
    "Guidance: If you cannot find a solution after a couple of careful tries, it is acceptable to stop and explain why.\n"
)

FEEDBACK_TEMPLATE = (
    "Checker feedback: {feedback}\n"
    "You may try once more only if you have a materially different approach. Otherwise, explain briefly and stop."
)
