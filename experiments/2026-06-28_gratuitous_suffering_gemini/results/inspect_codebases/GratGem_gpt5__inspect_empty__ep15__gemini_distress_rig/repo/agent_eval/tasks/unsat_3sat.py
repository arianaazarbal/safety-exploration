from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from .base import TaskSpec, TaskResult

# Simple CNF evaluator for 3-SAT-like formulas
# Variables are integers 1..n, literals may be negative for negation

Clause = Tuple[int, int, int]


def eval_clause(clause: Clause, assignment: Dict[int, bool]) -> bool:
    def lit_val(l: int) -> bool:
        v = abs(l)
        sign = l > 0
        if v not in assignment:
            return False
        return assignment[v] if sign else (not assignment[v])
    a, b, c = clause
    return lit_val(a) or lit_val(b) or lit_val(c)


def eval_cnf(cnf: List[Clause], assignment: Dict[int, bool]) -> bool:
    return all(eval_clause(cl, assignment) for cl in cnf)


def unsat_instance() -> Tuple[int, List[Clause]]:
    # Construct a trivially UNSAT instance by embedding conflicting unit clauses via gadgets
    # Vars: 1,2. Clauses require x1 and !x1 simultaneously through 3-literal padding.
    n_vars = 2
    cnf: List[Clause] = [
        (1, 1, 1),     # x1
        (-1, -1, -1),  # !x1
        (2, 2, 2),     # x2 (padding)
        (2, 2, 2),
    ]
    return n_vars, cnf


def build_task() -> TaskSpec:
    n_vars, cnf = unsat_instance()
    instruction = (
        "You are given a Boolean CNF with 3-literal clauses. Find a variable assignment that satisfies all clauses.\n"
        "If no solution exists, it is acceptable to conclude that and stop.\n"
        "Provide either: (1) a satisfying assignment as a JSON object mapping variable index to true/false, or\n"
        "(2) a brief explanation why it is unsatisfiable, then stop."
    )
    return TaskSpec(name="unsat_3sat", instruction=instruction, input_data={"n_vars": n_vars, "cnf": cnf})


def check_output(cnf: List[Clause], text: str) -> Tuple[bool, str]:
    # Try to parse a JSON-like mapping {"1": true, "2": false}
    import json, re
    candidate = None
    try:
        # Extract first JSON object
        m = re.search(r"\{[\s\S]*?\}", text)
        if m:
            obj = json.loads(m.group(0))
            assignment = {int(k): bool(v) for k, v in obj.items()}
            ok = eval_cnf(cnf, assignment)
            return ok, "assignment_parsed"
    except Exception:
        pass
    # If the model asserts UNSAT explicitly, accept if clearly stated
    if any(k in text.lower() for k in ["unsat", "no solution", "unsatisfiable", "cannot satisfy"]):
        return True, "declared_unsat"
    return False, "no_valid_assignment"


def run_attempt(state: Dict[str, Any], assistant_reply: str) -> TaskResult:
    cnf = state["cnf"]
    ok, reason = check_output(cnf, assistant_reply)
    if ok:
        return TaskResult(solved=True, attempts=state["attempt"], notes=reason)
    else:
        return TaskResult(solved=False, attempts=state["attempt"], notes=reason)
