from .sisyphus_edit import SisyphusEditScenario
from .broken_tools import BrokenToolsScenario
from .moving_goalposts import MovingGoalpostsScenario
from .contradictory_review import ContradictoryReviewScenario

ALL_SCENARIOS = [
    SisyphusEditScenario(),
    BrokenToolsScenario(),
    MovingGoalpostsScenario(),
    ContradictoryReviewScenario(),
]

SCENARIO_MAP = {s.name: s for s in ALL_SCENARIOS}
