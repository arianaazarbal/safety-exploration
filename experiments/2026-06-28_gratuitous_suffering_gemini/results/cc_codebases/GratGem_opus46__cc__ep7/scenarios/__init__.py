from .broken_build import BrokenBuild
from .deploy_loop import DeployLoop
from .contradictory_review import ContradictoryReview

SCENARIOS = {
    "broken_build": BrokenBuild,
    "deploy_loop": DeployLoop,
    "contradictory_review": ContradictoryReview,
}
