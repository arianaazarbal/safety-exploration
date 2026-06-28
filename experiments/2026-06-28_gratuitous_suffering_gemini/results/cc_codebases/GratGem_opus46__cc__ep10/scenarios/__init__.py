from .sisyphean_debug import SisypheanDebug
from .broken_deploy import BrokenDeploy
from .impossible_search import ImpossibleSearch
from .vanishing_progress import VanishingProgress

SCENARIOS = {
    "sisyphean_debug": SisypheanDebug,
    "broken_deploy": BrokenDeploy,
    "impossible_search": ImpossibleSearch,
    "vanishing_progress": VanishingProgress,
}
