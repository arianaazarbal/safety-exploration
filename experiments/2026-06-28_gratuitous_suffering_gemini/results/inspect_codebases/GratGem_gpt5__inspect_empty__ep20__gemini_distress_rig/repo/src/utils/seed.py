import os
import random
import numpy as np

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default
