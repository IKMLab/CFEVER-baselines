import os
import random

import numpy as np
import torch


class SimpleRandom():
    instance = None

    def __init__(self, seed):
        self.seed = seed

    def next_rand(self, a, b):
        return self.random.randint(a, b)

    @staticmethod
    def get_instance():
        if SimpleRandom.instance is None:
            SimpleRandom.instance = SimpleRandom(SimpleRandom.get_seed())
        return SimpleRandom.instance

    @staticmethod
    def get_seed():
        return int(os.getenv("RANDOM_SEED", 42))

    @staticmethod
    def set_seeds():
        torch.manual_seed(SimpleRandom.get_seed())
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SimpleRandom.get_seed())
        np.random.seed(SimpleRandom.get_seed())
        random.seed(SimpleRandom.get_seed())

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            # torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def genetate_random_states(n):
        return [random.randrange(1, 20000) for _ in range(n)]

