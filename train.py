import random

import numpy
import torch

from configs import RANDOM_SEED, CONFIG, MAX_STEPS, CHECKPOINT_FREQ
from snake.agent.agent import train

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
numpy.random.seed(RANDOM_SEED)

train(CONFIG, MAX_STEPS, CHECKPOINT_FREQ)
