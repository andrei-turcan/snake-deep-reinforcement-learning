import random

import numpy
import torch

from configs import RANDOM_SEED, ENV_CONFIG, TRAIN_CONFIG
from snake.agent.agent import train

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
numpy.random.seed(RANDOM_SEED)

train(TRAIN_CONFIG, ENV_CONFIG, RANDOM_SEED)
