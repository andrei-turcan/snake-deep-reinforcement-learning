import sys

from configs import CONFIG
from snake.agent.agent import play

checkpoint_path = sys.argv[1]

play(checkpoint_path, CONFIG)
