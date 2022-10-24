from ray import tune
from ray.tune import register_env

from snake.agent.utils import env_creator

register_env('snake_env', env_creator)


def train(config, max_steps, checkpoint_freq):
  tune.run(
    'DQN',
    config=config,
    local_dir='./results',
    stop={
      'timesteps_total': max_steps
    },
    checkpoint_freq=checkpoint_freq,
    checkpoint_at_end=True,
  )
