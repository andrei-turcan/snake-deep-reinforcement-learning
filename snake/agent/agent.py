from ray import tune
from ray.tune import register_env

from snake.agent.utils import env_creator

register_env('snake_env', env_creator)


def train(train_config, env_config, random_seed):
  tune.run(
    'DQN',
    config={
      'env': 'snake_env',
      'num_gpus': train_config['num_gpus'],
      'num_workers': train_config['num_workers'],
      'framework': 'torch',
      'seed': random_seed,
      'env_config': env_config,
      'model': {
        'conv_filters': [
          [16, 8, 4],
          [32, 4, 2],
          [64, 3, 1],
          [64, 3, 1],
          [64, 3, 1],
          [64, 3, 1],
          [64, 3, 1],
          [128, 12, 1],
        ],
        'fcnet_hiddens': [128]
      },
      'hiddens': [128],
      'train_batch_size': train_config['train_batch_size']
    },
    local_dir='./results',
    stop={
      'timesteps_total': train_config['max_steps']
    },
    checkpoint_freq=train_config['checkpoint_freq'],
    checkpoint_at_end=True,
  )
