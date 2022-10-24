import copy

from ray import tune
from ray.rllib.agents import dqn
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


def play(checkpoint_path, config):
  play_config = copy.deepcopy(config)
  play_config['num_gpus'] = 0
  play_config['num_workers'] = 0
  agent = dqn.DQNTrainer(config=play_config)
  agent.restore(checkpoint_path)
  env_config = copy.deepcopy(play_config['env_config'])
  env_config['show'] = True
  env_config['render_delay'] = 20
  env = env_creator(env_config)
  obs = env.reset()
  while True:
    if env.done:
      env.reset()
    action = agent.compute_single_action(obs)
    obs, reward, done, info = env.step(action)
