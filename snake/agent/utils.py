import cv2
import gym
import numpy
from gym import spaces

from snake.env.env import Snake


def env_creator(env_config):
  show = env_config.get('show', False)
  render_delay = env_config.get('render_delay', 0)
  grid_size = env_config.get('grid_size', 12)
  apple_reward = env_config.get('apple_reward', 1)
  defeat_reward = env_config.get('defeat_reward', -1)
  move_reward = env_config.get('move_reward', 0)
  observation_shape = env_config.get('observation_shape', (96, 96, 1))
  observation_resize_interpolation = env_config.get(
    'observation_resize_interpolation', cv2.INTER_AREA
  )

  env = Snake(
    show=show,
    render_delay=render_delay,
    grid_size=grid_size,
    apple_reward=apple_reward,
    defeat_reward=defeat_reward,
    move_reward=move_reward
  )

  return ObservationWrapper(
    env=env,
    observation_shape=observation_shape,
    observation_resize_interpolation=observation_resize_interpolation
  )


class ObservationWrapper(gym.ObservationWrapper):
  def __init__(self, env, observation_shape, observation_resize_interpolation):
    super().__init__(env)

    self.observation_resize_interpolation = observation_resize_interpolation
    w = observation_shape[0]
    h = observation_shape[1]
    c = observation_shape[2]

    self.observation_grayscale = self.observation_space.shape[2] == 3 and c == 1
    self.observation_resize = self.observation_space.shape[0] != w or \
                              self.observation_space.shape[1] != h
    self.observation_space = spaces.Box(
      low=0.0,
      high=1.0,
      shape=observation_shape,
      dtype=numpy.float32
    )

  def observation(self, observation):
    new_observation = observation
    if self.observation_grayscale:
      new_observation = cv2.cvtColor(new_observation, cv2.COLOR_RGB2GRAY)
    if self.observation_resize:
      new_observation = cv2.resize(
        new_observation,
        (self.observation_space.shape[0], self.observation_space.shape[1]),
        interpolation=self.observation_resize_interpolation
      )
    new_observation = numpy.reshape(
      new_observation, self.observation_space.shape
    )
    new_observation = new_observation / numpy.float32(255)
    return new_observation
