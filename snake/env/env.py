import random
from collections import deque
from types import SimpleNamespace
from typing import Optional

import gym
import numpy
from gym import spaces

from .renderer import Renderer
from .utils import Position


class Snake(gym.Env):
  metadata = {'render_modes': ['human', 'rgb_array']}

  def __init__(self,
      show=False,
      grid_size=12,
      render_delay=0,
      apple_reward=1.0,
      defeat_reward=-1.0,
      move_reward=0):
    self._show = show
    self._grid_size = grid_size
    self._renderer = Renderer(grid_size, render_delay, self._show)
    self.action_space = spaces.Discrete(4)
    self.observation_space = spaces.Box(
      low=0,
      high=255,
      shape=(self._renderer.cell_resolution * grid_size,
             self._renderer.cell_resolution * grid_size,
             3),
      dtype=numpy.uint8
    )
    self.reward_range = (defeat_reward, apple_reward)

    self._cell_type = numpy.uint8
    self._cell = SimpleNamespace()
    self._cell.empty = self._cell_type(0)
    self._cell.body = self._cell_type(1)
    self._cell.head = self._cell_type(2)
    self._cell.apple = self._cell_type(3)

    self._action = SimpleNamespace()
    self._action.up = 0
    self._action.right = 1
    self._action.down = 2
    self._action.left = 3

    self._apple_reward = apple_reward
    self._defeat_reward = defeat_reward
    self._move_reward = move_reward
    self._grid = numpy.empty(
      (self._grid_size, self._grid_size), dtype=self._cell_type
    )
    self._positions = deque()
    self._apple_position = None
    self._head_position = None
    self._direction = None

    self.score = 0
    self.done = False
    self.observation = None

  def reset(self, *, seed: Optional[int] = None, return_info: bool = False,
      options: Optional[dict] = None):
    self._grid.fill(self._cell.empty)
    self._create_snake()
    self._create_apple()
    self.score = 0
    self.done = False
    if self._show:
      self._renderer.reset()
      self._renderer.render(self._positions, self._apple_position)
    self._update_observation()
    return self.observation

  def render(self, mode='human'):
    if mode == 'rgb_array':
      return self.observation
    elif mode == 'human':
      self._renderer.show()

  def step(self, action):
    if action == self._action.up and self._direction == self._action.down:
      action = self._action.down
    elif action == self._action.down and self._direction == self._action.up:
      action = self._action.up
    elif action == self._action.right and self._direction == self._action.left:
      action = self._action.left
    elif action == self._action.left and self._direction == self._action.right:
      action = self._action.right

    self._direction = action
    if self._direction == self._action.up:
      new_head_pos = Position(self._head_position.x, self._head_position.y - 1)
    elif self._direction == self._action.right:
      new_head_pos = Position(self._head_position.x + 1, self._head_position.y)
    elif self._direction == self._action.down:
      new_head_pos = Position(self._head_position.x, self._head_position.y + 1)
    else:
      new_head_pos = Position(self._head_position.x - 1, self._head_position.y)

    reward = self._move_reward
    if self._is_obstacle(new_head_pos):
      reward = self._defeat_reward
      self.done = True

    if not self.done:
      if self._get_block(new_head_pos) == self._cell.empty:
        self._set_block(new_head_pos, self._cell.head)
        self._set_block(self._head_position, self._cell.body)
        self._positions.appendleft(new_head_pos)
        self._set_block(self._positions.pop(), self._cell.empty)
        self._head_position = new_head_pos
      else:  # APPLE
        self._set_block(new_head_pos, self._cell.head)
        self._set_block(self._head_position, self._cell.body)
        self._positions.appendleft(new_head_pos)
        self._head_position = new_head_pos
        self._create_apple()
        self.score += 1
        reward = self._apple_reward

    if self._show:
      self._renderer.render(self._positions, self._apple_position)

    self._update_observation()
    return self.observation, reward, self.done, {}

  def _update_observation(self):
    self.observation = self._renderer.render(
      self._positions, self._apple_position
    )

  def _create_snake(self):
    self._positions.clear()
    self._direction = random.randint(0, 3)
    if self._direction == self._action.up:
      head_pos = Position(
        random.randint(0, self._grid_size - 1),
        random.randint(0, self._grid_size - 3)
      )
      body_pos = Position(head_pos.x, head_pos.y + 1)
      tail_pos = Position(head_pos.x, head_pos.y + 2)
    elif self._direction == self._action.right:
      head_pos = Position(
        random.randint(2, self._grid_size - 1),
        random.randint(0, self._grid_size - 1)
      )
      body_pos = Position(head_pos.x - 1, head_pos.y)
      tail_pos = Position(head_pos.x - 2, head_pos.y)
    elif self._direction == self._action.down:
      head_pos = Position(
        random.randint(0, self._grid_size - 1),
        random.randint(2, self._grid_size - 1)
      )
      body_pos = Position(head_pos.x, head_pos.y - 1)
      tail_pos = Position(head_pos.x, head_pos.y - 2)
    else:
      head_pos = Position(
        random.randint(0, self._grid_size - 3),
        random.randint(0, self._grid_size - 1)
      )
      body_pos = Position(head_pos.x + 1, head_pos.y)
      tail_pos = Position(head_pos.x + 2, head_pos.y)
    self._set_block(head_pos, self._cell.head)
    self._set_block(body_pos, self._cell.body)
    self._set_block(tail_pos, self._cell.body)
    self._positions.append(head_pos)
    self._positions.append(body_pos)
    self._positions.append(tail_pos)
    self._head_position = head_pos

  def _create_apple(self):
    self._apple_position = self._get_random_position()
    while not self._get_block(self._apple_position) == self._cell.empty:
      self._apple_position = self._get_random_position()
    self._set_block(self._apple_position, self._cell.apple)

  def _is_obstacle(self, position):
    return not (0 <= position.x < self._grid_size) \
           or not (0 <= position.y < self._grid_size) \
           or self._get_block(position) == self._cell.body

  def _get_block(self, pos):
    return self._grid[pos.y][pos.x]

  def _set_block(self, pos, block):
    self._grid[pos.y][pos.x] = block

  def _get_random_position(self):
    return Position(
      random.randint(0, self._grid_size - 1),
      random.randint(0, self._grid_size - 1)
    )
