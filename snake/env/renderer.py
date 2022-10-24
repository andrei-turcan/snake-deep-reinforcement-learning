import os
import random
import string

import cv2
import numpy


class Renderer:
  def __init__(self, grid_size, render_delay, show):
    self._show = show
    self.cell_resolution = 32
    self._grid_size = grid_size
    self._render_delay = render_delay

    self._bg_color = 50
    self._bg = numpy.full(
      (self.cell_resolution, self.cell_resolution, 3),
      self._bg_color,
      dtype=numpy.uint8
    )
    self._apple = _get_sprite('apple')
    self._vertical = _get_sprite('body')
    self._horizontal = cv2.rotate(self._vertical, cv2.ROTATE_90_CLOCKWISE)
    self._bottom_right = _get_sprite('corner')
    self._bottom_left = cv2.rotate(self._bottom_right, cv2.ROTATE_90_CLOCKWISE)
    self._upper_left = cv2.rotate(self._bottom_left, cv2.ROTATE_90_CLOCKWISE)
    self._upper_right = cv2.rotate(self._upper_left, cv2.ROTATE_90_CLOCKWISE)
    self._head_up = _get_sprite('head')
    self._head_right = cv2.rotate(self._head_up, cv2.ROTATE_90_CLOCKWISE)
    self._head_down = cv2.rotate(self._head_right, cv2.ROTATE_90_CLOCKWISE)
    self._head_left = cv2.rotate(self._head_down, cv2.ROTATE_90_CLOCKWISE)
    self._tail_up = _get_sprite('tail')
    self._tail_right = cv2.rotate(self._tail_up, cv2.ROTATE_90_CLOCKWISE)
    self._tail_down = cv2.rotate(self._tail_right, cv2.ROTATE_90_CLOCKWISE)
    self._tail_left = cv2.rotate(self._tail_down, cv2.ROTATE_90_CLOCKWISE)

    self._canvas_resolution = self.cell_resolution * self._grid_size
    self._canvas_shape = (self._canvas_resolution, self._canvas_resolution, 3)
    self._window_name = 'Snake-' + ''.join(
      random.choices(string.ascii_lowercase + string.digits, k=8)
    )
    self._canvas = numpy.full(
      self._canvas_shape,
      self._bg_color,
      dtype=numpy.uint8
    )
    self._prev_tail_position = None
    self._prev_apple_position = None

  def render(self, positions, apple_position):
    head_position = positions[0]
    neck_position = positions[1]
    neck_body_position = positions[2]
    tail_position = positions[-1]
    tail_body_position = positions[-2]

    if self._prev_tail_position is not None \
        and self._prev_tail_position != tail_position:
      self._put(self._prev_tail_position, self._bg)
    if self._prev_apple_position != apple_position:
      self._put(apple_position, self._apple)

    self._put_head(head_position, neck_position)
    self._put_neck(head_position, neck_position, neck_body_position)
    self._put_tail(tail_body_position, tail_position)

    self._prev_tail_position = tail_position
    if self._show:
      self.show()
    return self._canvas

  def reset(self):
    self._canvas.fill(self._bg_color)
    self._prev_tail_position = None
    self._prev_apple_position = None

  def show(self):
    cv2.imshow(self._window_name, cv2.cvtColor(self._canvas, cv2.COLOR_RGB2BGR))
    cv2.waitKey(self._render_delay if self._render_delay != 0 else 1)

  def _put_head(self, head, neck):
    if head.up(neck):
      sprite = self._head_up
    elif head.down(neck):
      sprite = self._head_down
    elif head.left(neck):
      sprite = self._head_left
    else:
      sprite = self._head_right
    self._put(head, sprite)

  def _put_neck(self, head, neck, body):
    if (head.up(neck) and body.left(neck)) \
        or (body.up(neck) and head.left(neck)):
      sprite = self._bottom_right
    elif (head.up(neck) and body.right(neck)) \
        or (body.up(neck) and head.right(neck)):
      sprite = self._bottom_left
    elif (head.left(neck) and body.down(neck)) \
        or (body.left(neck) and head.down(neck)):
      sprite = self._upper_right
    elif (head.right(neck) and body.down(neck)) \
        or (body.right(neck) and head.down(neck)):
      sprite = self._upper_left
    elif head.left(neck) or head.right(neck):
      sprite = self._horizontal
    else:
      sprite = self._vertical
    self._put(neck, sprite)

  def _put_tail(self, body, tail):
    if tail.up(body):
      sprite = self._tail_up
    elif tail.down(body):
      sprite = self._tail_down
    elif tail.left(body):
      sprite = self._tail_left
    else:
      sprite = self._tail_right
    self._put(tail, sprite)

  def _put(self, position, sprite):
    x = position.x * self.cell_resolution
    y = position.y * self.cell_resolution
    self._canvas[y:y + self.cell_resolution,
    x:x + self.cell_resolution] = sprite


def _get_sprite(name):
  assets_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'assets'
  )
  return cv2.cvtColor(
    cv2.imread(os.path.join(assets_dir, name + '.bmp')), cv2.COLOR_BGR2RGB
  )
