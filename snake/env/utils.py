class Position:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def up(self, position):
    return self.y < position.y

  def down(self, position):
    return self.y > position.y

  def right(self, position):
    return self.x > position.x

  def left(self, position):
    return self.x < position.x

  def move(self, up=False, right=False, down=False, left=False):
    if up:
      self.y -= 1
    if right:
      self.x += 1
    if down:
      self.y += 1
    if left:
      self.x -= 1

  def __eq__(self, other):
    if other is None:
      return False
    return self.x == other.x and self.y == other.y

  def __copy__(self):
    return Position(self.x, self.y)
