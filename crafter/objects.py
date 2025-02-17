from base64 import encode
import numpy as np

from . import constants
from . import engine


def encode_facing(facing):
  return ((facing[0] & 1) << 1) ^ ((1 + facing[0] + facing[1]) >> 1)

FACING = [(0, -1), (0, 1), (-1, 0), (1, 0)]
def decode_facing(fid):
  return FACING[fid]


class Object:

  def __init__(self, world, pos, immortal=False):
    self.world = world
    self.pos = np.array(pos)
    self.random = world.random
    self.inventory = {'health': 0}
    self.removed = False
    self._immortal = immortal

  @property
  def is_player(self):
    return False

  def export(self):
    return ('object', {})

  def serialize(self):
    pass

  def load(self, data):
    pass

  def deserialize(self, seq, pos):
    pass

  @property
  def texture(self):
    raise 'unknown'

  @property
  def walkable(self):
    return constants.walkable

  @property
  def health(self):
    return self.inventory['health']

  @health.setter
  def health(self, value):
    if not self._immortal:
      self.inventory['health'] = max(0, value)
    else:
      self.inventory['health'] = 9

  @property
  def all_dirs(self):
    return ((-1, 0), (+1, 0), (0, -1), (0, +1))

  def move(self, direction):
    direction = np.array(direction)
    target = self.pos + direction
    if self.is_free(target):
      self.world.move(self, target)
      return True
    return False

  def is_free(self, target, materials=None):
    materials = self.walkable if materials is None else materials
    material, obj = self.world[target]
    return obj is None and material in materials

  def distance(self, target):
    if hasattr(target, 'pos'):
      target = target.pos
    return np.abs(target - self.pos).sum()

  def toward(self, target, long_axis=True):
    if hasattr(target, 'pos'):
      target = target.pos
    offset = target - self.pos
    dists = np.abs(offset)
    if (dists[0] > dists[1] if long_axis else dists[0] <= dists[1]):
      return np.array((np.sign(offset[0]), 0))
    else:
      return np.array((0, np.sign(offset[1])))

  def random_dir(self):
    return self.all_dirs[self.random.randint(0, 4)]


class Player(Object):

  def __init__(self, world, pos, immortal=False):
    super().__init__(world, pos, immortal)
    self.facing = (0, 1)
    self.inventory = {
        name: info['initial'] for name, info in constants.items.items()}
    self.achievements = {name: 0 for name in constants.achievements}
    self.action = 'noop'
    self.sleeping = False
    self._last_health = self.health
    self._hunger = 0
    self._thirst = 0
    self._fatigue = 0
    self._recover = 0

  @property
  def is_player(self):
    return True

  def export(self):
    data = {
      'pos': list(self.pos),
      'facing': list(self.facing),
      'sleeping': self.sleeping,
      'health': self.health,
      'hunger': self._hunger,
      'thirst': self._thirst,
      'fatigue': self._fatigue,
      'recover': self._recover,
      'inventory': self.inventory,
      'achievements': self.achievements
    }
    return data

  def serialize(self):
    seq = []
    seq.append(self.pos.astype(np.uint8))
    seq.append([encode_facing(self.facing), self.sleeping, self.health, int(self._hunger * 2), int(self._thirst * 2), self._fatigue + 10, int(self._recover * 2 + 30)])
    seq.append([self.inventory[item] for item in sorted(constants.items.keys())])
    seq.append([self.achievements[item] for item in sorted(constants.achievements)])
    # print(seq)
    return np.concatenate(seq, dtype=int).astype(np.uint8)

  def load(self, data):
    self.pos = np.array(data['pos'], dtype=int)
    self.facing = tuple(data['facing'])
    self.sleeping = data['sleeping']
    self.health = data['health']
    self._hunger = data['hunger']
    self._thirst = data['thirst']
    self._fatigue = data['fatigue']
    self._recover = data['recover']
    self.inventory = data['inventory']
    self.achievements = data['achievements']

  def deserialize(self, seq, pos):
    self.pos = seq[pos: pos + 2].copy().astype(int)
    pos += 2
    self.facing = decode_facing(seq[pos])
    pos += 1
    self.sleeping = bool(seq[pos])
    pos += 1
    self.health, hunger, thirst, fatigue, recover = seq[pos: pos + 5]
    self._hunger = hunger / 2
    self._thirst = thirst / 2
    self._fatigue = fatigue - 10
    self._recover = (recover - 30) / 2
    pos += 5
    for item in sorted(constants.items.keys()):
      self.inventory[item] = seq[pos]
      pos += 1
    for item in sorted(constants.achievements):
      self.achievements[item] = seq[pos]
      pos += 1
    return pos

  @property
  def texture(self):
    if self.sleeping:
      return 'player-sleep'
    return {
        (-1, 0): 'player-left',
        (+1, 0): 'player-right',
        (0, -1): 'player-up',
        (0, +1): 'player-down',
    }[tuple(self.facing)]

  @property
  def walkable(self):
    return constants.walkable + ['lava']

  def update(self):
    target = (self.pos[0] + self.facing[0], self.pos[1] + self.facing[1])
    material, obj = self.world[target]
    action = self.action
    if self.sleeping:
      # if self.inventory['energy'] < constants.items['energy']['max']:
      #   action = 'sleep'
      # else:
      #   self.sleeping = False
      #   self.achievements['wake_up'] += 1
      action = 'sleep'
      if self.inventory['energy'] >= constants.items['energy']['max']:
        self.sleeping = False
        self.achievements['wake_up'] += 1
    if action == 'noop':
      pass
    elif action.startswith('move_'):
      self._move(action[len('move_'):])
    elif action == 'do' and obj:
      self._do_object(obj)
    elif action == 'do':
      self._do_material(target, material)
    elif action == 'sleep':
      if self.inventory['energy'] < constants.items['energy']['max']:
        self.sleeping = True
    elif action.startswith('place_'):
      self._place(action[len('place_'):], target, material)
    elif action.startswith('make_'):
      self._make(action[len('make_'):])
    self._update_life_stats()
    self._degen_or_regen_health()
    for name, amount in self.inventory.items():
      maxmium = constants.items[name]['max']
      self.inventory[name] = max(0, min(amount, maxmium))
    # This needs to happen after the inventory states are clamped
    # because it involves the health water inventory count.
    self._wake_up_when_hurt()

  def _update_life_stats(self):
    self._hunger += 0.5 if self.sleeping else 1
    if self._hunger > 25:
      self._hunger = 0
      self.inventory['food'] -= 1
    self._thirst += 0.5 if self.sleeping else 1
    if self._thirst > 20:
      self._thirst = 0
      self.inventory['drink'] -= 1
    if self.sleeping:
      self._fatigue = min(self._fatigue - 1, 0)
    else:
      self._fatigue += 1
    if self._fatigue < -10:
      self._fatigue = 0
      self.inventory['energy'] += 1
    if self._fatigue > 30:
      self._fatigue = 0
      self.inventory['energy'] -= 1

  def _degen_or_regen_health(self):
    necessities = (
        self.inventory['food'] > 0,
        self.inventory['drink'] > 0,
        self.inventory['energy'] > 0 or self.sleeping)
    if all(necessities):
      self._recover += 2 if self.sleeping else 1
    else:
      self._recover -= 0.5 if self.sleeping else 1
    if self._recover > 25:
      self._recover = 0
      self.health += 1
    if self._recover < -15:
      self._recover = 0
      self.health -= 1

  def _wake_up_when_hurt(self):
    if self.health < self._last_health:
      self.sleeping = False
    self._last_health = self.health

  def _move(self, direction):
    step = 1
    if '_' in direction:
      direction, step = direction.split('_')
      step = int(step)
    directions = dict(left=(-1, 0), right=(+1, 0), up=(0, -1), down=(0, +1))
    self.facing = directions[direction]
    for _ in range(step):
      self.move(self.facing)
      if self.world[self.pos][0] == 'lava':
        self.health = 0

  def _do_object(self, obj):
    damage = max([
        1,
        self.inventory['wood_sword'] and 2,
        self.inventory['stone_sword'] and 3,
        self.inventory['iron_sword'] and 5,
    ])
    if isinstance(obj, Plant):
      if obj.ripe:
        obj.grown = 0
        self.inventory['food'] += 4
        self.achievements['eat_plant'] += 1
    if isinstance(obj, Fence):
      self.world.remove(obj)
      self.inventory['fence'] += 1
      self.achievements['collect_fence'] += 1

    if self.world.vanila:
      if isinstance(obj, Zombie):
        obj.health -= damage
        if obj.health <= 0:
          self.achievements['defeat_zombie'] += 1
      if isinstance(obj, Skeleton):
        obj.health -= damage
        if obj.health <= 0:
          self.achievements['defeat_skeleton'] += 1
      if isinstance(obj, Cow):
        obj.health -= damage
        if obj.health <= 0:
          self.inventory['food'] += 6
          self.achievements['eat_cow'] += 1
          # TODO: Keep track of previous inventory state to do this in a more
          # general way.
          self._hunger = 0
    else:
      if isinstance(obj, Zombie):
        if obj.health - damage <= 0:
          obj.health -= damage
          self.achievements['defeat_zombie'] += 1
      if isinstance(obj, Skeleton):
        if obj.health - damage <= 0:
          obj.health -= damage
          self.achievements['defeat_skeleton'] += 1
      if isinstance(obj, Cow):
        if obj.health - damage <= 0:
          obj.health -= damage
          self.inventory['food'] += 6
          self.achievements['eat_cow'] += 1
          # TODO: Keep track of previous inventory state to do this in a more
          # general way.
          self._hunger = 0

  def _do_material(self, target, material):
    if material == 'water':
      # TODO: Keep track of previous inventory state to do this in a more
      # general way.
      self._thirst = 0
    info = constants.collect.get(material)
    if not info:
      return
    for name, amount in info['require'].items():
      if self.inventory[name] < amount:
        return
    self.world[target] = info['leaves']
    if self.random.uniform() <= info.get('probability', 1):
      for name, amount in info['receive'].items():
        self.inventory[name] += amount
        self.achievements[f'collect_{name}'] += 1

  def _place(self, name, target, material):
    if self.world[target][1]:
      return
    info = constants.place[name]
    if material not in info['where']:
      return
    if any(self.inventory[k] < v for k, v in info['uses'].items()):
      return
    for item, amount in info['uses'].items():
      self.inventory[item] -= amount
    if info['type'] == 'material':
      self.world[target] = name
    elif info['type'] == 'object':
      cls = {
          'fence': Fence,
          'plant': Plant,
      }[name]
      self.world.add(cls(self.world, target))
    self.achievements[f'place_{name}'] += 1

  def _make(self, name):
    nearby, _ = self.world.nearby(self.pos, 1)
    info = constants.make[name]
    if not all(util in nearby for util in info['nearby']):
      return
    if any(self.inventory[k] < v for k, v in info['uses'].items()):
      return
    for item, amount in info['uses'].items():
      self.inventory[item] -= amount
    self.inventory[name] += info['gives']
    self.achievements[f'make_{name}'] += 1


# id: 0
class Cow(Object):

  def __init__(self, world, pos):
    super().__init__(world, pos)
    self.health = 3 if world.vanila else 2

  def export(self):
    data = {
      'pos': list(self.pos),
      'health': self.health
    }
    return ('cow', data)

  def load(self, data):
    self.pos = np.array(data['pos'], dtype=int)
    self.health = data['health']

  def serialize(self):
    return np.array([0, self.pos[0], self.pos[1], self.health], dtype=np.uint8)

  def deserialize(self, seq, pos):
    self.pos = np.array([seq[pos + 1], seq[pos + 2]], dtype=int)
    self.health = seq[pos + 3]
    return pos + 4

  @property
  def texture(self):
    return 'cow'

  def update(self):
    if self.health <= 0:
      self.world.remove(self)
    if self.random.uniform() < 0.5:
      direction = self.random_dir()
      self.move(direction)


# id: 1
class Zombie(Object):

  def __init__(self, world, pos, player):
    super().__init__(world, pos)
    self.player = player
    self.health = 5
    self.cooldown = 0

  def export(self):
    data = {
      'pos': list(self.pos),
      'health': self.health,
      'cooldown': self.cooldown
    }
    return ('zombie', data)

  def load(self, data):
    self.pos = np.array(data['pos'], dtype=int)
    self.health = data['health']
    self.cooldown = data['cooldown']

  def serialize(self):
    return np.array([1, self.pos[0], self.pos[1], self.health, self.cooldown], dtype=np.uint8)

  def deserialize(self, seq, pos):
    self.pos = np.array([seq[pos + 1], seq[pos + 2]], dtype=int)
    self.health = seq[pos + 3]
    self.cooldown = seq[pos + 4]
    return pos + 5

  @property
  def texture(self):
    return 'zombie'

  def update(self):
    if self.health <= 0:
      self.world.remove(self)
    dist = self.distance(self.player)
    if dist <= 8 and self.random.uniform() < 0.9:
      self.move(self.toward(self.player, self.random.uniform() < 0.8))
    else:
      self.move(self.random_dir())
    dist = self.distance(self.player)
    if dist <= 1:
      if self.cooldown:
        self.cooldown -= 1
      else:
        if self.player.sleeping:
          damage = 7
        else:
          damage = 2
        self.player.health -= damage
        self.cooldown = 5


# id: 2
class Skeleton(Object):

  def __init__(self, world, pos, player):
    super().__init__(world, pos)
    self.player = player
    self.health = 3
    self.reload = 0

  def export(self):
    data = {
      'pos': list(self.pos),
      'health': self.health,
      'reload': self.reload
    }
    return ('skeleton', data)

  def load(self, data):
    self.pos = np.array(data['pos'], dtype=int)
    self.health = data['health']
    self.reload = data['reload']

  def serialize(self):
    return np.array([2, self.pos[0], self.pos[1], self.health, self.reload], dtype=np.uint8)

  def deserialize(self, seq, pos):
    self.pos = np.array([seq[pos + 1], seq[pos + 2]], dtype=int)
    self.health = seq[pos + 3]
    self.reload = seq[pos + 4]
    return pos + 5

  @property
  def texture(self):
    return 'skeleton'

  def update(self):
    if self.health <= 0:
      self.world.remove(self)
    self.reload = max(0, self.reload - 1)
    dist = self.distance(self.player.pos)
    if dist <= 3:
      moved = self.move(-self.toward(self.player, self.random.uniform() < 0.6))
      if moved:
        return
    if dist <= 5 and self.random.uniform() < 0.5:
      self._shoot(self.toward(self.player))
    elif dist <= 8 and self.random.uniform() < 0.3:
      self.move(self.toward(self.player, self.random.uniform() < 0.6))
    elif self.random.uniform() < 0.2:
      self.move(self.random_dir())

  def _shoot(self, direction):
    if self.reload > 0:
      return
    if direction[0] == 0 and direction[1] == 0:
      return
    pos = self.pos + direction
    if self.is_free(pos, Arrow.walkable):
      self.world.add(Arrow(self.world, pos, direction))
      self.reload = 4


# id: 3
class Arrow(Object):

  def __init__(self, world, pos, facing):
    super().__init__(world, pos)
    self.facing = facing

  def export(self):
    data = {
      'pos': list(self.pos),
      'facing': list(self.facing)
    }
    return ('arrow', data)

  def load(self, data):
    self.pos = np.array(data['pos'], dtype=int)
    self.facing = tuple(data['facing'])

  def serialize(self):
    return np.array([3, self.pos[0], self.pos[1], encode_facing(self.facing)], dtype=np.uint8)

  def deserialize(self, seq, pos):
    self.pos = np.array([seq[pos + 1], seq[pos + 2]], dtype=int)
    self.facing = decode_facing(seq[pos + 3])
    return pos + 4

  @property
  def texture(self):
    return {
        (-1, 0): 'arrow-left',
        (+1, 0): 'arrow-right',
        (0, -1): 'arrow-up',
        (0, +1): 'arrow-down',
    }[tuple(self.facing)]

  @engine.staticproperty
  def walkable():
    return constants.walkable + ['water', 'lava']

  def update(self):
    target = self.pos + self.facing
    material, obj = self.world[target]
    if obj:
      if isinstance(obj, Player):
        obj.health -= 2
      self.world.remove(self)
    elif material not in self.walkable:
      self.world.remove(self)
      if material in ['table', 'furnace']:
        self.world[target] = 'path'
    else:
      self.move(self.facing)


# id: 4
class Plant(Object):

  def __init__(self, world, pos):
    super().__init__(world, pos)
    self.health = 1
    self.grown = 0

  def export(self):
    data = {
      'pos': list(self.pos),
      'health': self.health,
      'grown': self.grown
    }
    return ('plant', data)

  def load(self, data):
    self.pos = np.array(data['pos'], dtype=int)
    self.health = data['health']
    self.grown = data['grown']

  def serialize(self):
    return np.array([4, self.pos[0], self.pos[1], self.health, self.grown >> 8, self.grown & ((1 << 8) - 1)], dtype=np.uint8)

  def deserialize(self, seq, pos):
    self.pos = np.array([seq[pos + 1], seq[pos + 2]], dtype=int)
    self.health = seq[pos + 3]
    self.grown = (seq[pos + 4] << 8) ^ seq[pos + 5]
    return pos + 6

  @property
  def texture(self):
    if self.ripe:
      return 'plant-ripe'
    else:
      return 'plant'

  @property
  def ripe(self):
    return self.grown > 300

  def update(self):
    self.grown += 1
    objs = [self.world[self.pos + dir_][1] for dir_ in self.all_dirs]
    if any(isinstance(obj, (Zombie, Skeleton, Cow)) for obj in objs):
      self.health -= 1
    if self.health <= 0:
      self.world.remove(self)


# id: 5
class Fence(Object):

  def __init__(self, world, pos):
    super().__init__(world, pos)

  def export(self):
    data = {
      'pos': list(self.pos)
    }
    return ('fence', data)

  def load(self, data):
    self.pos = np.array(data['pos'], dtype=int)

  def serialize(self):
    return np.array([5, self.pos[0], self.pos[1]], dtype=np.uint8)

  def deserialize(self, seq, pos):
    self.pos = np.array([seq[pos + 1], seq[pos + 2]], dtype=int)
    return pos + 3

  @property
  def texture(self):
    return 'fence'

  def update(self):
    pass
