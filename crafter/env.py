import collections

import numpy as np

from copy import deepcopy

from . import constants
from . import engine
from . import objects
from . import worldgen

# Gym is an optional dependency.
try:
    import gym

    DiscreteSpace = gym.spaces.Discrete
    BoxSpace = gym.spaces.Box
    DictSpace = gym.spaces.Dict
    BaseClass = gym.Env
except ImportError:
    DiscreteSpace = collections.namedtuple('DiscreteSpace', 'n')
    BoxSpace = collections.namedtuple('BoxSpace', 'low, high, shape, dtype')
    DictSpace = collections.namedtuple('DictSpace', 'spaces')
    BaseClass = object


class Env(BaseClass):

    def __init__(
            self, area=(64, 64), view=(9, 9), size=(64, 64),
            reward=True, render_centering=True, show_inventory=True, partial_achievements=None, disable_place_stone=False, 
            large_step=False, static_environment=False, achievement_reward_coef=1.0,
            health_reward_coef=0.1, alive_reward=0.0, immortal=False, idle_death=False, length=10000, seed=None):
        view = np.array(view if hasattr(view, '__len__') else (view, view))
        size = np.array(size if hasattr(size, '__len__') else (size, size))
        seed = np.random.randint(0, 2 ** 31 - 1) if seed is None else seed
        self._area = area
        self._view = view
        self._size = size
        self._reward = reward
        self._length = length
        self._seed = seed
        self._episode = 0
        self._achievement_reward_coef = achievement_reward_coef
        self._health_reward_coef = health_reward_coef
        self._alive_reward = alive_reward
        self._immortal = immortal
        self._show_inventory = show_inventory
        self._world = engine.World(area, constants.materials, (12, 12))
        self._textures = engine.Textures(constants.root / 'assets')
        item_rows = int(np.ceil(len(constants.items) / view[0]))
        self._local_view = engine.LocalView(
            self._world, self._textures, [view[0], view[1] - item_rows])
        self._item_view = engine.ItemView(
            self._textures, [view[0], item_rows])
        self._sem_view = engine.SemanticView(self._world, [
            objects.Player, objects.Cow, objects.Zombie,
            objects.Skeleton, objects.Arrow, objects.Plant])
        self._step = None
        self._start_progress = None
        self._player = None
        self._last_health = None
        self._unlocked = None
        self._render_centering = render_centering
        self._idle_death = idle_death
        if idle_death:
            self._idle_countdown = idle_death
        if partial_achievements is not None and partial_achievements in constants.partial_achievements.keys():
            self._partial_achievements = set(constants.partial_achievements[partial_achievements])
        else:
            self._partial_achievements = None
        self._actions = deepcopy(constants.actions)
        if large_step:
            if type(large_step) != int:
                large_step = 4
            for d in ["left", "right", "up", "down"]:
                self._actions += f"move_{d}_{large_step}"
        self._disable_place_stone = disable_place_stone
        self._static_environment = static_environment
        # Some libraries expect these attributes to be set.
        self.reward_range = None
        self.metadata = None

    # def serialize(self):
    #     data = {
    #         'area': list(self._area),
    #         'view': list(self._view),
    #         'size': list(self._size),
    #         'reward': self._reward,
    #         'length': self._length,
    #         'seed': self._seed,
    #         'achievement_reward_coef': self.__achievement_reward_coef,
    #         'health_reward_coef': self._health_reward_coef,
    #         'alive_reward': self._alive_reward,
    #         'render_centering': self._render_centering,
    #         'show_inventory': self._show_inventory,
    #         'immortal': self._immortal,
    #     }

    @property
    def observation_space(self):
        return BoxSpace(0, 255, tuple(self._size) + (3,), np.uint8)

    @property
    def action_space(self):
        return DiscreteSpace(len(self._actions))

    @property
    def action_names(self):
        return self._actions

    def seed(self, _seed):
        self._seed = _seed

    def reset_episode(self):
        self._episode = 0

    def reset(self, data=None):
        # print(f"env seed: {self._seed}")
        self._episode += 1
        self._step = 0
        self._idle_countdown = self._idle_death
        world_seed = hash((self._seed, 0 if self._static_environment else self._episode)) % (2 ** 31 - 1)
        self._world.reset(seed=world_seed)

        if data is None:
            self._start_progress = 0.3
            self._update_time()
            center = (self._world.area[0] // 2, self._world.area[1] // 2)
            self._player = objects.Player(self._world, center, self._immortal)
            self._last_health = self._player.health
            self._world.add(self._player)
            self._unlocked = set()
            worldgen.generate_world(self._world, self._player)
        else:
            self.load(data)
        # print(f"world.reset(), ep={self._episode}, seed={world_seed}")
        # self._world.display()
        return self._obs()

    def export(self):
        return {
            'world': self._world.export(),
            'player': self._player.export(),
            'objects': self._world.export_objects(),
            'progress': self._progress
        }

    def load(self, data):
        self._start_progress = data['progress']
        self._update_time()
        self._player = objects.Player(self._world, data['player']['pos'], self._immortal)
        self._player.load(data['player'])
        self._last_health = self._player.health
        self._world.add(self._player)
        self._unlocked = {
            name for name, count in sorted(self._player.achievements.items())  # sorted to avoid randomness
            if count > 0}
        self._world.load(data['world'])
        worldgen.recover_objects(self._world, self._player, data['objects'])

    def step(self, action):
        self._step += 1
        self._update_time()
        self._player.action = self._process_action(self._actions[action])
        for obj in self._world.objects:
            if self._player.distance(obj) < 2 * max(self._view):
                obj.update()
        if self._step % 10 == 0:
            for chunk, objs in sorted(self._world.chunks.items()):  # sorted to avoid randomness
                # xmin, xmax, ymin, ymax = chunk
                # center = (xmax - xmin) // 2, (ymax - ymin) // 2
                # if self._player.distance(center) < 4 * max(self._view):
                self._balance_chunk(chunk, sorted(objs, key=lambda o: [type(o).__name__] + o.pos.tolist()))
        obs = self._obs()
        reward = (self._player.health - self._last_health) * self._health_reward_coef
        reward += self._alive_reward
        self._last_health = self._player.health
        unlocked = {
            name for name, count in sorted(self._player.achievements.items())  # sorted to avoid randomness
            if count > 0 and name not in self._unlocked}
        truely_unlocked = False
        if unlocked:
            self._unlocked |= unlocked
            if self._partial_achievements is not None:
                for name in unlocked:
                    if name in self._partial_achievements:
                        reward += self._achievement_reward_coef
                        truely_unlocked = True
                        break
            else:
                reward += 1.0
                truely_unlocked = True
        
        dead = self._player.health <= 0
        if self._idle_death:
            if truely_unlocked:
                self._idle_countdown = self._idle_death
            else:
                self._idle_countdown -= 1
                if self._idle_countdown == 0:
                    dead = True
        over = self._length and self._step >= self._length
        done = dead or over
        info = {
            'inventory': self._player.inventory.copy(),
            'achievements': self._player.achievements.copy(),
            'discount': 1 - float(dead),
            'semantic': self._sem_view(),
            'player_pos': self._player.pos,
            'reward': reward,
        }
        if not self._reward:
            reward = 0.0
        return obs, reward, done, info

    def render(self, size=None):
        size = size or self._size
        unit = size // self._view
        canvas = np.zeros(tuple(size) + (3,), np.uint8)
        local_view = self._local_view(self._player, unit)
        item_view = self._item_view(self._player.inventory, unit)
        if not self._show_inventory:
            item_view = np.zeros_like(item_view)
        view = local_view
        view = np.concatenate([local_view, item_view], 1)
        if self._render_centering:
            border = (size - (size // self._view) * self._view) // 2
        else:
            border = (0, 0)
        (x, y), (w, h) = border, view.shape[:2]
        canvas[x: x + w, y: y + h] = view
        return canvas.transpose((1, 0, 2))

    def _process_action(self, action):
        if self._disable_place_stone:
            if action == "place_stone":
                return "noop"
        return action

    def _obs(self):
        return self.render()

    @property
    def _progress(self):
        return (self._step / 300) % 1 + self._start_progress

    def _update_time(self):
        # https://www.desmos.com/calculator/grfbc6rs3h
        daylight = 1 - np.abs(np.cos(np.pi * self._progress)) ** 3
        self._world.daylight = daylight

    def _balance_chunk(self, chunk, objs):
        light = self._world.daylight
        self._balance_object(
            chunk, objs, objects.Zombie, 'grass', 6, 0, 0.3, 0.4,
            lambda pos: objects.Zombie(self._world, pos, self._player),
            lambda num, space: (
                0 if space < 50 else 3.5 - 3 * light, 3.5 - 3 * light))
        self._balance_object(
            chunk, objs, objects.Skeleton, 'path', 7, 7, 0.1, 0.1,
            lambda pos: objects.Skeleton(self._world, pos, self._player),
            lambda num, space: (0 if space < 6 else 1, 2))
        self._balance_object(
            chunk, objs, objects.Cow, 'grass', 5, 5, 0.01, 0.1,
            lambda pos: objects.Cow(self._world, pos),
            lambda num, space: (0 if space < 30 else 1, 1.5 + light))

    def _balance_object(
            self, chunk, objs, cls, material, span_dist, despan_dist,
            spawn_prob, despawn_prob, ctor, target_fn):
        xmin, xmax, ymin, ymax = chunk
        random = self._world.random
        creatures = [obj for obj in objs if isinstance(obj, cls)]
        mask = self._world.mask(*chunk, material)
        target_min, target_max = target_fn(len(creatures), mask.sum())
        if len(creatures) < int(target_min) and random.uniform() < spawn_prob:
            xs = np.tile(np.arange(xmin, xmax)[:, None], [1, ymax - ymin])
            ys = np.tile(np.arange(ymin, ymax)[None, :], [xmax - xmin, 1])
            xs, ys = xs[mask], ys[mask]
            i = random.randint(0, len(xs))
            pos = np.array((xs[i], ys[i]))
            empty = self._world[pos][1] is None
            away = self._player.distance(pos) >= span_dist
            if empty and away:
                # print(f"Step {self._step}, spawned {cls.__name__} at {pos}!")
                self._world.add(ctor(pos))
        elif len(creatures) > int(target_max) and random.uniform() < despawn_prob:
            i = random.randint(0, len(creatures))
            obj = creatures[i]
            away = self._player.distance(obj.pos) >= despan_dist
            if away:
                # print(f"Step {self._step}, despawned creatures[{i}] in chunk {chunk}: {cls.__name__} at {obj.pos}!")
                self._world.remove(obj)
