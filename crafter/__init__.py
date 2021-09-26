from .env import Env
from .recorder import Recorder

try:
    import gym

    gym.register(
        id='CrafterReward-v1',
        entry_point='crafter:Env',
        max_episode_steps=10000,
        kwargs={'reward': True})
    gym.register(
        id='CrafterNoReward-v1',
        entry_point='crafter:Env',
        max_episode_steps=10000,
        kwargs={'reward': False})
    gym.register(
        id='CrafterPartialCollectIron-v1',
        entry_point='crafter:Env',
        max_episode_steps=10000,
        kwargs={'reward': True, 'partial_achievements': 'collect_iron'})
except ImportError:
    pass
