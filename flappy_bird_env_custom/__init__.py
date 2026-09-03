from .flappy_bird_env import FlappyBirdEnv
from gymnasium.envs.registration import register

__all__ = [FlappyBirdEnv]

register(id="aFlappyBird-v1", entry_point="flappy_bird_env_custom:FlappyBirdEnv")
