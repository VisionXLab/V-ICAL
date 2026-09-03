"""
自定义 LavaCrossing 环境
暴露 size 和 num_crossings 参数，允许自由调整网格大小和熔岩河数量。
"""

from minigrid.envs.crossing import CrossingEnv
from minigrid.core.world_object import Lava


class CustomLavaCrossingEnv(CrossingEnv):
    """CrossingEnv 的自定义版本，通过 gym.make kwargs 自由配置参数。

    Parameters
    ----------
    size : int
        网格边长（必须为奇数）。默认 11。
    num_crossings : int
        熔岩河数量。默认 5。
    """

    def __init__(self, size=11, num_crossings=5, max_steps=None, **kwargs):
        super().__init__(
            size=size,
            num_crossings=num_crossings,
            obstacle_type=Lava,
            max_steps=max_steps,
            **kwargs,
        )
