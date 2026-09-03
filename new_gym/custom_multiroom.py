"""
自定义 MultiRoom 环境
暴露 grid_size、num_rooms、max_room_size 参数，解除原版 25×25 硬编码。
"""

from minigrid.envs.multiroom import MultiRoomEnv


class CustomMultiRoomEnv(MultiRoomEnv):
    """MultiRoomEnv 的自定义版本，总网格大小可自由配置。

    Parameters
    ----------
    grid_size : int
        总网格边长（原版硬编码为 25）。默认 25。
    num_rooms : int
        房间数量（minNumRooms = maxNumRooms）。默认 6。
    max_room_size : int
        单个房间最大边长（≥4）。默认 10。
    """

    def __init__(self, grid_size=25, num_rooms=6, max_room_size=10,
                 max_steps=None, **kwargs):
        # 先调用父类（会硬编码 self.size = 25）
        super().__init__(
            minNumRooms=num_rooms,
            maxNumRooms=num_rooms,
            maxRoomSize=max_room_size,
            max_steps=max_steps,
            **kwargs,
        )
        # 覆盖硬编码的网格大小
        self.size = grid_size
        self.width = grid_size
        self.height = grid_size
