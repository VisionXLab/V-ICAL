"""
自定义 UnlockPickup 环境
暴露 room_size、num_rows、num_cols 参数，解除原版硬编码。
"""

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid


class CustomUnlockPickupEnv(RoomGrid):
    """UnlockPickupEnv 的自定义版本，房间大小和布局可自由配置。

    Parameters
    ----------
    room_size : int
        每个房间的边长（≥4）。默认 6。
    num_rows : int
        房间行数。默认 1。
    num_cols : int
        房间列数（≥2，因为 box 在 room(1,0)）。默认 2。
    """

    def __init__(self, room_size=6, num_rows=1, num_cols=2,
                 max_steps=None, **kwargs):
        assert num_cols >= 2, "num_cols must be >= 2 (box is placed in room(1,0))"

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES],
        )

        if max_steps is None:
            max_steps = 8 * room_size ** 2

        super().__init__(
            mission_space=mission_space,
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str):
        return f"pick up the {color} box"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.carrying and self.carrying == self.obj:
            reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info
