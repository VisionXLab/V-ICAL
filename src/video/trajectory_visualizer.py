"""轨迹摘要图生成器

为 AI/人类游戏会话生成可视化轨迹图：
- 网格游戏（Taxi-v3 / CliffWalking-v1 / FrozenLake-v1 / MiniGrid）：
    在初始帧上叠加行走序号圆点
    Taxi-v3 额外：红色=接客前，绿色=接客后，蓝色大圆=接客格，金色大圆=放客格
- 非网格游戏（Atari / CartPole / Acrobot / Blackjack / CustomBreakout）：
    初始帧缩略图 + 换行动作时间轴（每行 20 步）

生成失败时返回 None，不影响正常保存流程。
"""
import hashlib
import io
import re
import traceback
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

from .frame_processor import get_caption_font


# ─── 游戏分类 ─────────────────────────────────────────────────────────────────

_GRID_TOYTEXT_GAMES = {"Taxi-v3", "CliffWalking-v1", "FrozenLake-v1"}
_MINIGRID_PREFIX = "MiniGrid-"


def _is_grid_game(game_name: str) -> bool:
    return game_name in _GRID_TOYTEXT_GAMES or game_name.startswith(_MINIGRID_PREFIX)


def _is_minigrid_game(game_name: str) -> bool:
    return game_name.startswith(_MINIGRID_PREFIX)


# ─── 状态解码 ─────────────────────────────────────────────────────────────────

def _decode_taxi(s: int) -> Tuple[int, int, int, int]:
    """解码 Taxi-v3 状态整数 → (row, col, pass_loc, dest)"""
    dest = s % 4
    s //= 4
    pass_loc = s % 5  # 0-3=固定站点, 4=在车上
    s //= 5
    col = s % 5
    row = s // 5
    return row, col, pass_loc, dest


def _decode_cliff(s: int) -> Tuple[int, int]:
    """解码 CliffWalking-v1 状态 → (row, col)，网格 4×12"""
    return s // 12, s % 12


def _decode_frozen(s: int, map_size: int) -> Tuple[int, int]:
    """解码 FrozenLake-v1 状态 → (row, col)"""
    return s // map_size, s % map_size


# ─── 网格尺寸 ─────────────────────────────────────────────────────────────────

def _get_grid_dims(session) -> Tuple[int, int]:
    """返回 (grid_rows, grid_cols)"""
    game = session.game_name
    if game == "Taxi-v3":
        return 5, 5
    elif game == "CliffWalking-v1":
        return 4, 12
    elif game == "FrozenLake-v1":
        try:
            uw = session._get_unwrapped()
            sz = len(uw.desc[0])
            return sz, sz
        except Exception:
            return 4, 4
    elif _is_minigrid_game(game):
        try:
            uw = session._get_unwrapped()
            return int(uw.height), int(uw.width)
        except Exception:
            m = re.search(r'(\d+)x(\d+)', game)
            if m:
                # MiniGrid-DoorKey-16x16-v0 → width=16, height=16
                return int(m.group(2)), int(m.group(1))
            return 19, 19  # FourRooms 默认
    return 8, 8


# ─── 位置提取 ─────────────────────────────────────────────────────────────────

def _extract_positions(session) -> List[Optional[Tuple[int, int]]]:
    """
    从 session._snapshots 提取每步的 (row, col)。
    列表长度 = len(_snapshots) = 步数 + 1（含初始状态）。
    无法解码时该位置为 None。
    """
    game = session.game_name
    positions = []

    frozen_size = None
    if game == "FrozenLake-v1":
        try:
            uw = session._get_unwrapped()
            frozen_size = len(uw.desc[0])
        except Exception:
            frozen_size = 4

    for snap in session._snapshots:
        if snap is None:
            positions.append(None)
            continue

        kind, data = snap
        if kind != "fields":
            # ALE / deepcopy 快照不含位置信息
            positions.append(None)
            continue

        if _is_minigrid_game(game):
            ap = data.get("agent_pos")
            if ap is not None:
                # MiniGrid agent_pos = (col, row)，转为 (row, col)
                positions.append((int(ap[1]), int(ap[0])))
            else:
                positions.append(None)

        elif "s" in data:
            s = int(data["s"])
            if game == "Taxi-v3":
                row, col, *_ = _decode_taxi(s)
                positions.append((row, col))
            elif game == "CliffWalking-v1":
                positions.append(_decode_cliff(s))
            elif game == "FrozenLake-v1":
                positions.append(_decode_frozen(s, frozen_size or 4))
            else:
                positions.append(None)
        else:
            positions.append(None)

    return positions


# ─── 标题条 ───────────────────────────────────────────────────────────────────

def _draw_title_bar(img: Image.Image, text: str, bar_h: int = 22):
    """在 RGB 图片顶部原地绘制深色标题条（不扩大图片尺寸）"""
    draw = ImageDraw.Draw(img)
    font = get_caption_font(13)
    draw.rectangle([(0, 0), (img.width, bar_h)], fill=(20, 20, 30))
    draw.text((6, 3), text, font=font, fill=(220, 230, 255))


# ─── 网格轨迹可视化 ───────────────────────────────────────────────────────────

def _grid_trajectory(session,seq_count=0) -> Image.Image:
    """在初始帧上叠加轨迹序号圆点，返回 RGB Image"""
    game = session.game_name

    # 初始帧作底图
    bg = Image.open(io.BytesIO(session.frames[seq_count])).convert("RGBA")
    img_w, img_h = bg.size

    grid_rows, grid_cols = _get_grid_dims(session)
    cell_w = img_w / grid_cols
    cell_h = img_h / grid_rows

    def cell_center(row, col):
        return (col * cell_w + cell_w / 2, row * cell_h + cell_h / 2)

    positions = _extract_positions(session)

    # ── Taxi-v3：接客状态跟踪 + 特殊事件 ──────────────────────────────────────
    taxi_carrying: List[bool] = []
    taxi_events: List[Tuple] = []  # (snap_idx, "pickup"|"dropoff", row, col)

    if game == "Taxi-v3":
        for snap in session._snapshots:
            if snap and snap[0] == "fields" and "s" in snap[1]:
                _, _, pass_loc, _ = _decode_taxi(int(snap[1]["s"]))
                taxi_carrying.append(pass_loc == 4)
            else:
                taxi_carrying.append(False)

        # 接客事件：carrying 首次变为 True
        for i in range(1, len(taxi_carrying)):
            if taxi_carrying[i] and not taxi_carrying[i - 1]:
                if i < len(positions) and positions[i] is not None:
                    taxi_events.append((i, "pickup", *positions[i]))

        # 放客事件：action=5 且奖励为正
        for i, (act, rew) in enumerate(zip(session.action_history[seq_count:], session.reward_history[seq_count:])):
            if act == 5 and rew > 0:
                snap_idx = i + 1
                if snap_idx < len(positions) and positions[snap_idx] is not None:
                    taxi_events.append((snap_idx, "dropoff", *positions[snap_idx]))

    # ── 绘制层 ───────────────────────────────────────────────────────────────
    overlay = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    radius = max(5, int(min(cell_w, cell_h) * 0.22))
    font_size_base = max(8, int(min(cell_w, cell_h) * 0.28))

    # 先画特殊事件大圆（底层）
    for _, ev_type, row, col in taxi_events:
        cx, cy = cell_center(row, col)
        r2 = radius * 1.7
        if ev_type == "pickup":
            fill = (100, 149, 237, 150)   # 蓝色：接客站
        else:
            fill = (255, 215, 0, 180)     # 金色：放客站
        draw.ellipse([(cx - r2, cy - r2), (cx + r2, cy + r2)], fill=fill)

    # 按格子分组：同一格子的所有访问步骤合并后一次性绘制，避免覆盖
    from collections import defaultdict
    cell_visits: dict = defaultdict(list)  # (row, col) -> [step_idx, ...]
    for i, pos in enumerate(positions):
        if pos is not None:
            cell_visits[pos].append(i)

    for (row, col), visit_steps in cell_visits.items():
        cx, cy = cell_center(row, col)

        # 决定颜色（Taxi-v3 区分接客前/后/混合）
        if game == "Taxi-v3":
            has_before = any(
                i < len(taxi_carrying) and not taxi_carrying[i] for i in visit_steps
            )
            has_carrying = any(
                i < len(taxi_carrying) and taxi_carrying[i] for i in visit_steps
            )
            if has_before and has_carrying:
                fill = (210, 140, 40, 210)   # 橙色：两阶段均有
            elif has_carrying:
                fill = (50, 200, 50, 210)    # 绿色：接客后
            else:
                fill = (220, 50, 50, 210)    # 红色：接客前
        else:
            fill = (80, 130, 220, 210)

        # 标签：≤5次全列，超过则压缩为 "首..尾 ×N"
        if len(visit_steps) <= 5:
            label = ",".join(str(s) for s in visit_steps)
        else:
            label = f"{visit_steps[0]}..{visit_steps[-1]}\n×{len(visit_steps)}"

        # 字体随标签长度动态缩小，防止溢出圆圈
        max_line_len = max(len(ln) for ln in label.split("\n"))
        shrink = max(1.0, max_line_len / 4)
        font_size = max(6, int(font_size_base / shrink))
        font = get_caption_font(font_size)

        draw.ellipse(
            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
            fill=fill,
        )

        # 多行文字垂直居中写入圆内
        lines = label.split("\n")
        line_sizes = []
        total_h = 0
        for ln in lines:
            bb = draw.textbbox((0, 0), ln, font=font)
            lh = bb[3] - bb[1]
            lw = bb[2] - bb[0]
            line_sizes.append((lw, lh))
            total_h += lh
        total_h += (len(lines) - 1) * 2

        y = cy - total_h / 2
        for ln, (lw, lh) in zip(lines, line_sizes):
            draw.text(
                (cx - lw / 2, y),
                ln,
                font=font,
                fill=(255, 255, 255, 240),
            )
            y += lh + 2

    result = Image.alpha_composite(bg, overlay).convert("RGB")

    _draw_title_bar(
        result,
        f"轨迹 — {game} | {len(session.action_history[seq_count:])} 步 | 总奖励 {sum(session.reward_history[seq_count:]):.1f}",
    )
    return result


# ─── 时间轴轨迹可视化 ─────────────────────────────────────────────────────────

def _action_color(action_name: str) -> Tuple[int, int, int]:
    """按动作名哈希生成背景色（确保亮度适中）"""
    h = hashlib.md5(action_name.encode()).digest()
    r = 60 + h[0] % 160
    g = 60 + h[1] % 160
    b = 60 + h[2] % 160
    return (r, g, b)


def _timeline_trajectory(session,seq_count=0) -> Image.Image:
    """初始帧缩略图（左）+ 换行动作时间轴（右），返回 RGB Image"""
    game = session.game_name
    actions = session.action_history[seq_count:]
    rewards = session.reward_history[seq_count:]
    action_info = session.action_info

    # 布局参数
    thumb_w = 200
    steps_per_row = 20
    block_w = 36
    block_h = 30
    gap = 2
    pad = 12
    header_h = 30

    # 缩略图
    thumb = Image.open(io.BytesIO(session.frames[seq_count])).convert("RGB")
    orig_w, orig_h = thumb.size
    thumb_h = max(60, int(thumb_w * orig_h / orig_w))
    thumb = thumb.resize((thumb_w, thumb_h), Image.LANCZOS)

    # 时间轴区域尺寸
    n_steps = len(actions)
    n_rows = max(1, (n_steps + steps_per_row - 1) // steps_per_row)
    tl_w = steps_per_row * (block_w + gap) + 2 * pad
    tl_h = n_rows * (block_h + gap) + 20 + pad  # 20 = 行标题

    content_h = max(thumb_h, tl_h)
    total_h = header_h + content_h + pad
    total_w = pad + thumb_w + pad + tl_w + pad

    img = Image.new("RGB", (total_w, total_h), (25, 28, 38))
    draw = ImageDraw.Draw(img)

    font_sm = get_caption_font(11)
    font_hd = get_caption_font(14)

    # 顶部标题
    pos_rew = sum(r for r in rewards if r > 0)
    neg_rew = sum(r for r in rewards if r < 0)
    title = (
        f"{game} | {n_steps} 步 | "
        f"总奖励 {sum(rewards):.1f}  (+{pos_rew:.0f} / {neg_rew:.0f})"
    )
    draw.rectangle([(0, 0), (total_w, header_h)], fill=(20, 22, 35))
    draw.text((pad, 6), title, font=font_hd, fill=(200, 220, 255))

    # 缩略图
    img.paste(thumb, (pad, header_h))

    # 时间轴标题
    tl_x = pad + thumb_w + pad
    tl_y = header_h
    draw.text(
        (tl_x, tl_y + 2),
        "动作时间轴（每行20步，★正奖励，✗负奖励）",
        font=font_sm,
        fill=(140, 160, 190),
    )

    # 动作块
    block_y0 = tl_y + 20
    for i, (act, rew) in enumerate(zip(actions, rewards)):
        r = i // steps_per_row
        c = i % steps_per_row
        bx = tl_x + c * (block_w + gap)
        by = block_y0 + r * (block_h + gap)

        act_name = action_info.get(act, str(act))
        color = _action_color(act_name)
        draw.rectangle([(bx, by), (bx + block_w - 1, by + block_h - 1)], fill=color)

        # 动作缩写（最多 4 字符）
        abbr = act_name[:4]
        bb = draw.textbbox((0, 0), abbr, font=font_sm)
        tw = bb[2] - bb[0]
        th = bb[3] - bb[1]
        draw.text(
            (bx + (block_w - tw) // 2, by + (block_h - th) // 2 - 1),
            abbr,
            font=font_sm,
            fill=(255, 255, 255),
        )

        # 奖励标记
        if rew > 0:
            draw.text((bx + block_w - 9, by), "★", font=font_sm, fill=(255, 220, 0))
        elif rew < 0:
            draw.text((bx + block_w - 9, by), "✗", font=font_sm, fill=(255, 80, 80))

    return img


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def generate_trajectory_image(session,seq_count=0) -> Optional[Image.Image]:
    """
    生成轨迹摘要图。

    Returns:
        PIL.Image.Image；失败时返回 None（不影响正常保存流程）。
    """
    try:
        if not session.frames:
            return None
        if _is_grid_game(session.game_name):
            return _grid_trajectory(session,seq_count)
        else:
            return _timeline_trajectory(session,seq_count)
    except Exception as e:
        print(f"[TrajectoryVisualizer] Error generating trajectory: {e}")
        traceback.print_exc()
        return None
