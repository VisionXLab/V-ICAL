"""从保存的 actions.json 确定性重建逐帧轨迹。"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from src.video.frame_processor import add_caption, add_ending_label, save_video
from tools._paths import ROOT


YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BOUNDARY_MAE_LIMIT = 12.0


@dataclass(frozen=True)
class PrimitiveAction:
    """一个可渲染的环境原始步。"""

    action: int
    is_keypress: bool
    source: str


@dataclass(frozen=True)
class TrajectorySpec:
    actions_path: Path
    game_name: str
    config: dict
    seed: int | None
    actions: list[int]
    ending: str | None
    video_fps: float

    @property
    def repeat(self) -> int:
        return max(1, int(self.config.get("repeat", 1)))

    @property
    def action_repeat(self) -> int | None:
        value = self.config.get("action_repeat")
        return None if value is None else int(value)

    @property
    def noop_fill(self) -> bool:
        return bool(self.config.get("noop_fill", False))

    @property
    def primitive_frameskip(self) -> int:
        if self.game_name.startswith("ALE/"):
            return max(1, int(self.config.get("frameskip", 4)))
        return 1


@dataclass(frozen=True)
class FillResult:
    actions_path: Path
    output_dir: Path
    frame_count: int
    fps: float
    boundary_checks: list[dict]


def _single_file_base(actions_path: Path) -> Path:
    for ancestor in actions_path.parents:
        if ancestor.name == "batch_results":
            return ancestor
    run_dir = actions_path.parent
    if run_dir.name.startswith("run_") and len(run_dir.parents) >= 2:
        return run_dir.parents[1]
    return run_dir.parent


def discover_actions(path: Path) -> tuple[list[Path], Path]:
    """发现输入下的 actions.json，并返回用于保持目录结构的扫描基准。"""

    source = path.expanduser().resolve()
    if source.is_file():
        if source.name != "actions.json":
            raise ValueError(f"输入文件必须名为 actions.json: {source}")
        return [source], _single_file_base(source)
    if not source.is_dir():
        raise FileNotFoundError(f"输入不存在: {source}")
    return sorted(source.rglob("actions.json")), source.parent


def output_relative_path(actions_path: Path, scan_base: Path) -> Path:
    """计算轨迹在时间戳输出根下的相对 run 目录。"""

    try:
        return actions_path.parent.resolve().relative_to(scan_base.resolve())
    except ValueError:
        return Path(actions_path.parent.name)


def expand_action(
    action: int,
    repeat: int,
    action_repeat: int | None,
    noop_fill: bool,
    frameskip: int,
) -> list[PrimitiveAction]:
    """把一次高层按键展开成项目 repeat 与 ALE frameskip 对应的原始步。"""

    repeat = max(1, int(repeat))
    frameskip = max(1, int(frameskip))
    if noop_fill and action_repeat is not None:
        pressed_slots = max(1, min(repeat, int(action_repeat)))
    else:
        pressed_slots = repeat

    result: list[PrimitiveAction] = []
    for slot in range(repeat):
        slot_action = int(action) if slot < pressed_slots else 0
        for inner in range(frameskip):
            first = slot == 0 and inner == 0
            if first:
                source = "keypress"
            elif inner > 0:
                source = "ale_frameskip"
            elif slot_action == 0 and slot >= pressed_slots:
                source = "noop"
            elif noop_fill:
                source = "action_repeat"
            else:
                source = "repeat"
            result.append(PrimitiveAction(slot_action, first, source))
    return result


def load_trajectory(actions_path: Path) -> TrajectorySpec:
    """读取并校验补帧所需的最小轨迹字段。"""

    path = actions_path.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    game_name = data.get("game_name")
    config = data.get("config")
    actions = data.get("actions")
    if not isinstance(game_name, str) or not game_name:
        raise ValueError(f"缺少有效 game_name: {path}")
    if not isinstance(config, dict):
        raise ValueError(f"缺少有效 config: {path}")
    if not isinstance(actions, list) or any(not isinstance(item, int) for item in actions):
        raise ValueError(f"actions 必须是整数列表: {path}")
    seed = data.get("seed", config.get("seed"))
    if seed is not None:
        seed = int(seed)
    video_fps = float(data.get("video_fps", 1.0))
    if video_fps <= 0:
        raise ValueError(f"video_fps 必须大于 0: {path}")
    return TrajectorySpec(
        actions_path=path,
        game_name=game_name,
        config=dict(config),
        seed=seed,
        actions=list(actions),
        ending=data.get("ending"),
        video_fps=video_fps,
    )


def normalized_replay_config(spec: TrajectorySpec) -> dict:
    """把原高层配置转换成一次 step 对应一张原始帧的重放配置。"""

    config = dict(spec.config)
    if spec.seed is not None:
        config["seed"] = spec.seed
    config["repeat"] = 1
    config["noop_fill"] = False
    config.pop("action_repeat", None)
    if spec.game_name.startswith("ALE/"):
        source_frameskip = spec.primitive_frameskip
        config["frameskip"] = 1
        if config.get("skip_initial_steps"):
            config["skip_initial_steps"] = int(config["skip_initial_steps"]) * source_frameskip
    return config


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return data


def resolve_preload_actions(actions_path: Path, root: Path = ROOT) -> list[int]:
    """按 result.json → ai_config → sequence.json 解析预加载动作。"""

    result_path = actions_path.parent / "result.json"
    if not result_path.exists():
        return []
    result = _read_json(result_path)
    config_ref = result.get("config_path")
    if not config_ref:
        return []
    config_path = Path(config_ref)
    if not config_path.is_absolute():
        config_path = root / config_path
    if config_path.is_dir():
        config_path = config_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"无法读取原 AI 配置: {config_path}")
    ai_config = _read_json(config_path)
    sequence_ref = ai_config.get("action_sequence")
    if not sequence_ref:
        return []
    if not isinstance(sequence_ref, dict):
        raise ValueError(f"action_sequence 必须是对象: {config_path}")
    game_safe = sequence_ref.get("game_safe")
    name = sequence_ref.get("name")
    if not game_safe or not name:
        raise ValueError(f"action_sequence 缺少 game_safe/name: {config_path}")
    sequence_path = root / "action_sequences" / str(game_safe) / str(name) / "sequence.json"
    if not sequence_path.exists():
        raise FileNotFoundError(f"预加载动作序列不存在: {sequence_path}")
    sequence = _read_json(sequence_path).get("actions", [])
    if not isinstance(sequence, list) or any(not isinstance(item, int) for item in sequence):
        raise ValueError(f"预加载 actions 必须是整数列表: {sequence_path}")
    return list(sequence)


def _image_from_jpeg(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")


def _boundary_check(generated: Image.Image, source_path: Path, action_index: int) -> dict:
    source = Image.open(source_path).convert("RGB")
    if source.size != generated.size:
        raise RuntimeError(
            f"边界帧尺寸不一致 action={action_index}: source={source.size}, replay={generated.size}"
        )
    source_array = np.asarray(source, dtype=np.int16)
    generated_array = np.asarray(generated, dtype=np.int16)
    mae = float(np.abs(source_array - generated_array).mean())
    check = {
        "action_index": action_index,
        "source_frame": source_path.name,
        "status": "exact" if mae == 0 else "near",
        "mean_absolute_error": round(mae, 6),
    }
    if mae > BOUNDARY_MAE_LIMIT:
        raise RuntimeError(
            f"重放边界明显漂移 action={action_index}, MAE={mae:.3f} > {BOUNDARY_MAE_LIMIT}"
        )
    return check


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fuse_flicker_frames(game_name: str, frames: list[Image.Image]) -> list[Image.Image]:
    """仅为 Berzerk 产物合并相邻奇偶闪烁帧，不改变帧数或回放状态。"""
    if game_name != "ALE/Berzerk-v5" or len(frames) < 2:
        return list(frames)

    fused = [frames[0]]
    for previous, current in zip(frames, frames[1:]):
        merged = np.maximum(np.asarray(previous), np.asarray(current))
        fused.append(Image.fromarray(merged.astype(np.uint8), mode="RGB"))
    return fused


def _write_artifacts(
    temp_dir: Path,
    spec: TrajectorySpec,
    session,
    original_frames: list[Image.Image],
    frame_records: list[dict],
    boundary_checks: list[dict],
    fps: float,
) -> None:
    artifact_frames = fuse_flicker_frames(spec.game_name, original_frames)
    frames_dir = temp_dir / "frames"
    subbed_dir = temp_dir / "subbed"
    subbed_nl_dir = temp_dir / "subbed_nl"
    for directory in (frames_dir, subbed_dir, subbed_nl_dir):
        directory.mkdir(parents=True, exist_ok=True)

    numeric_images: list[Image.Image] = []
    natural_images: list[Image.Image] = []
    numeric_captions: list[str] = []
    natural_captions: list[str] = []
    for index, (image, record) in enumerate(zip(artifact_frames, frame_records)):
        action = record["action"]
        if action is None:
            numeric_caption = "Start State" if record["source"] == "start" else "Ending"
            natural_caption = numeric_caption
        else:
            numeric_caption = f"Frame {index}: Action {action}"
            action_name = session.action_info.get(action, f"Action_{action}")
            natural_caption = f"Frame {index}: Action: {action_name}"
        color = YELLOW if record["is_keypress"] else WHITE
        numeric = add_caption(image, numeric_caption, text_color=color)
        natural = add_caption(image, natural_caption, text_color=color)
        image.save(frames_dir / f"step_{index:06d}.png")
        numeric.save(subbed_dir / f"step_{index:06d}.png")
        natural.save(subbed_nl_dir / f"step_{index:06d}.png")
        numeric_images.append(numeric)
        natural_images.append(natural)
        numeric_captions.append(numeric_caption)
        natural_captions.append(natural_caption)

    (temp_dir / "caption.txt").write_text("\n".join(numeric_captions), encoding="utf-8")
    (temp_dir / "caption_nl.txt").write_text("\n".join(natural_captions), encoding="utf-8")
    _write_json(
        temp_dir / "frame_manifest.json",
        {
            "game_name": spec.game_name,
            "fps": fps,
            "source_actions": len(spec.actions),
            "frames": frame_records,
            "boundary_checks": boundary_checks,
        },
    )
    source_data = _read_json(spec.actions_path)
    source_data["frame_fill"] = {
        "fps": fps,
        "total_frames": len(original_frames),
        "primitive_frameskip": spec.primitive_frameskip,
        "repeat": spec.repeat,
        "two_frame_max_pool": spec.game_name == "ALE/Berzerk-v5",
        "boundary_checks": boundary_checks,
    }
    _write_json(temp_dir / "actions.json", source_data)

    outputs = (
        (artifact_frames, temp_dir / "video_original.mp4"),
        (numeric_images, temp_dir / "video.mp4"),
        (natural_images, temp_dir / "video_nl.mp4"),
    )
    for images, output_path in outputs:
        result = save_video(images, output_path, fps)
        if not output_path.exists():
            raise RuntimeError(f"视频生成失败 {output_path.name}: {result}")


async def fill_one(
    actions_path: Path,
    target_dir: Path,
    fps_override: float | None = None,
    session_factory: Callable | None = None,
    root: Path = ROOT,
) -> FillResult:
    """重放一条轨迹，并在全部产物成功后原子发布目标目录。"""

    spec = load_trajectory(actions_path)
    if fps_override is not None and fps_override <= 0:
        raise ValueError("fps_override 必须大于 0")
    fps = float(fps_override or (spec.video_fps * spec.repeat * spec.primitive_frameskip))
    target_dir = target_dir.resolve()
    if target_dir.exists():
        raise FileExistsError(f"目标目录已存在: {target_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = target_dir.parent / f".{target_dir.name}.tmp-{uuid.uuid4().hex}"
    temp_dir.mkdir()

    session = None
    boundary_checks: list[dict] = []
    try:
        if session_factory is None:
            from src.core.game_session import GameSession

            session_factory = GameSession
        session = session_factory(
            f"frame-fill-{uuid.uuid4().hex}",
            spec.game_name,
            normalized_replay_config(spec),
        )
        await session.initialize()
        session.config["_max_steps"] = None

        preload_actions = resolve_preload_actions(spec.actions_path, root=root)
        saved_end_on_life_loss = getattr(session.env, "end_on_life_loss", False)
        if hasattr(session.env, "end_on_life_loss"):
            session.env.end_on_life_loss = False
        for preload_action in preload_actions:
            for primitive in expand_action(
                preload_action,
                spec.repeat,
                spec.action_repeat,
                spec.noop_fill,
                spec.primitive_frameskip,
            ):
                await session.step(primitive.action)
                if session.is_game_over:
                    raise RuntimeError("预加载动作序列提前结束，无法恢复正式轨迹起点")
        if hasattr(session.env, "end_on_life_loss"):
            session.env.end_on_life_loss = saved_end_on_life_loss

        initial = _image_from_jpeg(session.frames[-1])
        original_frames = [initial]
        frame_records = [
            {
                "frame_index": 0,
                "high_level_action_index": None,
                "action": None,
                "source": "start",
                "is_keypress": False,
                "is_noop": False,
                "caption_color": "white",
            }
        ]
        source_frames_dir = spec.actions_path.parent / "frames"
        source_initial = source_frames_dir / "step_000000.png"
        if source_initial.exists():
            boundary_checks.append(_boundary_check(initial, source_initial, -1))

        stopped = False
        for action_index, action in enumerate(spec.actions):
            primitives = expand_action(
                action,
                spec.repeat,
                spec.action_repeat,
                spec.noop_fill,
                spec.primitive_frameskip,
            )
            for primitive in primitives:
                await session.step(primitive.action)
                image = _image_from_jpeg(session.frames[-1])
                frame_index = len(original_frames)
                original_frames.append(image)
                frame_records.append(
                    {
                        "frame_index": frame_index,
                        "high_level_action_index": action_index,
                        "action": primitive.action,
                        "source": primitive.source,
                        "is_keypress": primitive.is_keypress,
                        "is_noop": primitive.action == 0,
                        "caption_color": "yellow" if primitive.is_keypress else "white",
                    }
                )
                if session.is_game_over:
                    stopped = True
                    break
            source_boundary = source_frames_dir / f"step_{action_index + 1:06d}.png"
            if source_boundary.exists():
                boundary_checks.append(_boundary_check(original_frames[-1], source_boundary, action_index))
            if stopped:
                break

        if spec.ending and original_frames:
            ending_index = len(original_frames)
            original_frames.append(add_ending_label(original_frames[-1].copy(), spec.ending))
            frame_records.append(
                {
                    "frame_index": ending_index,
                    "high_level_action_index": None,
                    "action": None,
                    "source": "ending",
                    "is_keypress": False,
                    "is_noop": False,
                    "caption_color": "white",
                }
            )

        _write_artifacts(
            temp_dir,
            spec,
            session,
            original_frames,
            frame_records,
            boundary_checks,
            fps,
        )
        temp_dir.replace(target_dir)
        return FillResult(
            actions_path=spec.actions_path,
            output_dir=target_dir,
            frame_count=len(original_frames),
            fps=fps,
            boundary_checks=boundary_checks,
        )
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    finally:
        if session is not None and getattr(session, "env", None) is not None:
            session.env.close()


def _new_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="扫描 actions.json，确定性重建环境并生成逐帧视频",
    )
    parser.add_argument("input", help="单个 actions.json 或包含它们的目录")
    parser.add_argument(
        "--output",
        default="interpolated_results",
        help="输出根目录（默认: interpolated_results）",
    )
    parser.add_argument("--fps", type=float, default=None, help="覆盖自动计算的视频 FPS")
    return parser


async def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps 必须大于 0")
    try:
        actions_files, scan_base = discover_actions(Path(args.input))
    except (OSError, ValueError) as exc:
        print(f"[补帧] 输入错误: {exc}")
        return 2
    if not actions_files:
        print(f"[补帧] 未发现 actions.json: {Path(args.input).resolve()}")
        return 2

    output_root = Path(args.output)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    timestamp_root = output_root.resolve() / _new_timestamp()
    succeeded = 0
    skipped = 0
    failures: list[tuple[Path, str]] = []
    print(f"[补帧] 发现 {len(actions_files)} 条轨迹")
    print(f"[补帧] 输出目录: {timestamp_root}")

    for actions_path in actions_files:
        relative = output_relative_path(actions_path, scan_base)
        target = timestamp_root / relative
        if target.exists():
            skipped += 1
            print(f"[跳过] {actions_path} -> 目标已存在")
            continue
        try:
            result = await fill_one(actions_path, target, fps_override=args.fps)
            succeeded += 1
            print(
                f"[成功] {actions_path} -> {result.output_dir} "
                f"({result.frame_count} 帧, {result.fps:g} FPS)"
            )
        except Exception as exc:
            failures.append((actions_path, str(exc)))
            print(f"[失败] {actions_path}: {exc}")

    print(f"[补帧汇总] 成功 {succeeded}，跳过 {skipped}，失败 {len(failures)}")
    if failures:
        print("[失败明细]")
        for path, reason in failures:
            print(f"  - {path}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
