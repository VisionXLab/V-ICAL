"""把 batch_results 中的 actions.json 按原目录层级导出到单独目录。"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from tools._paths import ROOT


TRAJECTORY_FILENAME = "actions.json"
BATCH_ROOT_NAME = "batch_results"


@dataclass(frozen=True)
class ExportSummary:
    copied: int
    skipped: int
    destinations: tuple[Path, ...]


def _batch_root(path: Path) -> Path:
    """返回离轨迹最近的 batch_results 祖先目录。"""

    for candidate in (path, *path.parents):
        if candidate.name == BATCH_ROOT_NAME:
            return candidate
    raise ValueError(f"路径不在 {BATCH_ROOT_NAME} 目录下: {path}")


def trajectory_relative_path(actions_path: Path) -> Path:
    """返回轨迹相对 batch_results 的完整路径（包含 actions.json）。"""

    source = actions_path.expanduser().resolve()
    if source.name != TRAJECTORY_FILENAME:
        raise ValueError(f"轨迹文件必须名为 {TRAJECTORY_FILENAME}: {source}")
    return source.relative_to(_batch_root(source))


def discover_trajectories(inputs: list[Path]) -> list[Path]:
    """发现一个或多个输入中的轨迹，去重后稳定排序。"""

    found: set[Path] = set()
    for raw_path in inputs:
        source = raw_path.expanduser().resolve()
        if source.is_file():
            if source.name != TRAJECTORY_FILENAME:
                raise ValueError(f"输入文件必须名为 {TRAJECTORY_FILENAME}: {source}")
            _batch_root(source)
            found.add(source)
            continue
        if not source.is_dir():
            raise FileNotFoundError(f"输入不存在: {source}")
        _batch_root(source)
        found.update(path.resolve() for path in source.rglob(TRAJECTORY_FILENAME))

    return sorted(found, key=lambda path: str(path).casefold())


def export_trajectories(
    actions_files: list[Path],
    output_root: Path,
    *,
    overwrite: bool = False,
) -> ExportSummary:
    """复制轨迹到 output_root，并保留 batch_results 下的相对层级。"""

    output_root = output_root.expanduser().resolve()
    copied = 0
    skipped = 0
    destinations: list[Path] = []

    for source in actions_files:
        source = source.expanduser().resolve()
        destination = output_root / trajectory_relative_path(source)
        destinations.append(destination)

        if destination == source:
            skipped += 1
            print(f"[跳过] 源文件与目标文件相同: {source}")
            continue
        if destination.exists() and not overwrite:
            skipped += 1
            print(f"[跳过] 目标已存在: {destination}")
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
        print(f"[已复制] {source} -> {destination}")

    return ExportSummary(copied, skipped, tuple(destinations))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="导出一个或多个 batch 轨迹，并保留 batch_results 下的目录层级",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="一个或多个 actions.json、run 目录或更高层 batch_results 目录",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="输出根目录（默认: output）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖输出目录中已存在的同名轨迹",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        actions_files = discover_trajectories([Path(value) for value in args.inputs])
    except (OSError, ValueError) as exc:
        print(f"[轨迹导出] 输入错误: {exc}", file=sys.stderr)
        return 2

    if not actions_files:
        print("[轨迹导出] 输入中未发现 actions.json", file=sys.stderr)
        return 2

    output_root = Path(args.output)
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    print(f"[轨迹导出] 发现 {len(actions_files)} 条轨迹")
    print(f"[轨迹导出] 输出目录: {output_root.resolve()}")
    try:
        summary = export_trajectories(
            actions_files,
            output_root,
            overwrite=args.overwrite,
        )
    except OSError as exc:
        print(f"[轨迹导出] 复制失败: {exc}", file=sys.stderr)
        return 1

    print(f"[轨迹导出汇总] 已复制 {summary.copied}，跳过 {summary.skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
