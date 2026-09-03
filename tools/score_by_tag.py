"""按标签维度（env_dynamics / info_horizon / game_type）汇总评测分数。

三个维度：
- env_dynamics: static / dynamic
- info_horizon: single_frame / few_frames / full_history
- game_type: shooter / dodge / maze / collect / control / sports（多标签，一个游戏
  可能同时属于多个 type，聚合时分别计入）

默认按游戏维度平均（每个游戏先取均值，再跨游戏平均），`--no-game-merge` 切换为
按 run 维度直接平均。

输入源（至少选一种）：
  ① eval-batch JSON 文件、含 eval_result.json 的 batch 目录，或 batch_results 下的批次名
     （最快，读取预计算结果）
  ② --batch-dir 目录（内部调 eval-batch 逐 run 重算，慢但准确）

用法：
    # 从预计算 JSON 读取（推荐：先 eval-batch --format json --out）
    python vcl.py score-by-tag eval_gemma.json eval_qwen.json

    # 直接写多个批次名：逐批独立统计，各自写入 <批次>/score_by_tag.txt
    python vcl.py score-by-tag gemma-4-31b qwen3.5-27b

    # 若明确需要把多个输入合并成一份结果，指定统一输出
    python vcl.py score-by-tag gemma-4-31b qwen3.5-27b --out combined.txt

    # 输出到目录（自动命名）
    python vcl.py score-by-tag eval_gemma.json --output-dir eval_results/ --format csv

    # 不按游戏取均值（run 维度平均）
    python vcl.py score-by-tag eval_gemma.json --no-game-merge

    # 只看某个维度
    python vcl.py score-by-tag eval_gemma.json --dim game_type

    # 从 batch_results 重算（慢，需跑 GameSession）
    python vcl.py score-by-tag --batch-dir batch_results/gemma-4-31b
"""
import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

from tools._paths import ROOT

TAGS_FILE = ROOT / "prompt" / "game_tags.json"

ALL_DIMS = ["env_dynamics", "info_horizon", "game_type"]

GAME_TYPE_ORDER = ["shooter", "dodge", "maze", "collect", "control", "sports"]

DIM_LABELS = {
    "env_dynamics": "游戏环境动静态",
    "info_horizon": "信息密集类型",
    "game_type": "游戏类型",
}


def _normalize_game_id(game: str) -> str:
    if game and game.startswith("procgen:") and not game.startswith("procgen_:"):
        return "procgen_:" + game[len("procgen:"):]
    return game


def load_tags() -> dict:
    return json.loads(TAGS_FILE.read_text("utf-8"))


def load_eval_json(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        data = json.loads(p.read_text("utf-8"))
        if isinstance(data, list):
            rows.extend(data)
        else:
            rows.append(data)
    return rows


def _resolve_eval_input(raw: str) -> Path:
    """把 JSON 路径、batch 目录或裸批次名解析为 eval_result.json。"""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path

    if path.is_file():
        return path
    if path.is_dir():
        eval_json = path / "eval_result.json"
        if eval_json.is_file():
            return eval_json
        raise FileNotFoundError(
            f"批次目录中没有 eval_result.json: {path}；"
            f"请先运行 `python vcl.py eval-batch \"{path}\"`，"
            "或改用 --batch-dir 现场重算"
        )

    raw_path = Path(raw)
    if not raw_path.is_absolute() and len(raw_path.parts) == 1:
        batch_dir = ROOT / "batch_results" / raw_path
        eval_json = batch_dir / "eval_result.json"
        if eval_json.is_file():
            return eval_json
        if batch_dir.is_dir():
            raise FileNotFoundError(
                f"批次目录中没有 eval_result.json: {batch_dir}；"
                f"请先运行 `python vcl.py eval-batch \"{raw}\"`，"
                "或改用 --batch-dir 现场重算"
            )

    raise FileNotFoundError(
        f"找不到输入: {raw}；可传入 JSON 文件、batch 目录，"
        "或 batch_results 下的批次名"
    )


def _resolve_batch_input_dir(raw: str) -> Path | None:
    """若位置参数表示 batch 目录或裸批次名，返回对应目录。"""
    raw_path = Path(raw).expanduser()
    path = raw_path if raw_path.is_absolute() else ROOT / raw_path
    if path.is_dir():
        return path

    if not raw_path.is_absolute() and len(raw_path.parts) == 1:
        batch_dir = ROOT / "batch_results" / raw_path
        if batch_dir.is_dir():
            return batch_dir
    return None


async def load_from_batch_dirs(dirs: list[Path], target_games: set) -> list[dict]:
    from tools.eval_batch import find_runs, evaluate_run, batch_label

    rows = []
    for batch_dir in dirs:
        if not batch_dir.is_absolute():
            batch_dir = ROOT / batch_dir
        if not batch_dir.is_dir():
            print(f"[score-by-tag] 跳过不存在的目录: {batch_dir}", file=sys.stderr)
            continue
        runs = find_runs(batch_dir, target_games)
        batch_name = batch_label(batch_dir)
        print(f"[score-by-tag] {batch_dir.name}: {len(runs)} runs", file=sys.stderr)
        for i, (game, label, run_dir) in enumerate(runs, 1):
            try:
                r = await evaluate_run(run_dir)
            except Exception as e:
                r = {"error": repr(e)}
            rows.append({"batch": batch_dir.name, "game": game, "config_run": label, **r})
            if i % 50 == 0 or i == len(runs):
                print(f"  [{i}/{len(runs)}]", file=sys.stderr, flush=True)
    return rows


def _get_dim_values(tags_entry: dict, dim: str) -> list[str]:
    val = tags_entry.get(dim)
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def aggregate(rows: list[dict], tags: dict, dims: list[str]) -> dict:
    """返回 {dim: {dim_value: {batch: [row, ...]}}}。"""
    result = {}
    for dim in dims:
        by_val: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if "error" in row and "pass" not in row:
                continue
            game = _normalize_game_id(row.get("game", ""))
            batch = row.get("batch", "all")
            entry = tags.get(game)
            if entry is None:
                continue
            for val in _get_dim_values(entry, dim):
                by_val[val][batch].append(row)
        result[dim] = dict(by_val)
    return result


def compute_overall(rows: list[dict], tags: dict) -> dict[str, list]:
    """计算 Overall（每个 run 只算一次，不受 game_type 多标签影响）。
    返回 {batch: [row, ...]}。
    """
    by_batch: dict[str, list] = defaultdict(list)
    for row in rows:
        if "error" in row and "pass" not in row:
            continue
        game = _normalize_game_id(row.get("game", ""))
        if game not in tags:
            continue
        batch = row.get("batch", "all")
        by_batch[batch].append(row)
    return dict(by_batch)


def _stats(rows: list[dict], game_merge: bool = True) -> dict:
    """计算统计指标。

    game_merge=True（默认）: 先按游戏分组取均值，再跨游戏平均。
    game_merge=False: 直接对所有 runs 取均值。
    score=None 表示该 run 没有得到有效分数，统计时按 0 分计入分母。
    """
    if not rows:
        return {"games": 0, "runs": 0, "pass": 0, "pass_rate": 0.0, "avg_score": None}

    if not game_merge:
        games = set()
        n_pass = 0
        scores = []
        for r in rows:
            games.add(r.get("game", ""))
            if r.get("pass") is True:
                n_pass += 1
            s = r.get("score")
            scores.append(float(s) if s is not None else 0.0)
        n_total = len(rows)
        return {
            "games": len(games),
            "runs": n_total,
            "pass": n_pass,
            "pass_rate": n_pass / n_total if n_total else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else None,
        }

    by_game: dict[str, list] = defaultdict(list)
    for r in rows:
        by_game[r.get("game", "")].append(r)

    game_scores = []
    game_pass_rates = []
    total_runs = 0
    total_pass = 0
    for game, game_rows in by_game.items():
        n = len(game_rows)
        total_runs += n
        n_pass = sum(1 for r in game_rows if r.get("pass") is True)
        total_pass += n_pass
        scores = [
            float(r["score"]) if r.get("score") is not None else 0.0
            for r in game_rows
        ]
        game_scores.append(sum(scores) / len(scores) if scores else 0)
        game_pass_rates.append(n_pass / n if n else 0)

    return {
        "games": len(by_game),
        "runs": total_runs,
        "pass": total_pass,
        "pass_rate": sum(game_pass_rates) / len(game_pass_rates) if game_pass_rates else 0.0,
        "avg_score": sum(game_scores) / len(game_scores) if game_scores else None,
    }


def _dim_order(dim: str, val: str) -> int:
    if dim == "game_type":
        try:
            return GAME_TYPE_ORDER.index(val)
        except ValueError:
            return 999
    if dim == "info_horizon":
        return {"single_frame": 0, "few_frames": 1, "full_history": 2}.get(val, 9)
    if dim == "env_dynamics":
        return {"static": 0, "dynamic": 1}.get(val, 9)
    return 0


def render_table(agg: dict, dims: list[str], overall: dict = None,
                 game_merge: bool = True) -> str:
    lines = []

    if overall is not None:
        lines.append(f"\n{'='*72}")
        lines.append(f"  Overall")
        lines.append(f"{'='*72}")
        all_rows = []
        for b_rows in overall.values():
            all_rows.extend(b_rows)
        st = _stats(all_rows, game_merge=game_merge)
        score_s = f"{st['avg_score']:.1f}" if st["avg_score"] is not None else "-"
        lines.append(f"\n  {'游戏':>5} {'runs':>6} {'pass':>6} {'rate':>8} {'avg_score':>10}")
        lines.append(f"  {'-'*40}")
        lines.append(
            f"  {st['games']:>5} {st['runs']:>6} "
            f"{st['pass']:>6} {st['pass_rate']:>7.1%} {score_s:>10}"
        )

    for dim in dims:
        by_val = agg.get(dim, {})
        if not by_val:
            continue
        label = DIM_LABELS.get(dim, dim)
        lines.append(f"\n{'='*72}")
        lines.append(f"  {label} ({dim})")
        lines.append(f"{'='*72}")

        batches = set()
        for val_data in by_val.values():
            batches.update(val_data.keys())
        batches = sorted(batches)
        multi_batch = len(batches) > 1

        lines.append(f"\n  {'[总览]':^12}")
        hdr = f"  {'标签':<20} {'游戏':>5} {'runs':>6} {'pass':>6} {'rate':>8} {'avg_score':>10}"
        lines.append(hdr)
        lines.append(f"  {'-'*60}")
        for val in sorted(by_val, key=lambda v: _dim_order(dim, v)):
            all_rows = []
            for b_rows in by_val[val].values():
                all_rows.extend(b_rows)
            st = _stats(all_rows, game_merge=game_merge)
            score_s = f"{st['avg_score']:.1f}" if st["avg_score"] is not None else "-"
            lines.append(
                f"  {val:<20} {st['games']:>5} {st['runs']:>6} "
                f"{st['pass']:>6} {st['pass_rate']:>7.1%} {score_s:>10}"
            )

        if multi_batch:
            lines.append(f"\n  {'[按 batch 细分]':^12}")
            hdr2 = f"  {'标签':<16} {'batch':<28} {'runs':>6} {'pass':>6} {'rate':>8} {'avg_score':>10}"
            lines.append(hdr2)
            lines.append(f"  {'-'*78}")
            for val in sorted(by_val, key=lambda v: _dim_order(dim, v)):
                for batch in batches:
                    b_rows = by_val[val].get(batch, [])
                    if not b_rows:
                        continue
                    st = _stats(b_rows, game_merge=game_merge)
                    score_s = f"{st['avg_score']:.1f}" if st["avg_score"] is not None else "-"
                    lines.append(
                        f"  {val:<16} {batch:<28} {st['runs']:>6} "
                        f"{st['pass']:>6} {st['pass_rate']:>7.1%} {score_s:>10}"
                    )

    return "\n".join(lines)


def render_csv(agg: dict, dims: list[str], overall: dict = None,
               game_merge: bool = True) -> str:
    lines = ["dim,tag,batch,games,runs,pass,pass_rate,avg_score"]
    if overall is not None:
        all_rows = []
        for b_rows in overall.values():
            all_rows.extend(b_rows)
        st = _stats(all_rows, game_merge=game_merge)
        score_s = f"{st['avg_score']:.2f}" if st["avg_score"] is not None else ""
        lines.append(
            f"overall,ALL,ALL,{st['games']},{st['runs']},"
            f"{st['pass']},{st['pass_rate']:.4f},{score_s}"
        )
    for dim in dims:
        by_val = agg.get(dim, {})
        for val in sorted(by_val, key=lambda v: _dim_order(dim, v)):
            batches = sorted(by_val[val].keys())
            all_rows = []
            for b_rows in by_val[val].values():
                all_rows.extend(b_rows)
            st = _stats(all_rows, game_merge=game_merge)
            score_s = f"{st['avg_score']:.2f}" if st["avg_score"] is not None else ""
            lines.append(
                f"{dim},{val},ALL,{st['games']},{st['runs']},"
                f"{st['pass']},{st['pass_rate']:.4f},{score_s}"
            )
            if len(batches) > 1:
                for batch in batches:
                    b_rows = by_val[val].get(batch, [])
                    if not b_rows:
                        continue
                    st = _stats(b_rows, game_merge=game_merge)
                    score_s = f"{st['avg_score']:.2f}" if st["avg_score"] is not None else ""
                    lines.append(
                        f"{dim},{val},{batch},{st['games']},{st['runs']},"
                        f"{st['pass']},{st['pass_rate']:.4f},{score_s}"
                    )
    return "\n".join(lines)


def _output_path(output_dir: Path, json_files: list[str], batch_dirs: list[str] | None,
                 fmt: str) -> Path:
    """根据输入文件名自动生成输出路径。"""
    ext = {"table": ".txt", "csv": ".csv", "json": ".json"}[fmt]
    if json_files:
        stem = Path(json_files[0]).stem
        if stem.startswith("eval_"):
            stem = stem[5:]
        name = f"score_by_tag_{stem}{ext}"
    elif batch_dirs:
        name = f"score_by_tag_{Path(batch_dirs[0]).name}{ext}"
    else:
        name = f"score_by_tag{ext}"
    return output_dir / name


def _render_output(
    rows: list[dict],
    tags: dict,
    dims: list[str],
    fmt: str,
    game_merge: bool,
) -> str:
    """渲染一组 rows；调用方决定多批次合并还是逐批调用。"""
    agg = aggregate(rows, tags, dims)
    overall = compute_overall(rows, tags)

    if fmt == "csv":
        return render_csv(agg, dims, overall=overall, game_merge=game_merge)
    if fmt == "json":
        entries = []
        all_ov = []
        for batch_rows in overall.values():
            all_ov.extend(batch_rows)
        st = _stats(all_ov, game_merge=game_merge)
        entries.append({"dim": "overall", "tag": "ALL", "batch": "ALL", **st})
        for dim in dims:
            by_val = agg.get(dim, {})
            for val in sorted(by_val, key=lambda v: _dim_order(dim, v)):
                all_rows = []
                for batch_rows in by_val[val].values():
                    all_rows.extend(batch_rows)
                st = _stats(all_rows, game_merge=game_merge)
                entries.append({"dim": dim, "tag": val, "batch": "ALL", **st})
                batches = sorted(by_val[val].keys())
                if len(batches) > 1:
                    for batch in batches:
                        batch_rows = by_val[val].get(batch, [])
                        if batch_rows:
                            st = _stats(batch_rows, game_merge=game_merge)
                            entries.append(
                                {"dim": dim, "tag": val, "batch": batch, **st}
                            )
        return json.dumps(entries, ensure_ascii=False, indent=2)
    return render_table(agg, dims, overall=overall, game_merge=game_merge)


async def main(argv=None):
    ap = argparse.ArgumentParser(
        description="按标签维度汇总 eval-batch 评测分数"
    )
    ap.add_argument(
        "json_files",
        nargs="*",
        metavar="INPUT",
        help=(
            "eval-batch JSON、含 eval_result.json 的 batch 目录，"
            "或 batch_results 下的批次名"
        ),
    )
    ap.add_argument("--batch-dir", action="append", default=None,
                     help="batch_results 目录（内部调 eval-batch 重算）")
    ap.add_argument("--game", action="append", default=None,
                     help="限定游戏（gym_id），可多次指定")
    ap.add_argument("--dim", action="append", default=None,
                     choices=ALL_DIMS,
                     help="只看指定维度（可多次），默认全部")
    ap.add_argument("--format", choices=["table", "csv", "json"], default="table")
    ap.add_argument("--out", default=None, help="结果写入文件（与 --output-dir 互斥）")
    ap.add_argument("--output-dir", default=None,
                     help="输出目录（自动根据输入文件名生成输出文件名）")
    ap.add_argument("--no-game-merge", action="store_true",
                     help="不按游戏取均值，直接对所有 runs 平均（默认按游戏维度平均）")
    args = ap.parse_args(argv)

    if not args.json_files and not args.batch_dir:
        ap.error("需要至少一个 JSON、batch 目录、批次名或 --batch-dir 目录")
    if args.out and args.output_dir:
        ap.error("--out 和 --output-dir 不能同时使用")

    eval_sources: list[tuple[Path, Path | None]] = []
    if args.json_files:
        try:
            eval_sources = [
                (_resolve_eval_input(raw), _resolve_batch_input_dir(raw))
                for raw in args.json_files
            ]
        except FileNotFoundError as exc:
            ap.error(str(exc))

    tags = load_tags()
    dims = args.dim if args.dim else ALL_DIMS
    game_merge = not args.no_game_merge
    target = set(args.game) if args.game else None

    # 裸批次名 / batch 目录默认逐个统计，并分别写回各自目录。
    # 显式 --out / --output-dir 时仍按旧行为生成一份合并结果。
    if (
        eval_sources
        and all(batch_dir is not None for _, batch_dir in eval_sources)
        and not args.batch_dir
        and not args.out
        and not args.output_dir
    ):
        ext = {"table": ".txt", "csv": ".csv", "json": ".json"}[args.format]
        for eval_path, batch_dir in eval_sources:
            batch_rows = load_eval_json([eval_path])
            if target:
                batch_rows = [
                    row for row in batch_rows
                    if _normalize_game_id(row.get("game", "")) in target
                ]
            out = _render_output(
                batch_rows, tags, dims, args.format, game_merge
            )
            out_path = batch_dir / f"score_by_tag{ext}"
            out_path.write_text(out, encoding="utf-8")
            print(f"[score-by-tag] 结果写入 {out_path}", file=sys.stderr)
        return

    rows = []
    if eval_sources:
        rows.extend(load_eval_json([path for path, _ in eval_sources]))
    if args.batch_dir:
        target_games = target or set()
        dirs = [Path(d) for d in args.batch_dir]
        rows.extend(await load_from_batch_dirs(dirs, target_games))

    if target:
        rows = [r for r in rows if _normalize_game_id(r.get("game", "")) in target]

    out = _render_output(rows, tags, dims, args.format, game_merge)

    out_path = None
    if args.out:
        out_path = Path(args.out)
    elif args.output_dir:
        out_path = _output_path(
            Path(args.output_dir), args.json_files, args.batch_dir, args.format
        )

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding="utf-8")
        print(f"[score-by-tag] 结果写入 {out_path}", file=sys.stderr)
    else:
        print(out)


def main_sync():
    asyncio.run(main())
