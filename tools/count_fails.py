"""
统计 batch_results 里某个 run 的"不及格"数量。

定义：total_steps < max_steps → 不及格（提前结束）
      total_steps >= max_steps → 及格（跑满）
      status != "completed" → error（API/环境出错，单独计数）

用法:
    python vcl.py count-fails <run_id> [--root batch_results] [--verbose]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from tools._paths import ROOT  # 锚回仓库根（count-fails 用 ROOT/<root>/<run_id> 定位）


def load_run(run_dir: Path) -> dict | None:
    """读取单次 run 的 result.json + actions.json，返回合并 dict 或 None。"""
    result_file = run_dir / "result.json"
    if not result_file.exists():
        return None
    try:
        result = json.loads(result_file.read_text(encoding="utf-8"))
    except Exception:
        return None

    max_steps = None
    actions_file = run_dir / "actions.json"
    if actions_file.exists():
        try:
            actions = json.loads(actions_file.read_text(encoding="utf-8"))
            max_steps = actions.get("config", {}).get("max_steps")
        except Exception:
            pass

    return {
        "game_id": result.get("game_id", "?"),
        "config_name": result.get("config_name", "?"),
        "run_index": result.get("run_index", 0),
        "status": result.get("status", "?"),
        "total_steps": result.get("total_steps", 0),
        "max_steps": max_steps,
        "game_over_reason": result.get("game_over_reason"),
        "total_reward": result.get("total_reward", 0.0),
        "run_dir": run_dir,
    }


def classify(run: dict) -> str:
    """返回 'pass' / 'fail' / 'error' / 'no_max_steps'。"""
    if run["status"] != "completed":
        return "error"
    if run["max_steps"] is None:
        return "no_max_steps"
    if run["total_steps"] < run["max_steps"]:
        return "fail"
    return "pass"


def main():
    parser = argparse.ArgumentParser(description="统计 batch run 的不及格数量")
    parser.add_argument("run_id", help="batch_results 下的 run_id，如 20260420_172629")
    parser.add_argument("--root", default="batch_results", help="batch_results 根目录")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="打印每个不及格 run 的明细")
    args = parser.parse_args()

    base = ROOT / args.root / args.run_id
    if not base.exists():
        print(f"[Error] 目录不存在: {base}")
        sys.exit(1)

    runs = []
    for result_file in base.rglob("run_*/result.json"):
        run = load_run(result_file.parent)
        if run is not None:
            runs.append(run)

    if not runs:
        print(f"[Info] {base} 下没有找到任何 result.json")
        sys.exit(0)

    # 分类
    by_class = defaultdict(list)
    by_game = defaultdict(lambda: {"pass": 0, "fail": 0, "error": 0, "no_max_steps": 0})
    for run in runs:
        cls = classify(run)
        by_class[cls].append(run)
        by_game[run["game_id"]][cls] += 1

    # 汇总
    total = len(runs)
    n_pass = len(by_class["pass"])
    n_fail = len(by_class["fail"])
    n_err = len(by_class["error"])
    n_nomax = len(by_class["no_max_steps"])

    print(f"=== Run {args.run_id} ===")
    print(f"Path: {base}")
    print(f"Total: {total} | Pass: {n_pass} | Fail: {n_fail} | Error: {n_err}"
          + (f" | NoMaxSteps: {n_nomax}" if n_nomax else ""))
    print()

    # 按游戏
    print("Per-game:")
    for game in sorted(by_game):
        c = by_game[game]
        parts = [f"pass={c['pass']}", f"fail={c['fail']}"]
        if c["error"]:
            parts.append(f"err={c['error']}")
        if c["no_max_steps"]:
            parts.append(f"no_max={c['no_max_steps']}")
        print(f"  {game:30s} {' '.join(parts)}")

    # 明细
    if args.verbose and by_class["fail"]:
        print("\nFailed runs (total_steps < max_steps):")
        for r in sorted(by_class["fail"], key=lambda x: (x["game_id"], x["config_name"], x["run_index"])):
            reason = r["game_over_reason"] or "?"
            print(f"  {r['game_id']}/{r['config_name']}/run_{r['run_index']}: "
                  f"{r['total_steps']}/{r['max_steps']} steps ({reason}, reward={r['total_reward']})")

    if args.verbose and by_class["error"]:
        print("\nError runs (status != completed):")
        for r in sorted(by_class["error"], key=lambda x: (x["game_id"], x["config_name"], x["run_index"])):
            print(f"  {r['game_id']}/{r['config_name']}/run_{r['run_index']}: status={r['status']}")


if __name__ == "__main__":
    main()
