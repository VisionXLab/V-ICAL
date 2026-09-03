"""crossval_operation 及格判断评估器。

针对一组目标游戏，扫描 crossval_operation/ 下的 plain session（不带 _full 后缀），
利用对应 _full session 推断预操作序列长度，跑 GameSession 后读 last_info["pass"]。

设计要点：
- plain.actions = 人类操作；_full.actions = 预操作 + 人类操作（前缀完全匹配）
- 预操作通过 env.step 推进物理状态，但在执行后重置 session.step_count /
  episode_reward / event_counts，保证 max_steps 边界正确、pass 判定不被预操作干扰
- 完全复用 GameSession 的 ending / pass 判定逻辑

用法：
    python vcl.py eval-crossval                     # 默认 5 个游戏
    python vcl.py eval-crossval --game ALE/Tennis-v5
    python vcl.py eval-crossval --format csv > out.csv
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from tools._paths import ROOT  # 锚回仓库根 + 注入 sys.path

from src.core.game_session import GameSession  # noqa: E402
from src.utils.score_config import compute_score  # noqa: E402

CROSSVAL_DIR = ROOT / "crossval_operation"

DEFAULT_GAMES = [
    "ALE/Asterix-v5",
    "ALE/Frostbite-v5",
    "ALE/Seaquest-v5",
    "ALE/Tennis-v5",
    "ALE/ChopperCommand-v5",
]


def find_session_pairs(target_games: set[str]) -> list[tuple[str, Path, Optional[Path]]]:
    """扫描 crossval_operation，返回 [(game, plain_dir, full_dir|None), ...]"""
    out = []
    for d in CROSSVAL_DIR.iterdir():
        if not d.is_dir() or d.name.endswith("_full"):
            continue
        aj = d / "actions.json"
        if not aj.exists():
            continue
        try:
            data = json.loads(aj.read_text("utf-8"))
        except Exception:
            continue
        game = data.get("game_name")
        if game not in target_games:
            continue
        full_dir = d.parent / (d.name + "_full")
        if not (full_dir / "actions.json").exists():
            full_dir = None
        out.append((game, d, full_dir))
    out.sort(key=lambda x: (x[0], x[1].name))
    return out


def derive_preload(plain_actions: list, full_dir: Optional[Path]) -> list:
    """从 _full session 反推预操作序列。若无 _full → []。"""
    if full_dir is None:
        return []
    full = json.loads((full_dir / "actions.json").read_text("utf-8"))
    diff = len(full["actions"]) - len(plain_actions)
    if diff <= 0:
        return []
    if full["actions"][diff:] != plain_actions:
        # 前缀不匹配 —— 数据不一致，保守不用预操作
        return []
    return full["actions"][:diff]


async def evaluate_session(plain_dir: Path, full_dir: Optional[Path]) -> dict:
    plain = json.loads((plain_dir / "actions.json").read_text("utf-8"))
    game = plain["game_name"]
    config = dict(plain["config"])
    human_actions = plain["actions"]
    preload = derive_preload(human_actions, full_dir)

    session = GameSession(f"eval_{plain_dir.name}", game, config)
    try:
        await session.initialize()

        # ① 跑预操作：env 内部状态前进，但要绕开 max_steps
        # max_steps 只该限制人类操作；预操作期间临时禁用
        saved_max = session.config.get("_max_steps")
        session.config["_max_steps"] = None
        try:
            for a in preload:
                await session.step(int(a))
                # 只有 terminated（真死亡 / 真胜利）才中断；truncated 在这里不应该发生
                if session.is_game_over and session.last_info.get("terminated"):
                    return {
                        "pass": False,
                        "reward": 0.0,
                        "score": compute_score(
                            game, 0.0, info=session.last_info,
                            step_count=0, max_steps=session.config.get("_max_steps"),
                            max_score=session.config.get("_max_score"),
                        ),
                        "ending": "preload_ended",
                        "event_counts": dict(session.event_counts),
                        "preload_steps": len(preload),
                        "human_steps": 0,
                        "note": "预操作中途已 terminated",
                    }
        finally:
            session.config["_max_steps"] = saved_max
            if "freeway" in session.game_name.lower() or "cliffwalking" in session.game_name.lower():
                session.env.env.state_reset()

        # ② 清零计数：让 max_steps / pass 判定只看人类操作
        session.step_count = 0
        session.episode_reward = 0.0
        session.event_counts = {}
        session.is_game_over = False
        if session.env is not None and not "custombreakout" in session.game_name.lower():
            session.env.reset_tracking_state()
        # ScoreState 也重置：preload 已达成的状态作为 baseline，不算人类的分
        session.score_state.reset(session.env)
        # 清掉预操作阶段写入的 ending/pass（若有）
        for k in ("ending", "pass", "truncated"):
            session.last_info.pop(k, None)

        # ③ 跑人类操作
        for a in human_actions:
            await session.step(int(a))
            if session.is_game_over:
                break

        info = session.last_info or {}
        reward = float(session.episode_reward)
        max_steps = session.config.get("_max_steps")
        score = compute_score(
            game, reward,
            info=info, step_count=session.step_count, max_steps=max_steps,
            max_score=session.config.get("_max_score"),
        )
        return {
            "pass": info.get("pass"),
            "reward": reward,
            "raw_episode_reward": info.get("raw_episode_reward"),
            "score": score,
            "ending": info.get("ending"),
            "event_counts": dict(session.event_counts),
            "milestones": dict(info.get("milestones") or {}),
            "preload_steps": len(preload),
            "human_steps": session.step_count,
        }
    finally:
        try:
            session.close()
        except Exception:
            pass


def render_table(rows: list[tuple[str, str, dict]]) -> str:
    by_game: dict[str, list[tuple[str, dict]]] = {}
    for game, name, r in rows:
        by_game.setdefault(game, []).append((name, r))

    out: list[str] = []
    for game in sorted(by_game):
        items = sorted(by_game[game], key=lambda x: x[0])
        out.append(f"\n## {game}  ({len(items)} sessions)\n")
        header = (f"{'session':<70} {'pass':>6} {'reward':>9} {'score':>9} "
                  f"{'preload':>8} {'human':>6}  ending      event_counts")
        out.append(header)
        out.append("-" * len(header))
        n_pass = 0
        for name, r in items:
            if "error" in r:
                out.append(f"{name:<70} ERROR: {r['error']}")
                continue
            ec = ",".join(f"{k}={v}" for k, v in r["event_counts"].items()) or "-"
            passed = r.get("pass")
            if passed is True:
                n_pass += 1
            score_v = r.get("score")
            score_s = f"{score_v:>9.1f}" if score_v is not None else f"{'-':>9}"
            out.append(
                f"{name:<70} {str(passed):>6} {r['reward']:>9.1f} {score_s} "
                f"{r['preload_steps']:>8} {r['human_steps']:>6}  "
                f"{(r.get('ending') or '-'):<11} {ec}"
            )
        out.append(f"\n→ {game}: pass {n_pass}/{len(items)}")
    return "\n".join(out)


def render_csv(rows: list[tuple[str, str, dict]]) -> str:
    lines = ["game,session,pass,reward,score,ending,preload_steps,human_steps,event_counts"]
    for game, name, r in sorted(rows, key=lambda x: (x[0], x[1])):
        if "error" in r:
            lines.append(f'{game},{name},ERROR,,,,,,"{r["error"]}"')
            continue
        ec = ";".join(f"{k}={v}" for k, v in r["event_counts"].items())
        score_v = r.get("score")
        score_s = f"{score_v:.2f}" if score_v is not None else ""
        lines.append(
            f"{game},{name},{r.get('pass')},{r['reward']:.2f},{score_s},"
            f"{r.get('ending') or ''},{r['preload_steps']},{r['human_steps']},\"{ec}\""
        )
    return "\n".join(lines)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", action="append", default=None,
                    help="限定游戏，可多次指定；默认评估 5 个目标游戏")
    ap.add_argument("--format", choices=["table", "csv", "json"], default="table")
    ap.add_argument("--out", default=None, help="结果写入文件而非 stdout（避免 env 噪声混入）")
    args = ap.parse_args()

    target = set(args.game) if args.game else set(DEFAULT_GAMES)
    pairs = find_session_pairs(target)
    print(f"[eval] 找到 {len(pairs)} 个 plain session", file=sys.stderr)

    rows: list[tuple[str, str, dict]] = []
    for i, (game, plain_dir, full_dir) in enumerate(pairs, 1):
        tag = "(+preload)" if full_dir else "         "
        print(f"  [{i}/{len(pairs)}] {tag} {plain_dir.name}", file=sys.stderr, flush=True)
        try:
            r = await evaluate_session(plain_dir, full_dir)
        except Exception as e:
            r = {"error": repr(e)}
        rows.append((game, plain_dir.name, r))

    if args.format == "csv":
        out = render_csv(rows)
    elif args.format == "json":
        payload = [{"game": g, "session": n, **r} for g, n, r in rows]
        out = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        out = render_table(rows)

    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"[eval] 结果写入 {args.out}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    asyncio.run(main())
