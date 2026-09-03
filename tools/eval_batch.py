"""batch_results 及格判断评估器。

对一个或多个 batch_results/<run_id>/ 目录，扫描每个 run 的 result.json，
通过其元数据加载对应 ai_config，跑 GameSession 后读 last_info["pass"]，
按游戏汇总及格数。

支持本地 batch、人类评测与集群结果；三者都以 result.json 中的 actions[] 和
配置元数据为准，并要求工作区存在对应的 ai_configs 配置。

与 eval_crossval.py 的关键区别——预加载动作序列的来源：
- eval_crossval：靠 `<name>_full` session 反推（diff full.actions vs plain.actions）
- eval_batch  ：batch_results 没有 _full。batch_eval 在 AI 开始前会重放一段预加载
  序列（见 batch_eval.replay_action_sequence），这段推进了 env 状态但**不写入
  result.json 的 actions[]**。本脚本从 result.json 的 config_path 或
  game_id/config_name 定位 ai_config，随后读取 `action_sequence` → sequence.json，
  把 preload 找回来，先跑 preload 再跑 AI 动作，pass 判定才正确。

error 截断规则（适用于 actions 不完整 / step 出错的未正常结束 run）：
- 有 gameover 机制的游戏 → pass=False（判 fail）
- 无 gameover 机制的游戏 → 截断，保留当前 score

preload 期间禁用 `_max_steps` 和 `end_on_life_loss`（与 batch_eval 一致）。

用法：
    python vcl.py eval-batch batch_results/20260414_110123  # 默认 20 workers，写入同目录 eval_result.json
    python vcl.py eval-batch batch_results/A batch_results/B --format csv > out.csv
    python vcl.py eval-batch batch_results/A --game ALE/Tennis-v5 --out result.txt
    # 集群数据（只有 result.json）：
    python vcl.py eval-batch cluster_results/gemma-4-31b --out gemma.txt
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from tools._paths import ROOT  # 锚回仓库根 + 注入 sys.path
from tools.runtime_init import ensure_pygame_surfarray

from src.core.game_session import GameSession  # noqa: E402
from src.utils.score_config import compute_score  # noqa: E402

ACTION_SEQ_DIR = ROOT / "action_sequences"

# 没有 gameover（死亡/失败）机制的游戏：error 截断时保留当前结果而非强制 fail
NO_GAMEOVER_GAMES = {
    "Acrobot-v1",
    "MountainCar-v0",
    "ALE/Freeway-v5",
}


def _normalize_game_id(game: str) -> str:
    """统一 game_id 格式。集群数据用 procgen:xxx，本地用 procgen_:xxx。"""
    if game and game.startswith("procgen:") and not game.startswith("procgen_:"):
        return "procgen_:" + game[len("procgen:"):]
    return game


def _load_ai_config(result: dict) -> dict | None:
    """从 run 元数据找到对应的 ai_configs/.../config.json。"""
    cfg_path = result.get("config_path")
    cfg_name = result.get("config_name", "")
    game_id = result.get("game_id") or result.get("game_name", "")

    candidates = []
    if cfg_path:
        candidates.append(ROOT / cfg_path.replace("\\", "/") / "config.json")

    if game_id and cfg_name:
        game_safe = game_id.replace("/", "_").replace(":", "_")
        candidates.append(ROOT / "ai_configs" / game_safe / cfg_name / "config.json")
        # procgen 特殊：集群 procgen:xxx → 本地目录 procgen_xxx（单下划线）
        if "procgen" in game_id:
            alt_safe = game_id.replace("procgen_:", "procgen_").replace("procgen:", "procgen_").replace("/", "_")
            if alt_safe != game_safe:
                candidates.append(ROOT / "ai_configs" / alt_safe / cfg_name / "config.json")

    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text("utf-8"))
            except Exception:
                continue
    return None


def _load_run_ai_config(run_dir: Path) -> dict | None:
    """根据 run/result.json 的元数据加载对应 ai_config。"""
    result_f = run_dir / "result.json"
    if not result_f.is_file():
        return None
    try:
        return _load_ai_config(json.loads(result_f.read_text("utf-8")))
    except Exception:
        return None


def load_preload(run_dir: Path) -> list:
    """从 run 的 ai_config 找到 action_sequence 引用并加载；无则 []。"""
    ai_cfg = _load_run_ai_config(run_dir)
    if ai_cfg is None:
        return []

    seq_ref = ai_cfg.get("action_sequence") or {}
    game_safe = seq_ref.get("game_safe", "")
    seq_name = seq_ref.get("name", "")
    if not game_safe or not seq_name:
        return []
    seq_file = ACTION_SEQ_DIR / game_safe / seq_name / "sequence.json"
    if not seq_file.exists():
        return []
    try:
        seq_data = json.loads(seq_file.read_text("utf-8"))
    except Exception:
        return []
    return list(seq_data.get("actions", []) or [])


def _build_config_from_ai_config(result: dict) -> dict:
    """从集群 result.json 的 config_path 读 ai_config，生成 GameSession config。"""
    from tools.batch import convert_game_settings

    ai_cfg = _load_ai_config(result)
    if ai_cfg is None:
        return {}

    game_id = _normalize_game_id(result.get("game_id", ""))
    return convert_game_settings(game_id, ai_cfg.get("game_settings", {}))


def _reset_after_preload(session: GameSession) -> None:
    """隔离 preload 与 AI 阶段，并用干净的计分快照替换旧 ``last_info``。

    仅重置 ``ScoreState`` 不够：如果 AI 在第一轮就失败、没有执行任何动作，
    ``session.last_info`` 仍指向 preload 最后一步，其中的 raw_episode_reward、
    battlezone_first_hit_step 等字段会泄漏到最终 score。重建 last_info 还能为
    零动作轨迹提供 raw_episode_reward=0，避免 ratio_capped 返回 None。
    """
    session.step_count = 0
    session.episode_reward = 0.0
    session.event_counts = {}
    session.is_game_over = False
    if session.env is not None and "custombreakout" not in session.game_name.lower():
        session.env.reset_tracking_state()

    # preload 阶段达成的里程碑 / 已开门作为 baseline，不算 AI 分。
    session.score_state.reset(session.env)
    session.last_info = dict(session.score_state.to_info_dict())


async def evaluate_run(run_dir: Path) -> dict:
    # 本地 batch、人类评测与集群格式统一从 result.json + ai_config 构建。
    result_data = json.loads((run_dir / "result.json").read_text("utf-8"))
    game = _normalize_game_id(result_data["game_id"])
    ai_actions = result_data.get("actions", [])
    config = _build_config_from_ai_config(result_data)

    preload = load_preload(run_dir)

    session = GameSession(f"eval_{run_dir.parent.name}", game, config)
    try:
        await session.initialize()

        # ① 跑预加载序列：禁用 max_steps + end_on_life_loss（与 batch_eval 一致）
        saved_max = session.config.get("_max_steps")
        session.config["_max_steps"] = None
        saved_eoll = None
        if session.env is not None and hasattr(session.env, "end_on_life_loss"):
            saved_eoll = session.env.end_on_life_loss
            session.env.end_on_life_loss = False
        try:
            for a in preload:
                if session.is_game_over:
                    break
                await session.step(int(a))
        finally:
            session.config["_max_steps"] = saved_max
            if saved_eoll is not None:
                session.env.end_on_life_loss = saved_eoll
            if "freeway" in session.game_name.lower() or "cliffwalking" in session.game_name.lower():
                session.env.env.state_reset()

        # ② 清零计数与 last_info：max_steps / pass / score 只看 AI 动作
        _reset_after_preload(session)

        # ③ 跑 AI 动作（step 出错时截断而非整体失败）
        step_error = None
        for a in ai_actions:
            try:
                await session.step(int(a))
            except Exception as e:
                step_error = repr(e)
                break
            if session.is_game_over:
                break

        info = session.last_info or {}
        reward = float(session.episode_reward)

        # 未正常结束：actions 不完整 或 step 出错
        if not session.is_game_over:
            if game not in NO_GAMEOVER_GAMES:
                info["pass"] = False
                info.setdefault("ending", "error")
            else:
                info.setdefault("ending", "truncated")

        max_steps = session.config.get("_max_steps")
        score = compute_score(
            game, reward,
            info=info, step_count=session.step_count, max_steps=max_steps,
            max_score=session.config.get("_max_score"),
        )
        result = {
            "pass": info.get("pass"),
            "reward": reward,
            "raw_episode_reward": info.get("raw_episode_reward"),
            "score": score,
            "ending": info.get("ending"),
            "event_counts": dict(session.event_counts),
            "milestones": dict(info.get("milestones") or {}),
            "opened_doors_count": info.get("opened_doors_count"),
            "total_doors": info.get("total_doors"),
            "shortest_path": info.get("shortest_path"),
            "preload_steps": len(preload),
            "ai_steps": session.step_count,
        }
        if step_error:
            result["step_error"] = step_error
        return result
    finally:
        try:
            session.close()
        except Exception:
            pass


def batch_label(batch_dir: Path) -> str:
    """batch 标题：有 config.json 则附上 ai.model，否则只用目录名（时间戳）。"""
    cfg_f = batch_dir / "config.json"
    if cfg_f.exists():
        try:
            model = (json.loads(cfg_f.read_text("utf-8")).get("ai") or {}).get("model")
            if model:
                return f"{batch_dir.name}  [model: {model}]"
        except Exception:
            pass
    return batch_dir.name


def find_runs(batch_dir: Path, target_games: set) -> list:
    """返回 [(game, label, run_dir), ...]，label = <game>/<config>/<run>。"""
    out = []
    for game_dir in sorted(batch_dir.iterdir()):
        if not game_dir.is_dir() or game_dir.name == "token_analysis":
            continue
        for cfg_dir in sorted(game_dir.iterdir()):
            if not cfg_dir.is_dir() or cfg_dir.name == "token_analysis":
                continue
            for run_dir in sorted(cfg_dir.iterdir()):
                # 只认 run_<数字> 目录，排除备份 / 杂目录（如 run_0_apierror_bak）
                if not re.fullmatch(r"run_\d+", run_dir.name):
                    continue
                if not run_dir.is_dir():
                    continue
                result_file = run_dir / "result.json"
                if not result_file.exists():
                    continue
                try:
                    game = json.loads(result_file.read_text("utf-8")).get("game_id")
                    game = _normalize_game_id(game)
                except Exception:
                    continue
                if target_games and game not in target_games:
                    continue
                label = f"{game_dir.name}/{cfg_dir.name}/{run_dir.name}"
                out.append((game, label, run_dir))
    return out


def render_table(batch_name: str, rows: list) -> list:
    by_game = {}
    for game, label, r in rows:
        by_game.setdefault(game, []).append((label, r))
    lines = [f"\n{'='*80}\n# {batch_name}\n{'='*80}"]
    for game in sorted(by_game):
        items = sorted(by_game[game])
        n_pass = sum(1 for _, r in items if r.get("pass") is True)
        n_none = sum(1 for _, r in items if r.get("pass") is None and "error" not in r)
        extra = f"  ({n_none} 未判定)" if n_none else ""
        lines.append(f"\n## {game}  —  pass {n_pass}/{len(items)}{extra}")
        for label, r in items:
            if "error" in r:
                lines.append(f"  {label:<58} ERROR: {r['error']}")
                continue
            ec = ",".join(f"{k}={v}" for k, v in r["event_counts"].items()) or "-"
            score_v = r.get("score")
            score_s = f"{score_v:>9.1f}" if score_v is not None else f"{'-':>9}"
            lines.append(
                f"  {label:<58} pass={str(r.get('pass')):<5} "
                f"reward={r['reward']:>9.1f}  score={score_s}  "
                f"preload={r['preload_steps']:>3} ai_steps={r['ai_steps']:>3}  "
                f"ending={(r.get('ending') or '-'):<10} {ec}"
            )
    return lines


def render_csv(all_rows: list) -> str:
    lines = ["batch,game,config_run,pass,reward,score,ending,preload_steps,ai_steps,event_counts"]
    for batch_name, game, label, r in all_rows:
        if "error" in r:
            lines.append(f'{batch_name},{game},{label},ERROR,,,,,,"{r["error"]}"')
            continue
        ec = ";".join(f"{k}={v}" for k, v in r["event_counts"].items())
        score_v = r.get("score")
        score_s = f"{score_v:.2f}" if score_v is not None else ""
        lines.append(
            f"{batch_name},{game},{label},{r.get('pass')},{r['reward']:.2f},{score_s},"
            f"{r.get('ending') or ''},{r['preload_steps']},{r['ai_steps']},\"{ec}\""
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_dirs", nargs="+",
                    help="一个或多个 batch_results/<run_id> 目录，也可只写 run_id")
    ap.add_argument("--game", action="append", default=None,
                    help="限定游戏（gym_id），可多次指定")
    ap.add_argument("--format", choices=["table", "csv", "json"], default="json")
    ap.add_argument("--out", default=None,
                    help="结果写入文件而非 stdout（避免 env 噪声混入）")
    ap.add_argument("--workers", type=int, default=20,
                    help="并发重放评分数量（默认 20）")
    return ap


def _resolve_batch_dir(raw: str) -> Path:
    """解析 batch 目录；单独的 run_id 自动到 batch_results/ 下查找。"""
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path

    rooted = ROOT / path
    if rooted.is_dir():
        return rooted

    if len(path.parts) == 1:
        batch_result_path = ROOT / "batch_results" / path
        if batch_result_path.is_dir():
            return batch_result_path

    return rooted


async def evaluate_runs_concurrently(runs: list, workers: int) -> list:
    """并发评估 runs；限制并发、隔离单项错误并保持输入顺序。"""
    if workers <= 0:
        raise ValueError("workers must be greater than 0")
    semaphore = asyncio.Semaphore(workers)

    async def evaluate_one(game: str, label: str, run_dir: Path):
        async with semaphore:
            try:
                result = await evaluate_run(run_dir)
            except Exception as exc:
                result = {"error": repr(exc)}
        return game, label, result

    return await asyncio.gather(
        *(evaluate_one(game, label, run_dir) for game, label, run_dir in runs)
    )


def render_json_rows(batch_name: str, rows: list) -> str:
    return json.dumps(
        [{"batch": batch_name, "game": game, "config_run": label, **result}
         for game, label, result in rows],
        ensure_ascii=False,
        indent=2,
    )


async def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.workers <= 0:
        ap.error("--workers must be greater than 0")

    ensure_pygame_surfarray()

    target_games = set(args.game) if args.game else set()
    all_rows = []          # (batch_name, game, label, result)
    table_lines = []

    for raw in args.batch_dirs:
        batch_dir = _resolve_batch_dir(raw)
        if not batch_dir.is_dir():
            print(f"[eval_batch] 跳过不存在的目录: {batch_dir}", file=sys.stderr)
            continue
        runs = find_runs(batch_dir, target_games)
        print(f"[eval_batch] {batch_dir.name}: {len(runs)} runs", file=sys.stderr)
        rows = await evaluate_runs_concurrently(runs, args.workers)
        for i, (game, label, result) in enumerate(rows, 1):
            all_rows.append((batch_dir.name, game, label, result))
            print(f"  [{i}/{len(runs)}] {label}", file=sys.stderr, flush=True)
        if args.format == "table":
            table_lines += render_table(batch_label(batch_dir), rows)

        if args.format == "json" and not args.out:
            output_path = batch_dir / "eval_result.json"
            output_path.write_text(
                render_json_rows(batch_dir.name, rows), encoding="utf-8"
            )
            print(f"[eval_batch] 结果写入 {output_path}", file=sys.stderr)

    if args.format == "json" and not args.out:
        return

    if args.format == "csv":
        out = render_csv(all_rows)
    elif args.format == "json":
        out = json.dumps(
            [{"batch": b, "game": g, "config_run": l, **r}
             for b, g, l, r in all_rows], ensure_ascii=False, indent=2
        )
    else:
        out = "\n".join(table_lines)

    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"[eval_batch] 结果写入 {args.out}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    asyncio.run(main())
