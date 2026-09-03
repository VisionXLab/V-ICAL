#!/usr/bin/env python3
"""V-ICAL 统一命令行入口。

两种用法：

  1. 子命令直达（脚本化友好）：
        python vcl.py serve --port 8888
        python vcl.py eval-batch batch_results/seed-2_0-lite --game ALE/Pong-v5
        python vcl.py rule-vs-pass

  2. 无参数 → 交互式菜单：
        python vcl.py
     列出全部功能，输入编号选择，再按提示输入该命令的参数（回车用默认）。

所有功能逻辑都在 tools/ 包里（每个子命令对应一个模块的 main()）。本文件只做
「解析子命令 → 透传剩余参数 → 调用对应 main()」的调度，按需惰性 import，
避免轻命令被 matplotlib / fastapi 等重依赖拖慢启动。
"""
import importlib
import os
import shlex
import sys
from pathlib import Path

# 强制 UTF-8：避免中文 config 名/日志在 Windows GBK 控制台或子进程转发时被打死。
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(_stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass

# 确保仓库根在 sys.path（无论从哪里调用 vcl.py）
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 子命令 -> (模块路径, 入口函数名, 是否 async, 一句话说明)
COMMANDS = {
    "serve":            ("tools.serve",            "main", False, "启动 Web UI 服务器 (webui:app)"),
    "play":             ("interactive_env",        "main", False, "人类交互玩游戏 (终端选择)"),
    "batch":            ("tools.batch",            "main", True,  "headless 批量评测 (ai_configs × 游戏 × runs)"),
    "viewer":           ("tools.viewer",           "main", False, "批结果网页 viewer"),
    "compare-viewer":   ("tools.viewer",           "main", False, "模型评测对比 viewer (Gemini Flash vs Pro)"),
    "eval-batch":       ("tools.eval_batch",       "main", True,  "batch_results 及格(pass)评估"),
    "eval-crossval":    ("tools.eval_crossval",    "main", True,  "crossval_operation 及格(pass)评估"),
    "count-fails":      ("tools.count_fails",      "main", False, "统计某 run 的不及格(提前结束)数"),
    "tokens":           ("tools.tokens",           "main", False, "token 用量单调性分析 + 折线图"),
    "rule-extract":     ("tools.rule_extract",     "main", False, "LLM 规则提取 (--scan / --rebuild-views 等)"),
    "rule-extract-run": ("tools.rule_extract_run", "main", True,  "并发驱动规则提取 (抗 502)"),
    "rule-vs-pass":     ("tools.rule_vs_pass",     "main", False, "规则主动输出 × 及格 关联分析"),
    "score-by-tag":     ("tools.score_by_tag",     "main", True,  "按标签维度 (动静态/信息密集/游戏类型) 汇总分数"),
    "nonvideo-batch":   ("tools.nonvideo_batch",   "main", True,  "无视频消融 batch 评测"),
    "fill-frames":      ("tools.frame_fill",       "main", True,  "重放 actions.json 并生成补帧视频"),
    "export-trajectories": ("tools.export_trajectories", "main", False, "按原层级导出 batch 的 actions.json"),
    "audit-trajectories": ("tools.trajectory_audit", "main", True, "Audit trajectories against demo knowledge"),
}

_ORDER = list(COMMANDS.keys())


def _dispatch(cmd: str, rest: list[str]) -> int:
    """设好 sys.argv 后惰性 import 目标模块并执行其 main()。"""
    mod_path, func_name, is_async, _desc = COMMANDS[cmd]
    if cmd == "compare-viewer":
        rest = [
            "--compare-base", "eval_results/eval_gemini-3-flash-preview.json",
            "--compare-target", "eval_results/eval_gemini-3.1-pro-preview.json",
            *rest,
        ]
    # 让目标模块的 argparse / sys.argv 解析就像被直接调用一样
    sys.argv = [f"vcl {cmd}"] + rest
    module = importlib.import_module(mod_path)
    entry = getattr(module, func_name)
    if is_async:
        import asyncio
        result = asyncio.run(entry())
    else:
        result = entry()
    return result if isinstance(result, int) else 0


def _print_menu() -> None:
    print("\nV-ICAL — 选择要运行的功能：\n")
    for i, cmd in enumerate(_ORDER, 1):
        print(f"  {i:>2}. {cmd:<17} {COMMANDS[cmd][3]}")
    print(f"   q. 退出\n")


def _split_interactive_args(arg_line: str) -> list[str]:
    """拆分交互菜单参数，同时保留 Windows 路径中的反斜杠。"""
    if sys.platform != "win32":
        return shlex.split(arg_line)

    lexer = shlex.shlex(arg_line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    lexer.escape = ""
    return list(lexer)


def _interactive() -> int:
    """无参数时的交互菜单。"""
    _print_menu()
    raw = input("输入编号 (或子命令名): ").strip()
    if raw.lower() in ("q", "quit", "exit", ""):
        return 0
    if raw.isdigit():
        idx = int(raw) - 1
        if not (0 <= idx < len(_ORDER)):
            print(f"无效编号: {raw}")
            return 2
        cmd = _ORDER[idx]
    elif raw in COMMANDS:
        cmd = raw
    else:
        print(f"未知功能: {raw}")
        return 2
    print(f"\n[{cmd}] {COMMANDS[cmd][3]}")
    arg_line = input("输入该命令的参数 (回车用默认，-h 看帮助): ").strip()
    rest = _split_interactive_args(arg_line)
    return _dispatch(cmd, rest)


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        return _interactive()

    cmd = argv[0]
    if cmd in ("-h", "--help", "help"):
        _print_menu()
        print("用法: python vcl.py <子命令> [参数...]   或   python vcl.py  (进入菜单)")
        return 0
    if cmd not in COMMANDS:
        print(f"未知子命令: {cmd}\n")
        _print_menu()
        return 2
    return _dispatch(cmd, argv[1:])


if __name__ == "__main__":
    sys.exit(main())
