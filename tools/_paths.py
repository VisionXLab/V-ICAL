"""共享的项目根锚定。

迁入 tools/ 的功能脚本原先各自 `ROOT = Path(__file__).parent` 指向仓库根、
再 `sys.path.insert(0, ROOT)` 以 import `src` / `ai_backends` / `env_wrapper` 等
顶层模块。迁入子包后 `__file__` 变成 tools/xxx.py，故统一在此用
`parent.parent` 锚回仓库根，并注入 sys.path。各脚本只需 `from tools._paths import ROOT`。
"""
import sys
from pathlib import Path

# tools/_paths.py -> tools/ -> 仓库根
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
