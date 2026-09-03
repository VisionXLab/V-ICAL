# V-ICAL

V-ICAL（Vision-based Interactive game-playing and Cross-evaluation with Language models）是一个基于视觉的游戏交互与评测平台。它提供 Web 端人类/AI 对局、上下文示例加载、批量运行、严格动作解析、跨验证 replay，以及面向完整交互轨迹的 LLM 审计工具。

## 目录说明

```text
V-ICAL/
├── webui.py, vcl.py              # Web UI 与统一 CLI
├── src/                          # 会话、AI、API、视频与环境辅助逻辑
├── ai_backends/                  # Gemini/OpenAI-compatible 后端
├── prompt/                       # 游戏规则、动作映射与提示词模板
├── static/                       # 前端资源
├── new_gym/, flappy_bird_env_custom/  # 自定义环境
├── procgen_/                     # 本地 Procgen C++ 源码与 Python 封装
├── tools/                        # serve、batch、eval、轨迹审计等正式工具
├── demo_video_analysis/          # 轨迹审计所需的演示知识索引
└── ai_configs/                   # 从私有 Hugging Face 仓库下载
```

实验结果、运行日志、缓存、中间脚本和测试程序不属于本发行版。

## 1. 安装环境

建议使用 Python 3.11 或更新版本，并在项目目录创建独立虚拟环境：

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

项目使用仓库内的 `procgen_` 源码构建，不要额外安装 PyPI 的 `procgen` 包。

### Procgen 安装提醒

Procgen 包含 C++ 扩展，编译阶段比普通 Python 依赖更容易失败。Windows 用户需要 Visual Studio 2022 Build Tools、Windows SDK，以及 Qt 5（推荐在 conda 环境中安装 `qt-main=5`）。Linux/macOS 用户需要可用的 C++ 编译器、CMake 和 Qt 开发包。请先完成这些系统依赖，再执行：

```bash
pip install -e ./procgen_
```

不要把 C++ 编译失败误判为 V-ICAL Python 代码错误；先确认编译器、CMake、Qt 和 Python 环境来自同一个虚拟环境。安装完成后用下面的命令验证环境是否能创建并重置：

```bash
python -c "from env_wrapper import create_env; env=create_env('procgen:procgen-coinrun-v0'); obs,info=env.reset(); print(type(obs), env.action_meanings)"
```

## 2. 配置 API 与 AI 配置

复制配置模板并填写 API 信息：

```bash
copy config.example.json config.json       # Windows
cp config.example.json config.json         # Linux/macOS
```

`config.json` 支持 `openai`、`openrouter` 和 `gemini_native` 等项目内 API 模式。也可以通过环境变量提供密钥，避免把密钥写入文件。

AI 对局、批量评测和轨迹审计所需的上下文示例与动作序列位于私有 Hugging Face 仓库。登录后下载：

```bash
hf auth login
hf download <你的账号>/V-ICAL-ai-configs --repo-type dataset --local-dir ai_configs
```

下载完成后，确认 `ai_configs/<游戏>/<配置>/config.json` 存在。

## 3. 启动 Web 平台

```bash
python vcl.py serve --host 127.0.0.1 --port 8888
```

浏览器打开 <http://127.0.0.1:8888>。在平台中可以选择游戏、配置动作模式（自然语言或数字）、设置视频/上下文示例、运行 AI 对局并保存会话。

## 4. 完整评测流程

### 4.1 运行 batch

```bash
python vcl.py batch --runs 3 --workers 3 --game "ALE/Seaquest-v5"
```

`batch` 负责批量运行并保存原始轨迹、对话和视频。注意：`batch` 过程中写入的 `score` 只是运行时记录，不是经过统一 replay 判定的最终评测结果，不能直接用于论文、表格或模型比较。

### 4.2 必须用 eval-batch 重算 pass/score

```bash
python vcl.py eval-batch batch_results/<run_id> --format json --out batch_results/<run_id>/eval_result.json
```

`eval-batch` 会加载对应 AI 配置、重放预加载动作并重新执行判定逻辑。后续所有分数、Pass/Fail 统计都应以它生成的结果为准。

### 4.3 按标签汇总

```bash
python vcl.py score-by-tag batch_results/<run_id>/eval_result.json
python vcl.py score-by-tag batch_results/<run_id>/eval_result.json --dim game_type --format csv
```

需要浏览结果时可启动：

```bash
python vcl.py viewer --port 8890
```

## 5. 消融实验

消融实验应为每个条件建立独立的 AI 配置，只改变一个变量，例如 `frame_window`、`hide_reward`、`frame_source`、`video_mode`、`action_mode`、上下文示例开关或模型参数。配置放入 Hugging Face 私有仓库后，可按配置名筛选运行：

```bash
python vcl.py batch --runs 3 --workers 3 --config baseline --output batch_results/ablation_baseline
python vcl.py batch --runs 3 --workers 3 --config no_video --output batch_results/ablation_no_video
python vcl.py eval-batch batch_results/ablation_baseline --format json --out batch_results/ablation_baseline/eval_result.json
python vcl.py eval-batch batch_results/ablation_no_video --format json --out batch_results/ablation_no_video/eval_result.json
python vcl.py score-by-tag batch_results/ablation_baseline/eval_result.json
python vcl.py score-by-tag batch_results/ablation_no_video/eval_result.json
```

每个条件都必须经过 `eval-batch` 后再比较；不要比较 batch 原始 `score`。

## 6. 轨迹分析

轨迹审计使用 `audit-trajectories`，不是评测汇总命令。它会读取 batch 产生的完整交互轨迹、演示视频知识和详细游戏规则，对每条轨迹审计规则理解与策略利用：

```bash
python vcl.py export-trajectories batch_results/<run_id> --output exported_trajectories
python vcl.py audit-trajectories --batch batch_results/<run_id> --output audit_results/<run_id>
```

`audit-trajectories` 支持 `--game`、`--task`、`--run-index`、`--concurrency`、`--resume` 和 `--prepare-only` 等参数，适合分批审计和断点续跑。审计结果位于指定输出目录的 `trajectories/`、`packets/` 和 `audit_results.csv`。

## 7. 其他正式入口

```bash
python vcl.py eval-crossval crossval_operation/<run_id>
python vcl.py count-fails <run_id>
python vcl.py tokens batch_results/<run_id>
python vcl.py rule-extract --scan batch_results/<run_id>
python vcl.py rule-vs-pass --batch-dir batch_results/<run_id> --pass-json batch_results/<run_id>/eval_result.json
```

这些命令分别用于跨验证结果评估、失败运行统计、Token 使用分析、从对话提取规则，以及规则输出与 Pass 的关联分析。

## 8. 添加游戏

新增游戏时同步更新 `webui.py`、`env_wrapper.py`、`prompt/action_mappings/natural_language.json` 和 `prompt/templates/game_rules.json`；必要时更新前端默认参数、标签、奖励塑形和 cross-validation wrapper。动作名称必须是简洁英文，并与自然语言映射中的 canonical name 完全一致。

## 仓库关系

- GitHub 私有仓库 `V-ICAL`：源码、前端、Prompt、环境实现和本 README。
- Hugging Face 私有仓库 `V-ICAL-ai-configs`：大体积 AI 配置、动作序列和上下文示例。

两个仓库均不应提交 API 密钥、实验结果或运行日志。
