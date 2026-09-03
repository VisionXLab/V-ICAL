# V-ICAL

V-ICAL is a vision-based platform for interactive game playing and evaluation with multimodal language models. It provides a browser-based interface for human and AI play, visual demonstration loading, strict action parsing, batch execution, replay-based evaluation, ablation studies, cross-validation, and LLM-based trajectory auditing.

## Repository layout

```text
V-ICAL/
├── webui.py, vcl.py                   # Web application and unified CLI
├── src/                               # Sessions, AI loop, API, video, and environment helpers
├── ai_backends/                       # Gemini and OpenAI-compatible backends
├── prompt/                            # Prompt templates, rules, mappings, and game tags
├── static/                            # Browser frontend
├── new_gym/                           # Custom Gymnasium environments
├── flappy_bird_env_custom/            # Custom Flappy Bird environment
├── procgen_/                          # Repository-local Procgen C++ source and Python wrapper
├── tools/                             # Serve, batch, evaluation, and analysis commands
├── demo_video_analysis/               # Demonstration knowledge used by trajectory auditing
└── ai_configs/                        # Downloaded separately from Hugging Face
```

Experiment outputs, intermediate artifacts, logs, caches, development notes, and test programs are intentionally excluded.

## 1. Environment setup

Use Python 3.11 or a newer compatible version in an isolated environment:

```bash
python -m venv .venv

# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

V-ICAL uses the repository-local `procgen_` implementation. Do not install the unrelated PyPI `procgen` package.

### Procgen build requirements

Procgen includes a native C++ extension, so its installation is more fragile than ordinary Python packages.

On Windows, install:

- Visual Studio 2022 Build Tools with the **Desktop development with C++** workload.
- A matching Windows SDK.
- Qt 5. The recommended conda installation is:

```bash
conda install -n vcl -c conda-forge qt-main=5
```

Linux and macOS users need a working C++ compiler, CMake, and Qt development packages. Build the repository-local extension after installing the system dependencies:

```bash
pip install -e ./procgen_
```

If this step fails, first verify that the compiler, CMake, Qt, and Python interpreter belong to the same active environment. A native build failure is not necessarily a V-ICAL Python error.

Validate the completed installation with:

```bash
python -c "from env_wrapper import create_env; env=create_env('procgen:procgen-coinrun-v0'); obs,info=env.reset(); print(type(obs), env.action_meanings)"
```

## 2. API and AI configuration

Create the local API configuration:

```bash
# Windows
copy config.example.json config.json

# Linux or macOS
cp config.example.json config.json
```

Edit `config.json` to select the API mode, endpoint, and model. V-ICAL supports the project-defined `openai`, `openrouter`, and `gemini_native` modes. Keep real API keys out of version control; environment variables are recommended when supported by the selected backend.

### Required: download `ai_configs` from Hugging Face

The GitHub repository does not contain the visual demonstrations and per-task AI configurations. Before starting the platform, running evaluations, or auditing trajectories, authenticate with Hugging Face and explicitly download the private dataset into the repository's `ai_configs/` directory:

```bash
hf auth login
hf download Mingqian-233/V-ICAL-ai-configs --repo-type dataset --local-dir ai_configs
```

After downloading, verify that paths such as `ai_configs/Acrobot-v1/s0/config.json` exist.

## 3. Start the Web platform

```bash
python vcl.py serve --host 127.0.0.1 --port 8888
```

Open <http://127.0.0.1:8888>. The interface supports game selection, human or AI play, natural-language or numeric actions, context demonstrations, video settings, saved configurations, and session export.

## 4. Evaluation workflow

### 4.1 Run a batch

```bash
python vcl.py batch --runs 3 --workers 3 --game "ALE/Seaquest-v5"
```

`batch` runs the selected configurations and stores the raw conversations, actions, frames, videos, and runtime metadata.

> **Important:** the `score` shown or saved directly by `batch` is not the authoritative evaluation result. Do not use it in papers, tables, or model comparisons. Every batch must be replayed with `eval-batch`.

### 4.2 Recompute the authoritative results

```bash
python vcl.py eval-batch batch_results/<run_id> \
  --format json \
  --out batch_results/<run_id>/eval_result.json
```

`eval-batch` reloads the corresponding AI configuration, replays any configured preload action sequence, executes the recorded model actions, and applies the current unified pass and score logic. All downstream statistics must use this output.

### 4.3 Aggregate by game tags

```bash
python vcl.py score-by-tag batch_results/<run_id>/eval_result.json
python vcl.py score-by-tag batch_results/<run_id>/eval_result.json \
  --dim game_type \
  --format csv
```

To browse saved runs in a Web interface:

```bash
python vcl.py viewer --port 8890
```

## 5. Ablation studies

Create one AI configuration for each condition and change only one experimental variable at a time. Common ablation dimensions include:

- `frame_window`
- `hide_reward`
- `frame_source`
- `video_mode`
- `action_mode`
- demonstration/context availability
- prompt or game-rule content
- model, temperature, or reasoning settings

Run each configuration into a separate result directory:

```bash
python vcl.py batch --runs 3 --workers 3 \
  --config baseline \
  --output batch_results/ablation_baseline

python vcl.py batch --runs 3 --workers 3 \
  --config no_video \
  --output batch_results/ablation_no_video
```

The dedicated non-video command is also available:

```bash
python vcl.py nonvideo-batch --runs 3 --workers 3
```

Evaluate every condition separately:

```bash
python vcl.py eval-batch batch_results/ablation_baseline \
  --format json --out batch_results/ablation_baseline/eval_result.json

python vcl.py eval-batch batch_results/ablation_no_video \
  --format json --out batch_results/ablation_no_video/eval_result.json
```

Compare only the outputs produced by `eval-batch`; never compare the raw `batch` scores.

## 6. Trajectory analysis

Trajectory analysis is performed by `audit-trajectories`. It is separate from the `batch → eval-batch → score-by-tag` evaluation workflow.

The auditor examines the full chronological interaction, the demonstration-derived knowledge, detailed game rules, model requests and responses, parsed actions, resulting frames, and terminal outcome. It classifies rule understanding and demonstrated-strategy utilization for each trajectory.

Run an audit with:

```bash
python vcl.py audit-trajectories \
  --batch batch_results/<run_id> \
  --output audit_results/<run_id>
```

Useful options include `--game`, `--task`, `--run-index`, `--limit`, `--concurrency`, `--resume`, and `--prepare-only`. Results are written to `trajectories/`, `packets/`, summary files, and `audit_results.csv` under the selected output directory.

To extract recorded action trajectories while preserving their original hierarchy:

```bash
python vcl.py export-trajectories \
  batch_results/<run_id> \
  --output exported_trajectories
```

## 7. Additional commands

```bash
# Evaluate cross-validation replays
python vcl.py eval-crossval crossval_operation/<run_id>

# Count prematurely terminated runs
python vcl.py count-fails <run_id>

# Analyze token usage
python vcl.py tokens batch_results/<run_id>

# Extract rules from model conversations
python vcl.py rule-extract --scan batch_results/<run_id>

# Relate rule output to replay-evaluated pass results
python vcl.py rule-vs-pass \
  --batch-dir batch_results/<run_id> \
  --pass-json batch_results/<run_id>/eval_result.json
```

## 8. Adding a game

At minimum, update these files in order:

1. `webui.py`: register the game ID, description, and category.
2. `env_wrapper.py`: define its semantic action meanings.
3. `env_wrapper.py`: define `ACTION_SIMPLE_IDS` when the game has more than six actions.
4. `prompt/action_mappings/natural_language.json`: add canonical action names and accepted variants.
5. `prompt/templates/game_rules.json`: describe the objective, mechanics, scoring, and terminal conditions.

Update frontend defaults, tags, reward shaping, BFS support, or cross-validation wrappers only when the game requires them. Canonical action names must be concise English labels and must exactly match the meanings defined in `env_wrapper.py`.

## Repository separation

- Private GitHub repository: [`Mingqian-233/V-ICAL`](https://github.com/Mingqian-233/V-ICAL) contains source code, environments, prompts, frontend assets, and this guide.
- Private Hugging Face dataset: [`Mingqian-233/V-ICAL-ai-configs`](https://huggingface.co/datasets/Mingqian-233/V-ICAL-ai-configs) contains the larger AI configurations and visual demonstrations.

Neither repository should contain API keys, provider credentials, evaluation outputs, or runtime logs.
