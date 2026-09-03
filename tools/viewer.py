"""Batch Results Web Viewer.

独立的 FastAPI 服务，用于浏览 batch_results/ 下的批量运行结果。

用法：
    python vcl.py viewer --port 8890
    浏览器打开 http://localhost:8890
"""
from __future__ import annotations

import argparse
import html
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from tools._paths import ROOT as PROJECT_ROOT  # 锚回仓库根 + 注入 sys.path
BATCH_RESULTS_DIR = PROJECT_ROOT / "batch_results"

COMPARE_DATA: dict[str, Any] | None = None
COMPARE_BASE_LABEL = "base"
COMPARE_TARGET_LABEL = "target"
COMPARE_INITIAL_THRESHOLD = 5.0
BATCH_TASK_INDEX: dict[tuple[str, str], Path] | None = None

app = FastAPI(title="Batch Results Viewer")


# ---------------------------------------------------------------------------
# 数据解析
# ---------------------------------------------------------------------------

def list_runs() -> list[dict[str, Any]]:
    if not BATCH_RESULTS_DIR.exists():
        return []
    runs: list[dict[str, Any]] = []
    for entry in BATCH_RESULTS_DIR.iterdir():
        if not entry.is_dir():
            continue
        summary_path = entry / "summary.json"
        info: dict[str, Any] = {
            "run_id": entry.name,
            "mtime": entry.stat().st_mtime,
            "created_at": None,
            "total_tasks": None,
            "completed": None,
            "errors": None,
            "has_summary": summary_path.exists(),
        }
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                info["created_at"] = data.get("created_at")
                info["total_tasks"] = data.get("total_tasks")
                info["completed"] = data.get("completed")
                info["errors"] = data.get("errors")
            except (json.JSONDecodeError, OSError):
                info["has_summary"] = False
        runs.append(info)
    runs.sort(key=lambda x: x["mtime"], reverse=True)
    return runs


def load_run_summary(run_id: str) -> dict[str, Any]:
    path = BATCH_RESULTS_DIR / run_id / "summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"summary.json not found: {run_id}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_batch_eval_index(
    run_id: str,
    batch_aliases: set[str] | None = None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], str | None]:
    """读取批次的标准 eval 结果；支持目录改名后保留的原始批次 ID。"""
    path = BATCH_RESULTS_DIR / run_id / "eval_result.json"
    if not path.exists():
        return {}, "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, "invalid"
    if not isinstance(data, list):
        return {}, "invalid"

    accepted_batch_ids = {run_id}
    if batch_aliases:
        accepted_batch_ids.update(str(alias) for alias in batch_aliases if alias)

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in data:
        if not isinstance(row, dict) or str(row.get("batch")) not in accepted_batch_ids:
            continue
        game = row.get("game")
        config_run = row.get("config_run")
        if game and config_run:
            index[(str(game), str(config_run))] = row
    return index, None


def game_id_to_slug(game_id: str) -> str:
    return game_id.replace("/", "_").replace(":", "_")


def load_eval_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"eval JSON must be an array: {path}")
    rows = [row for row in data if isinstance(row, dict)]
    if len(rows) != len(data):
        raise ValueError(f"eval JSON contains non-object rows: {path}")
    return rows


def eval_row_key(row: dict[str, Any]) -> tuple[str, str] | None:
    game = row.get("game")
    config_run = row.get("config_run")
    if not game or not config_run:
        return None
    return str(game), str(config_run)


def _eval_label(path: Path, rows: list[dict[str, Any]]) -> str:
    stem = path.stem
    return stem.removeprefix("eval_")


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def batch_game_slug(game_id: str) -> str:
    return game_id_to_slug(game_id)


def _config_run_from_summary_row(row: dict[str, Any]) -> str | None:
    game_id = row.get("game_id")
    config_name = row.get("config_name")
    if not game_id or not config_name:
        return None
    run_idx = row.get("run_index", 0)
    return f"{batch_game_slug(str(game_id))}/{config_name}/run_{run_idx}"


def build_batch_task_index() -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    if not BATCH_RESULTS_DIR.exists():
        return index
    for run_dir in BATCH_RESULTS_DIR.iterdir():
        summary_path = run_dir / "summary.json"
        if not summary_path.exists() or not summary_path.is_file():
            continue
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        run_id = str(summary.get("run_id") or run_dir.name)
        for row in summary.get("results", []):
            if not isinstance(row, dict):
                continue
            config_run = _config_run_from_summary_row(row)
            if config_run is None:
                continue
            index[(run_id, config_run)] = run_dir / config_run
    return index


def _batch_task_dir(row: dict[str, Any]) -> Path | None:
    batch = row.get("batch")
    config_run = row.get("config_run")
    if not batch or not config_run:
        return None
    if BATCH_TASK_INDEX is not None:
        task_dir = BATCH_TASK_INDEX.get((str(batch), str(config_run)))
        if task_dir is not None:
            return task_dir
    return BATCH_RESULTS_DIR / str(batch) / str(config_run)


def _compare_file_url(row: dict[str, Any], filename: str) -> str | None:
    task_dir = _batch_task_dir(row)
    if task_dir is None:
        return None
    target = task_dir / filename
    if not target.exists() or not target.is_file():
        return None
    rel = target.relative_to(BATCH_RESULTS_DIR).as_posix()
    run_id, subpath = rel.split("/", 1)
    return f"/file/{quote(run_id)}/{quote(subpath, safe='/')}"


def _compare_side(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch": row.get("batch"),
        "score": row.get("score"),
        "pass": row.get("pass"),
        "ending": row.get("ending"),
        "reward": row.get("reward"),
        "raw_episode_reward": row.get("raw_episode_reward"),
        "preload_steps": row.get("preload_steps"),
        "ai_steps": row.get("ai_steps"),
        "chat_url": _compare_file_url(row, "chat.html"),
        "video_url": _compare_file_url(row, "video_nl.mp4"),
        "row": row,
    }


def build_eval_compare(base_path: Path, target_path: Path) -> dict[str, Any]:
    base_rows = load_eval_rows(base_path)
    target_rows = load_eval_rows(target_path)
    base_index = {key: row for row in base_rows if (key := eval_row_key(row)) is not None}
    target_index = {key: row for row in target_rows if (key := eval_row_key(row)) is not None}
    paired_keys = sorted(base_index.keys() & target_index.keys())

    cases = []
    for game, config_run in paired_keys:
        base_row = base_index[(game, config_run)]
        target_row = target_index[(game, config_run)]
        base_score = _float_or_none(base_row.get("score"))
        target_score = _float_or_none(target_row.get("score"))
        delta_score = None
        if base_score is not None and target_score is not None:
            delta_score = target_score - base_score
        cases.append({
            "game": game,
            "config_run": config_run,
            "delta_score": delta_score,
            "base": _compare_side(base_row),
            "target": _compare_side(target_row),
        })

    cases.sort(key=lambda c: (c["delta_score"] is None, c["delta_score"] or 0, c["game"], c["config_run"]))
    return {
        "base_label": _eval_label(base_path, base_rows),
        "target_label": _eval_label(target_path, target_rows),
        "base_path": str(base_path),
        "target_path": str(target_path),
        "base_total": len(base_rows),
        "target_total": len(target_rows),
        "pair_count": len(cases),
        "base_only_count": len(base_index.keys() - target_index.keys()),
        "target_only_count": len(target_index.keys() - base_index.keys()),
        "cases": cases,
    }


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def find_task_dir(run_id: str, game_slug: str, config_name: str, run_dir: str) -> Path:
    root = (BATCH_RESULTS_DIR / run_id).resolve()
    target = (root / game_slug / config_name / run_dir).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path (traversal)")
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail=f"Task not found: {target}")
    return target


def parse_caption_nl(path: Path) -> list[tuple[int, str]]:
    """解析 caption_nl.txt，返回 [(step_idx, action_text)]。

    格式：
        Start State
        Step 1: Action: Up
    """
    if not path.exists():
        return []
    entries: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("start"):
                entries.append((0, line))
                continue
            if line.startswith("Step "):
                try:
                    head, rest = line.split(":", 1)
                    step_idx = int(head.replace("Step", "").strip())
                    action = rest.strip()
                    low = action.lower()
                    if low.startswith("action:"):
                        action = action[len("action:"):].strip()
                    elif low.startswith("action "):
                        action = action[len("action "):].strip()
                    entries.append((step_idx, action))
                    continue
                except (ValueError, IndexError):
                    pass
            entries.append((len(entries), line))
    return entries


def list_step_frames(task_dir: Path) -> list[Path]:
    for sub in ("subbed_nl", "subbed", "frames"):
        d = task_dir / sub
        if d.exists():
            frames = sorted(d.glob("step_*.png"))
            if frames:
                return frames
    return []


def aggregate_by_game(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[float]] = {}
    for r in results:
        reward = r.get("total_reward")
        if reward is None:
            continue
        groups.setdefault(r["game_id"], []).append(float(reward))
    out = []
    for game_id, rws in sorted(groups.items()):
        n = len(rws)
        passed = sum(1 for x in rws if x >= 0)
        out.append({
            "game_id": game_id,
            "n": n,
            "pass_count": passed,
            "pass_rate": passed / n if n else 0.0,
            "avg": sum(rws) / n if n else 0.0,
            "min": min(rws) if rws else 0.0,
            "max": max(rws) if rws else 0.0,
        })
    return out


# ---------------------------------------------------------------------------
# HTML 模板（公共部分）
# ---------------------------------------------------------------------------

BASE_CSS = """
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: #fafbfc;
    color: #24292e;
    line-height: 1.5;
    overflow: hidden;
}
a { color: #0366d6; text-decoration: none; }
a:hover { text-decoration: underline; }
.pass { color: #28a745; font-weight: 600; }
.fail { color: #d73a49; font-weight: 600; }
.mono { font-family: "JetBrains Mono", Consolas, monospace; font-size: 12px; }

/* ===== Home / Run list (simple centered list) ===== */
.home {
    max-width: 900px; margin: 0 auto; padding: 32px 24px;
    height: 100vh; overflow: auto;
}
.home h1 { margin: 0 0 6px 0; font-size: 22px; }
.home .sub { color: #586069; font-size: 13px; margin-bottom: 20px; }
.run-card {
    display: block; background: #fff; border: 1px solid #e1e4e8;
    border-radius: 6px; padding: 14px 18px; margin-bottom: 10px;
    text-decoration: none !important; color: #24292e;
    transition: transform .1s, border-color .1s;
}
.run-card:hover { border-color: #0366d6; transform: translateX(4px); }
.run-card .rid { font-weight: 600; font-size: 15px; color: #0366d6; }
.run-card .sub-info { font-size: 12px; color: #586069; margin-top: 4px; }

/* ===== Workspace (run detail) ===== */
.ws { display: flex; height: 100vh; width: 100vw; }
.ws-side {
    width: 280px; min-width: 240px; max-width: 400px;
    background: #f6f8fa; border-right: 1px solid #e1e4e8;
    display: flex; flex-direction: column; overflow: hidden;
}
.ws-side-header {
    padding: 10px 12px; border-bottom: 1px solid #e1e4e8;
    background: #fff;
}
.ws-side-header .title { font-weight: 600; font-size: 13px; }
.ws-side-header .title a { color: #586069; font-size: 11px; margin-right: 6px; }
.ws-side-header .count { font-size: 11px; color: #586069; margin-top: 2px; }
.ws-side-list { flex: 1; overflow-y: auto; padding: 6px 0; }
.game-group { margin-bottom: 4px; }
.game-group-header {
    padding: 6px 12px; font-size: 11px; font-weight: 700;
    color: #586069; text-transform: uppercase; letter-spacing: 0.5px;
    cursor: pointer; user-select: none; display: flex; justify-content: space-between;
}
.game-group-header:hover { background: #eaecef; }
.game-group-header .count-tag { font-weight: normal; color: #959da5; }
.task-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 12px 6px 22px; font-size: 12px; cursor: pointer;
    border-left: 3px solid transparent;
}
.task-item:hover { background: #eaecef; }
.task-item.active {
    background: #0366d6; color: #fff !important; border-left-color: #ffeb3b;
}
.task-item.active .reward-badge { color: #fff !important; }
.task-item .tname {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    flex: 1; margin-right: 6px;
}
.reward-badge { font-size: 11px; font-weight: 600; white-space: nowrap; }
.score-badge {
    background: #fff5cc; padding: 0 4px; border-radius: 3px;
    border: 1px solid #ffe066;
}
.task-item.active .score-badge {
    background: rgba(255,255,255,0.15); border-color: rgba(255,255,255,0.4);
}
.game-group.collapsed .task-item { display: none; }

/* ===== Main pane ===== */
.ws-main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.ws-topbar {
    display: flex; align-items: center; gap: 12px; padding: 8px 14px;
    background: #fff; border-bottom: 1px solid #e1e4e8; flex-wrap: wrap;
}
.ws-topbar .title { font-size: 13px; font-weight: 600; }
.ws-topbar .sub { font-size: 12px; color: #586069; }
.ws-topbar .big-r { font-size: 18px; font-weight: 700; }
.ws-topbar .spacer { flex: 1; }
.ws-topbar button, .ws-topbar .btn {
    background: #f6f8fa; color: #24292e !important; border: 1px solid #d1d5da;
    padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 12px;
    text-decoration: none; display: inline-block;
}
.ws-topbar button:hover, .ws-topbar .btn:hover { background: #e1e4e8; }
.ws-topbar button.primary, .ws-topbar .btn.primary {
    background: #0366d6; color: #fff !important; border-color: #0366d6;
}
.ws-topbar button.primary:hover { background: #0256b9; }
.kbd {
    display: inline-block; font-family: monospace; font-size: 11px;
    background: #fafbfc; border: 1px solid #d1d5da; border-bottom-width: 2px;
    border-radius: 3px; padding: 1px 5px; color: #586069;
}

/* ===== Main content ===== */
.ws-body { flex: 1; position: relative; overflow: hidden; background: #fff; }
.ws-body iframe {
    width: 100%; height: 100%; border: none; background: #fff;
}
.ws-body .empty {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    color: #959da5; font-size: 14px;
}

/* ===== Extras panel (collapsible drawer) ===== */
.extras-drawer {
    position: absolute; top: 0; right: 0; bottom: 0;
    width: 420px; background: #fff; border-left: 1px solid #e1e4e8;
    transform: translateX(100%); transition: transform .2s;
    overflow-y: auto; padding: 14px; z-index: 5;
    box-shadow: -4px 0 12px rgba(0,0,0,.06);
}
.extras-drawer.open { transform: translateX(0); }
.extras-drawer h3 { margin: 12px 0 6px; font-size: 13px; color: #586069; text-transform: uppercase; letter-spacing: .5px; }
.extras-drawer img, .extras-drawer video {
    max-width: 100%; display: block; border-radius: 4px; background: #000;
}
.extras-drawer .meta-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; }
.extras-drawer .meta-row .k { color: #586069; }
.frame-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; }
.frame-card { border: 1px solid #eaecef; border-radius: 4px; overflow: hidden; }
.frame-card img { width: 100%; cursor: pointer; }
.frame-card .caption { padding: 4px 6px; font-size: 10px; border-top: 1px solid #eaecef; }
.frame-card .caption .step { color: #586069; margin-right: 4px; }
.frame-card .caption .action { font-weight: 600; }

/* Lightbox for frames */
.lightbox {
    display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.9); z-index: 1001;
    align-items: center; justify-content: center; cursor: pointer;
}
.lightbox.active { display: flex; }
.lightbox img { max-width: 92vw; max-height: 92vh; }

/* ===== Compare workspace ===== */
.compare-page { height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
.compare-header {
    background: #fff; border-bottom: 1px solid #e1e4e8;
    padding: 12px 16px; display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
}
.compare-header h1 { margin: 0; font-size: 18px; }
.compare-header .meta { color: #586069; font-size: 12px; }
.compare-controls {
    background: #f6f8fa; border-bottom: 1px solid #e1e4e8;
    padding: 10px 16px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
    font-size: 12px;
}
.compare-controls input, .compare-controls select {
    border: 1px solid #d1d5da; border-radius: 4px; padding: 4px 7px; font-size: 12px;
    background: #fff;
}
.compare-body { flex: 1; display: grid; grid-template-columns: 360px 1fr; overflow: hidden; }
.compare-list { border-right: 1px solid #e1e4e8; overflow: auto; background: #fff; }
.compare-item {
    border-bottom: 1px solid #eaecef; padding: 8px 12px; cursor: pointer;
    display: grid; grid-template-columns: 1fr auto; gap: 6px;
}
.compare-item:hover { background: #f6f8fa; }
.compare-item.active { background: #eaf5ff; border-left: 4px solid #0366d6; padding-left: 8px; }
.compare-item .game { font-weight: 600; font-size: 12px; }
.compare-item .run { color: #586069; font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.compare-item .scores { font-size: 11px; color: #586069; text-align: right; }
.compare-item .delta { font-size: 14px; font-weight: 700; }
.compare-detail { overflow: hidden; background: #fafbfc; display: flex; flex-direction: column; min-width: 0; }
.compare-empty { color: #959da5; margin: 40px auto; text-align: center; }
.compare-summary { background: #fff; border-bottom: 1px solid #e1e4e8; padding: 8px 12px; }
.compare-summary-title { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
.compare-summary-title h2 { margin: 0; font-size: 15px; }
.compare-summary-title .run { color: #586069; }
.compare-columns { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
.compare-card { background: #fff; border: 1px solid #e1e4e8; border-radius: 6px; padding: 8px; }
.compare-card h3 { margin: 0 0 6px 0; font-size: 12px; display: flex; justify-content: space-between; gap: 8px; }
.compare-metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; font-size: 11px; }
.compare-metric { background: #f6f8fa; border: 1px solid #eaecef; border-radius: 4px; padding: 3px 5px; }
.compare-metric .k { color: #586069; display: block; }
.compare-links { display: flex; gap: 6px; flex-wrap: wrap; }
.compare-links a {
    border: 1px solid #d1d5da; border-radius: 4px; padding: 2px 6px;
    background: #f6f8fa; font-size: 11px; color: #24292e;
}
.compare-frames { flex: 1; min-height: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 0; background: #d1d5da; }
.compare-frame-pane { min-width: 0; min-height: 0; display: flex; flex-direction: column; background: #fff; }
.compare-frame-pane + .compare-frame-pane { border-left: 1px solid #d1d5da; }
.compare-frame-head { padding: 5px 8px; background: #f6f8fa; border-bottom: 1px solid #e1e4e8; font-size: 12px; font-weight: 600; display: flex; justify-content: space-between; gap: 8px; }
.compare-frame-pane iframe { flex: 1; width: 100%; border: 0; background: #fff; }
.compare-frame-missing { flex: 1; display: flex; align-items: center; justify-content: center; color: #959da5; font-size: 13px; }
"""

LIGHTBOX_HTML = """
<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <img id="lightboxImg" src="" alt="">
</div>
<script>
function openLightbox(src) {
    document.getElementById('lightboxImg').src = src;
    document.getElementById('lightbox').classList.add('active');
}
function closeLightbox() {
    document.getElementById('lightbox').classList.remove('active');
}
</script>
"""


def page(title: str, body: str, extra_scripts: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{BASE_CSS}</style>
</head>
<body>
{body}
{LIGHTBOX_HTML}
{extra_scripts}
</body>
</html>
"""


def fmt_duration(sec: float | None) -> str:
    if sec is None:
        return "-"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def fmt_pct(x: float | None) -> str:
    return f"{x*100:.1f}%" if x is not None else "-"


# ---------------------------------------------------------------------------
# 路由：文件服务
# ---------------------------------------------------------------------------

@app.get("/file/{run_id}/{subpath:path}")
def serve_file(run_id: str, subpath: str):
    root = (BATCH_RESULTS_DIR / run_id).resolve()
    if not root.exists():
        raise HTTPException(404, "run not found")
    target = (root / subpath).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise HTTPException(400, "invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "file not found")
    mime, _ = mimetypes.guess_type(str(target))
    if mime is None:
        mime = "application/octet-stream"
    if target.suffix.lower() == ".html":
        mime = "text/html; charset=utf-8"
    return FileResponse(target, media_type=mime)


# ---------------------------------------------------------------------------
# 路由：API
# ---------------------------------------------------------------------------

@app.get("/api/run/{run_id}/summary")
def api_run_summary(run_id: str):
    return JSONResponse(load_run_summary(run_id))


@app.get("/api/compare")
def api_compare():
    if COMPARE_DATA is None:
        raise HTTPException(404, "compare mode disabled")
    return JSONResponse({
        **COMPARE_DATA,
        "initial_threshold": COMPARE_INITIAL_THRESHOLD,
    })


# ---------------------------------------------------------------------------
# 路由：首页 - Run 列表
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    runs = list_runs()
    cards = []
    if COMPARE_DATA is not None:
        cards.append(
            '<a class="run-card" href="/compare">'
            '<div class="rid">Model Compare</div>'
            f'<div class="sub-info">{html.escape(COMPARE_DATA["base_label"])} vs '
            f'{html.escape(COMPARE_DATA["target_label"])} · '
            f'{COMPARE_DATA["pair_count"]} paired cases · '
            f'initial threshold {COMPARE_INITIAL_THRESHOLD:g}</div></a>'
        )
    for r in runs:
        rid = r["run_id"]
        if not r["has_summary"]:
            sub = "<span style='color:#999'>无 summary.json</span>"
        else:
            sub = (
                f"Created: <span class='mono'>{html.escape(r['created_at'] or '-')}</span>"
                f" · Total: {r['total_tasks'] or '-'}"
                f" · Completed: {r['completed'] or '-'}"
                f" · Errors: {r['errors'] if r['errors'] is not None else '-'}"
            )
        cards.append(
            f'<a class="run-card" href="/run/{quote(rid)}">'
            f'<div class="rid">{html.escape(rid)}</div>'
            f'<div class="sub-info">{sub}</div></a>'
        )
    body = f"""
<div class="home">
  <h1>📊 Batch Results Viewer</h1>
  <div class="sub">{len(runs)} 个 run · 选一个进入工作台</div>
  {''.join(cards) if cards else '<div style="color:#999">(无数据)</div>'}
</div>
"""
    return HTMLResponse(page("Batch Results Viewer", body))


@app.get("/compare", response_class=HTMLResponse)
def compare_detail():
    if COMPARE_DATA is None:
        raise HTTPException(404, "compare mode disabled")
    payload = {
        **COMPARE_DATA,
        "initial_threshold": COMPARE_INITIAL_THRESHOLD,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    body = f"""
<div class="compare-page">
  <header class="compare-header">
    <h1>Model Compare</h1>
    <div class="meta">
      <a href="/">← 首页</a>
      <span class="mono">{html.escape(COMPARE_DATA["base_label"])} → {html.escape(COMPARE_DATA["target_label"])}</span>
      · {COMPARE_DATA["pair_count"]} paired
      · base-only {COMPARE_DATA["base_only_count"]}
      · target-only {COMPARE_DATA["target_only_count"]}
    </div>
  </header>
  <section class="compare-controls">
    <label>threshold <input id="thresholdInput" type="number" step="1" value="{COMPARE_INITIAL_THRESHOLD:g}" style="width:80px"></label>
    <label>game <select id="gameFilter"><option value="">All games</option></select></label>
    <label><input id="failOnly" type="checkbox"> target fail / base pass</label>
    <label>sort
      <select id="sortSelect">
        <option value="delta_asc">delta ↑</option>
        <option value="delta_desc">delta ↓</option>
        <option value="game">game</option>
        <option value="base_desc">base score ↓</option>
        <option value="target_asc">target score ↑</option>
      </select>
    </label>
    <span id="resultCount" class="meta"></span>
  </section>
  <main class="compare-body">
    <section class="compare-list" id="caseList"></section>
    <section class="compare-detail" id="caseDetail">
      <div class="compare-empty">选择左侧 case 查看详情</div>
    </section>
  </main>
</div>
"""
    js = f"""<script>
const COMPARE = {payload_json};
let filteredCases = [];
let currentCase = null;

function num(v) {{
    if (v === null || v === undefined || Number.isNaN(Number(v))) return null;
    return Number(v);
}}
function fmt(v, digits=2) {{
    const n = num(v);
    return n === null ? '-' : n.toFixed(digits);
}}
function esc(s) {{
    return String(s ?? '').replace(/[&<>\"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));
}}
function passClass(v) {{ return v === true ? 'pass' : (v === false ? 'fail' : ''); }}
function passText(v) {{ return v === true ? 'pass' : (v === false ? 'fail' : '-'); }}
function caseMatches(c) {{
    const threshold = Number(document.getElementById('thresholdInput').value || 0);
    const baseScore = num(c.base.score);
    const targetScore = num(c.target.score);
    if (baseScore === null || targetScore === null) return false;
    if (targetScore > baseScore + threshold) return false;
    const game = document.getElementById('gameFilter').value;
    if (game && c.game !== game) return false;
    if (document.getElementById('failOnly').checked && !(c.base.pass === true && c.target.pass === false)) return false;
    return true;
}}
function sortCases(items) {{
    const mode = document.getElementById('sortSelect').value;
    const copy = [...items];
    copy.sort((a, b) => {{
        if (mode === 'delta_desc') return (num(b.delta_score) ?? -Infinity) - (num(a.delta_score) ?? -Infinity);
        if (mode === 'game') return (a.game + a.config_run).localeCompare(b.game + b.config_run);
        if (mode === 'base_desc') return (num(b.base.score) ?? -Infinity) - (num(a.base.score) ?? -Infinity);
        if (mode === 'target_asc') return (num(a.target.score) ?? Infinity) - (num(b.target.score) ?? Infinity);
        return (num(a.delta_score) ?? Infinity) - (num(b.delta_score) ?? Infinity);
    }});
    return copy;
}}
function renderList() {{
    filteredCases = sortCases(COMPARE.cases.filter(caseMatches));
    document.getElementById('resultCount').textContent = `${{filteredCases.length}} / ${{COMPARE.cases.length}} cases`;
    const list = document.getElementById('caseList');
    if (!filteredCases.length) {{
        list.innerHTML = '<div class="compare-empty">没有符合当前阈值的 case</div>';
        document.getElementById('caseDetail').innerHTML = '<div class="compare-empty">调大 threshold 或放宽过滤条件</div>';
        currentCase = null;
        return;
    }}
    list.innerHTML = filteredCases.map((c, i) => `
        <div class="compare-item ${{currentCase === c ? 'active' : ''}}" onclick="selectCompareCase(${{i}})">
          <div>
            <div class="game">${{esc(c.game)}}</div>
            <div class="run mono">${{esc(c.config_run)}}</div>
          </div>
          <div class="scores">
            <div class="delta ${{num(c.delta_score) <= 0 ? 'fail' : 'pass'}}">Δ ${{fmt(c.delta_score)}}</div>
            <div>${{fmt(c.base.score)}} → ${{fmt(c.target.score)}}</div>
          </div>
        </div>`).join('');
    if (!currentCase || !filteredCases.includes(currentCase)) selectCompareCase(0);
}}
function metricItems(side) {{
    return [
        ['score', fmt(side.score)],
        ['pass', `<span class="${{passClass(side.pass)}}">${{passText(side.pass)}}</span>`],
        ['ending', esc(side.ending ?? '-')],
        ['reward', fmt(side.reward)],
        ['raw', fmt(side.raw_episode_reward)],
        ['preload', esc(side.preload_steps ?? '-')],
        ['ai_steps', esc(side.ai_steps ?? '-')],
        ['batch', esc(side.batch ?? '-')],
    ].map(([k, v]) => `<div class="compare-metric"><span class="k">${{k}}</span><span>${{v}}</span></div>`).join('');
}}
function linksHtml(side) {{
    const links = [];
    if (side.chat_url) links.push(`<a href="${{side.chat_url}}" target="_blank">open chat</a>`);
    if (side.video_url) links.push(`<a href="${{side.video_url}}" target="_blank">video</a>`);
    return links.length ? `<span class="compare-links">${{links.join('')}}</span>` : '<span class="meta">无 chat/video</span>';
}}
function frameHtml(side, label) {{
    const body = side.chat_url
        ? `<iframe src="${{side.chat_url}}"></iframe>`
        : '<div class="compare-frame-missing">这个 case 没有 chat.html</div>';
    return `<section class="compare-frame-pane">
        <div class="compare-frame-head"><span>${{esc(label)}}</span>${{linksHtml(side)}}</div>
        ${{body}}
    </section>`;
}}
function renderDetail(c) {{
    document.getElementById('caseDetail').innerHTML = `
      <div class="compare-summary">
        <div class="compare-summary-title">
          <h2>${{esc(c.game)}}</h2>
          <span class="run mono">${{esc(c.config_run)}}</span>
          <span class="delta ${{num(c.delta_score) <= 0 ? 'fail' : 'pass'}}">Δ score ${{fmt(c.delta_score)}}</span>
        </div>
        <div class="compare-columns">
          <div class="compare-card">
            <h3><span>${{esc(COMPARE.base_label)}}</span><span>${{fmt(c.base.score)}}</span></h3>
            <div class="compare-metrics">${{metricItems(c.base)}}</div>
          </div>
          <div class="compare-card">
            <h3><span>${{esc(COMPARE.target_label)}}</span><span>${{fmt(c.target.score)}}</span></h3>
            <div class="compare-metrics">${{metricItems(c.target)}}</div>
          </div>
        </div>
      </div>
      <div class="compare-frames">
        ${{frameHtml(c.base, COMPARE.base_label)}}
        ${{frameHtml(c.target, COMPARE.target_label)}}
      </div>`;
}}
function selectCompareCase(i) {{
    currentCase = filteredCases[i];
    document.querySelectorAll('.compare-item').forEach(el => el.classList.remove('active'));
    const active = document.querySelectorAll('.compare-item')[i];
    if (active) active.classList.add('active');
    if (currentCase) renderDetail(currentCase);
}}
function initCompare() {{
    const games = [...new Set(COMPARE.cases.map(c => c.game))].sort();
    const select = document.getElementById('gameFilter');
    games.forEach(g => {{
        const opt = document.createElement('option');
        opt.value = g; opt.textContent = g;
        select.appendChild(opt);
    }});
    ['thresholdInput', 'gameFilter', 'failOnly', 'sortSelect'].forEach(id => {{
        document.getElementById(id).addEventListener('input', renderList);
        document.getElementById(id).addEventListener('change', renderList);
    }});
    renderList();
}}
initCompare();
</script>"""
    return HTMLResponse(page("Model Compare", body, js))


# ---------------------------------------------------------------------------
# 路由：Run 详情
# ---------------------------------------------------------------------------

@app.get("/run/{run_id}", response_class=HTMLResponse)
def run_detail(run_id: str):
    """Workspace 视图：左栏 task 树 + 右栏 chat.html iframe。"""
    summary = load_run_summary(run_id)
    results = summary.get("results", [])
    original_run_id = summary.get("run_id")
    batch_aliases = {str(original_run_id)} if original_run_id else None
    eval_index, eval_error = load_batch_eval_index(run_id, batch_aliases)

    # 为 JS 准备 task 列表（带 chat URL、reward 等）
    tasks_data = []
    for i, r in enumerate(results):
        game_id = r["game_id"]
        config = r["config_name"]
        run_idx = r.get("run_index", 0)
        game_slug = game_id_to_slug(game_id)
        rel_base = f"{game_slug}/{quote(config, safe='')}/run_{run_idx}"
        task_dir = BATCH_RESULTS_DIR / run_id / game_slug / config / f"run_{run_idx}"
        config_run = f"{batch_game_slug(game_id)}/{config}/run_{run_idx}"
        eval_row = eval_index.get((game_id, config_run))
        score_v = eval_row.get("score") if eval_row is not None else None
        pass_v = eval_row.get("pass") if eval_row is not None else None
        tasks_data.append({
            "idx": i,
            "game_id": game_id,
            "config_name": config,
            "run_index": run_idx,
            "game_slug": game_slug,
            "pass": pass_v,
            "score": score_v,
            "evaluated": eval_row is not None,
            "steps": r.get("total_steps"),
            "duration": r.get("duration_seconds"),
            "status": r.get("status"),
            "chat_url": f"/file/{quote(run_id)}/{rel_base}/chat.html"
                       if (task_dir / "chat.html").exists() else None,
            "trajectory_url": f"/file/{quote(run_id)}/{rel_base}/trajectory.png"
                             if (task_dir / "trajectory.png").exists() else None,
            "video_url": f"/file/{quote(run_id)}/{rel_base}/video_nl.mp4"
                        if (task_dir / "video_nl.mp4").exists() else None,
            "extras_url": f"/api/run/{quote(run_id)}/task/{i}/extras",
        })

    # 按游戏分组（保持原顺序）
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(tasks_data):
        groups.setdefault(t["game_id"], []).append(i)

    # 渲染侧栏
    side_parts: list[str] = []
    for game_id, idxs in groups.items():
        items = []
        for i in idxs:
            t = tasks_data[i]
            passed = t["pass"]
            score = t["score"]
            badges = []
            if passed is not None:
                cls = "pass" if passed else "fail"
                badges.append(
                    f'<span class="reward-badge {cls}" title="pass">'
                    f'pass:{str(bool(passed)).lower()}</span>'
                )
            elif not t["evaluated"]:
                badges.append(
                    '<span class="reward-badge" title="尚未运行 eval-batch" '
                    'style="color:#8a6d3b;background:#fff8dc">pass:未评估</span>'
                )
            if score is not None:
                cls = "pass" if score >= 0 else "fail"
                badges.append(
                    f'<span class="reward-badge score-badge {cls}" title="score">'
                    f's:{score:.1f}</span>'
                )
            badge_html = (' <span style="display:inline-flex;gap:4px">'
                          + ''.join(badges) + '</span>') if badges else ''
            items.append(
                f'<div class="task-item" data-idx="{i}" onclick="selectTask({i})">'
                f'<span class="tname" title="{html.escape(t["config_name"])}">{html.escape(t["config_name"])}</span>'
                f'{badge_html}</div>'
            )
        side_parts.append(
            '<div class="game-group">'
            f'<div class="game-group-header" onclick="toggleGroup(this)">'
            f'<span>{html.escape(game_id)}</span>'
            f'<span class="count-tag">{len(idxs)}</span>'
            f'</div>'
            f'{"".join(items)}'
            '</div>'
        )

    tasks_json = json.dumps(tasks_data, ensure_ascii=False)
    eval_command = (
        f"python vcl.py eval-batch batch_results/{run_id} --format json "
        f"--out batch_results/{run_id}/eval_result.json"
    )
    unevaluated_count = sum(1 for t in tasks_data if not t["evaluated"])
    eval_notice = ""
    if eval_error == "invalid":
        eval_notice = (
            '<div style="padding:8px 12px;background:#fff3cd;color:#664d03;'
            'border-bottom:1px solid #ffecb5">'
            '评估文件无效，当前分数标记为“未评估”。请重新运行：'
            f'<code>{html.escape(eval_command)}</code></div>'
        )
    elif eval_error == "missing" or unevaluated_count:
        detail = "未找到评估文件" if eval_error == "missing" else f"有 {unevaluated_count} 个任务缺少评估结果"
        eval_notice = (
            '<div style="padding:8px 12px;background:#fff3cd;color:#664d03;'
            'border-bottom:1px solid #ffecb5">'
            f'{detail}，缺失分数标记为“未评估”。请运行：'
            f'<code>{html.escape(eval_command)}</code></div>'
        )

    body = f"""
{eval_notice}
<div class="ws">
  <aside class="ws-side">
    <div class="ws-side-header">
      <div class="title"><a href="/">← 首页</a></div>
      <div class="title" style="margin-top:4px">{html.escape(run_id)}</div>
      <div class="count">{len(tasks_data)} tasks · {len(groups)} games</div>
    </div>
    <div class="ws-side-list" id="taskList">
      {''.join(side_parts)}
    </div>
  </aside>
  <main class="ws-main">
    <div class="ws-topbar">
      <div>
        <div class="title" id="tbTitle">(选择左侧 task)</div>
        <div class="sub" id="tbSub"></div>
      </div>
      <div>
        <div class="big-r" id="tbPass" title="pass"></div>
        <div class="big-r" id="tbScore" title="score" style="font-size:13px;color:#b08800"></div>
      </div>
      <div class="spacer"></div>
      <button onclick="prevTask()" title="Prev (←/k)">◀ Prev</button>
      <button onclick="nextTask()" title="Next (→/j)">Next ▶</button>
      <button onclick="toggleExtras()" title="Trajectory / Video / Frames (e)">📎 Extras</button>
      <a class="btn" id="tbOpen" href="#" target="_blank" title="在新标签页打开 chat.html">↗ New tab</a>
      <span style="font-size:11px;color:#959da5">
        <span class="kbd">←</span>/<span class="kbd">→</span>
        <span class="kbd">j</span>/<span class="kbd">k</span>
        <span class="kbd">e</span>
      </span>
    </div>
    <div class="ws-body">
      <iframe id="chatFrame" src="about:blank"></iframe>
      <div class="empty" id="emptyHint">← 从左侧选择一个 task</div>
      <aside class="extras-drawer" id="extrasDrawer">
        <div id="extrasContent" style="font-size:12px;color:#586069">(no task)</div>
      </aside>
    </div>
  </main>
</div>
"""

    js = f"""<script>
const TASKS = {tasks_json};
let currentIdx = -1;
let extrasOpen = false;

function fmtDur(s) {{
    if (s == null) return '-';
    s = Math.floor(s);
    const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = s%60;
    return h ? `${{h}}:${{String(m).padStart(2,'0')}}:${{String(sec).padStart(2,'0')}}`
             : `${{m}}:${{String(sec).padStart(2,'0')}}`;
}}

function selectTask(i) {{
    if (i < 0 || i >= TASKS.length) return;
    const t = TASKS[i];
    currentIdx = i;

    // highlight sidebar
    document.querySelectorAll('.task-item').forEach(el => el.classList.remove('active'));
    const active = document.querySelector(`.task-item[data-idx="${{i}}"]`);
    if (active) {{
        active.classList.add('active');
        // expand its group
        let g = active.closest('.game-group');
        if (g) g.classList.remove('collapsed');
        active.scrollIntoView({{block: 'nearest'}});
    }}

    // topbar
    document.getElementById('tbTitle').textContent = t.game_id;
    document.getElementById('tbSub').textContent =
        `${{t.config_name}} · ${{t.steps ?? '-'}} steps · ${{fmtDur(t.duration)}} · ${{t.status ?? '-'}}`;
    const pEl = document.getElementById('tbPass');
    if (!t.evaluated) {{ pEl.textContent = 'pass:未评估'; pEl.className = 'big-r'; }}
    else if (t.pass === true) {{ pEl.textContent = 'pass:true'; pEl.className = 'big-r pass'; }}
    else if (t.pass === false) {{ pEl.textContent = 'pass:false'; pEl.className = 'big-r fail'; }}
    else {{ pEl.textContent = 'pass:-'; pEl.className = 'big-r'; }}
    const s = t.score;
    const sEl = document.getElementById('tbScore');
    if (!t.evaluated) {{ sEl.textContent = 'score: 未评估'; sEl.style.color = '#8a6d3b'; }}
    else if (s == null) {{ sEl.textContent = 'score: -'; sEl.style.color = '#959da5'; }}
    else {{ sEl.textContent = 'score: ' + s.toFixed(2); sEl.style.color = '#b08800'; }}

    // iframe
    const frame = document.getElementById('chatFrame');
    const hint = document.getElementById('emptyHint');
    if (t.chat_url) {{
        frame.src = t.chat_url;
        frame.style.display = '';
        hint.style.display = 'none';
        document.getElementById('tbOpen').href = t.chat_url;
        document.getElementById('tbOpen').style.pointerEvents = '';
        document.getElementById('tbOpen').style.opacity = '';
    }} else {{
        frame.src = 'about:blank';
        frame.style.display = 'none';
        hint.style.display = '';
        hint.textContent = '(这个 task 没有 chat.html)';
        document.getElementById('tbOpen').href = '#';
        document.getElementById('tbOpen').style.pointerEvents = 'none';
        document.getElementById('tbOpen').style.opacity = '0.5';
    }}

    // extras drawer — 延迟加载
    if (extrasOpen) loadExtras();

    // URL sync
    history.replaceState(null, '', `?task=${{i}}`);
    document.title = `${{t.game_id}} | ${{t.config_name}}`;
}}

function prevTask() {{ if (currentIdx > 0) selectTask(currentIdx - 1); }}
function nextTask() {{ if (currentIdx < TASKS.length - 1) selectTask(currentIdx + 1); }}

function toggleGroup(header) {{
    header.parentElement.classList.toggle('collapsed');
}}

function toggleExtras() {{
    const d = document.getElementById('extrasDrawer');
    extrasOpen = !extrasOpen;
    d.classList.toggle('open', extrasOpen);
    if (extrasOpen) loadExtras();
}}

function loadExtras() {{
    if (currentIdx < 0) return;
    const t = TASKS[currentIdx];
    const c = document.getElementById('extrasContent');
    let html = '';
    if (t.trajectory_url) {{
        html += '<h3>Trajectory</h3>';
        html += `<img src="${{t.trajectory_url}}" onclick="openLightbox('${{t.trajectory_url}}')">`;
    }}
    if (t.video_url) {{
        html += '<h3>Video (NL)</h3>';
        html += `<video controls preload="metadata"><source src="${{t.video_url}}" type="video/mp4"></video>`;
    }}
    html += '<h3>Frames</h3><div id="extrasFrames" style="color:#959da5">loading…</div>';
    c.innerHTML = html;

    // 懒加载 frame 列表
    fetch(t.extras_url).then(r => r.json()).then(data => {{
        const fc = document.getElementById('extrasFrames');
        if (!data.frames || !data.frames.length) {{ fc.textContent = '(无帧)'; return; }}
        const grid = document.createElement('div');
        grid.className = 'frame-grid';
        data.frames.forEach(fr => {{
            const card = document.createElement('div');
            card.className = 'frame-card';
            card.innerHTML =
                `<img src="${{fr.url}}" loading="lazy" onclick="openLightbox('${{fr.url}}')">` +
                `<div class="caption"><span class="step">${{fr.step}}</span>` +
                `<span class="action">${{fr.action.replace(/</g,'&lt;')}}</span></div>`;
            grid.appendChild(card);
        }});
        fc.innerHTML = ''; fc.appendChild(grid);
    }}).catch(e => {{ document.getElementById('extrasFrames').textContent = 'load failed'; }});
}}

// 键盘快捷键
document.addEventListener('keydown', e => {{
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft' || e.key === 'k') {{ prevTask(); e.preventDefault(); }}
    else if (e.key === 'ArrowRight' || e.key === 'j') {{ nextTask(); e.preventDefault(); }}
    else if (e.key === 'e') {{ toggleExtras(); e.preventDefault(); }}
    else if (e.key === 'Escape') {{ closeLightbox(); }}
}});

// 初始化
(function init() {{
    const params = new URLSearchParams(location.search);
    const start = parseInt(params.get('task'));
    if (!isNaN(start) && start >= 0 && start < TASKS.length) selectTask(start);
    else if (TASKS.length) selectTask(0);
}})();
</script>"""

    return HTMLResponse(page(f"Run {run_id}", body, js))


@app.get("/api/run/{run_id}/task/{task_idx}/extras")
def api_task_extras(run_id: str, task_idx: int):
    summary = load_run_summary(run_id)
    results = summary.get("results", [])
    if task_idx < 0 or task_idx >= len(results):
        raise HTTPException(404, "task not found")
    r = results[task_idx]
    game_slug = game_id_to_slug(r["game_id"])
    config = r["config_name"]
    run_idx = r.get("run_index", 0)
    task_dir = find_task_dir(run_id, game_slug, config, f"run_{run_idx}")

    rel_base = f"{game_slug}/{quote(config, safe='')}/run_{run_idx}"
    captions = dict(parse_caption_nl(task_dir / "caption_nl.txt"))
    frames = list_step_frames(task_dir)
    subdir = frames[0].parent.name if frames else "subbed_nl"

    items = []
    for fp in frames:
        stem = fp.stem
        try:
            idx = int(stem.split("_")[-1])
        except ValueError:
            idx = len(items)
        action = "Start State" if idx == 0 else captions.get(idx, "")
        items.append({
            "step": f"Step {idx}" if idx > 0 else "Start",
            "action": action,
            "url": f"/file/{quote(run_id)}/{rel_base}/{subdir}/{fp.name}",
        })
    return JSONResponse({"frames": items})


# ---------------------------------------------------------------------------
# 路由：Task 详情（deprecated —— workspace 已替代，但保留直链向后兼容，
# 访问时重定向到 workspace 并定位 task）
# ---------------------------------------------------------------------------

from fastapi.responses import RedirectResponse


@app.get("/run/{run_id}/task/{game_slug}/{config_name}/{run_dir}")
def task_detail_legacy(run_id: str, game_slug: str, config_name: str, run_dir: str):
    summary = load_run_summary(run_id)
    results = summary.get("results", [])
    target_run = int(run_dir.replace("run_", "")) if run_dir.startswith("run_") else 0
    for i, r in enumerate(results):
        if (
            game_id_to_slug(r["game_id"]) == game_slug
            and r["config_name"] == config_name
            and r.get("run_index", 0) == target_run
        ):
            return RedirectResponse(f"/run/{quote(run_id)}?task={i}")
    return RedirectResponse(f"/run/{quote(run_id)}")


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------

def main(argv=None):
    global COMPARE_DATA, COMPARE_BASE_LABEL, COMPARE_TARGET_LABEL, COMPARE_INITIAL_THRESHOLD, BATCH_TASK_INDEX

    parser = argparse.ArgumentParser(prog="vcl viewer", description="批结果网页 viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--compare-base", help="base 模型的 eval_results JSON（如 Gemini Flash）")
    parser.add_argument("--compare-target", help="target 模型的 eval_results JSON（如 Gemini Pro）")
    parser.add_argument("--compare-threshold", type=float, default=5.0, help="compare 页面初始 score 阈值")
    args = parser.parse_args(argv)

    if bool(args.compare_base) != bool(args.compare_target):
        parser.error("--compare-base 和 --compare-target 必须同时提供")
    if args.compare_base and args.compare_target:
        BATCH_TASK_INDEX = build_batch_task_index()
        base_path = resolve_project_path(args.compare_base)
        target_path = resolve_project_path(args.compare_target)
        COMPARE_DATA = build_eval_compare(base_path, target_path)
        COMPARE_BASE_LABEL = COMPARE_DATA["base_label"]
        COMPARE_TARGET_LABEL = COMPARE_DATA["target_label"]
        COMPARE_INITIAL_THRESHOLD = args.compare_threshold
        initial_count = sum(
            1 for c in COMPARE_DATA["cases"]
            if c["base"].get("score") is not None
            and c["target"].get("score") is not None
            and float(c["target"]["score"]) <= float(c["base"]["score"]) + COMPARE_INITIAL_THRESHOLD
        )
        print(
            f"[batch_viewer] compare: {COMPARE_BASE_LABEL} vs {COMPARE_TARGET_LABEL} "
            f"· paired={COMPARE_DATA['pair_count']} · initial_matches={initial_count} "
            f"· threshold={COMPARE_INITIAL_THRESHOLD:g}"
        )

    print(f"[batch_viewer] 扫描目录: {BATCH_RESULTS_DIR}")
    print(f"[batch_viewer] 浏览器打开: http://{args.host}:{args.port}")
    if COMPARE_DATA is not None:
        print(f"[batch_viewer] 对比页面: http://{args.host}:{args.port}/compare")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
