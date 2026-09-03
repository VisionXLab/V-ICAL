"""
Token Usage 分析工具 — 检查单调递增性 + 生成折线图

用法:
    python vcl.py tokens <path>

    path 可以是:
      - 单个 result.json 文件
      - 工作区目录 (如 batch_results/20260408_101227)，自动扫描所有 result.json
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 需要检查单调递增的字段
MONOTONIC_FIELDS = ["input", "input_text_tokens", "input_image_tokens", "input_video_tokens"]

# 折线图绘制的字段分组
PLOT_GROUPS = {
    "input_breakdown": {
        "title": "Input Token Breakdown",
        "fields": ["input", "input_text_tokens", "input_image_tokens", "input_video_tokens"],
        "ylabel": "Tokens",
    },
    "output_thoughts": {
        "title": "Output & Thoughts Tokens",
        "fields": ["output", "thoughts"],
        "ylabel": "Tokens",
    },
    "total": {
        "title": "Total Tokens per Round",
        "fields": ["total"],
        "ylabel": "Tokens",
    },
}


def load_results(path: Path) -> list[tuple[str, dict]]:
    """加载 result.json，返回 [(label, data), ...]"""
    results = []
    if path.is_file() and path.name == "result.json":
        data = json.loads(path.read_text(encoding="utf-8"))
        label = f"{data.get('game_id', '?')}/{data.get('config_name', '?')}/run_{data.get('run_index', '?')}"
        results.append((label, data))
    elif path.is_dir():
        for f in sorted(path.rglob("result.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("status") != "completed":
                    continue
                label = f"{data.get('game_id', '?')}/{data.get('config_name', '?')}/run_{data.get('run_index', '?')}"
                results.append((label, data))
            except Exception:
                continue
    return results


def check_monotonic(per_round: list[dict]) -> dict:
    """检查各字段是否单调递增，返回 {field: {"ok": bool, "violations": [...]}}"""
    report = {}
    for field in MONOTONIC_FIELDS:
        values = [(r["round"], r[field]) for r in per_round if field in r]
        violations = []
        for i in range(1, len(values)):
            if values[i][1] < values[i - 1][1]:
                violations.append({
                    "round": values[i][0],
                    "prev": values[i - 1][1],
                    "curr": values[i][1],
                    "diff": values[i][1] - values[i - 1][1],
                })
        report[field] = {"ok": len(violations) == 0, "violations": violations}
    return report


def plot_token_usage(per_round: list[dict], label: str, out_dir: Path):
    """为一个 result 生成折线图。"""
    rounds = [r["round"] for r in per_round]

    for group_key, group in PLOT_GROUPS.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        has_data = False
        for field in group["fields"]:
            values = [r.get(field) for r in per_round]
            if any(v is not None for v in values):
                ax.plot(rounds, values, marker=".", markersize=3, label=field, linewidth=1.2)
                has_data = True

        if not has_data:
            plt.close(fig)
            continue

        ax.set_title(f"{group['title']}\n{label}", fontsize=11)
        ax.set_xlabel("Round")
        ax.set_ylabel(group["ylabel"])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe_label = label.replace("/", "_").replace("\\", "_").replace(" ", "_")
        fig.savefig(out_dir / f"{safe_label}__{group_key}.png", dpi=120)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Token Usage Analyzer")
    parser.add_argument("path", type=str, help="result.json 文件或工作区目录")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"[Error] 路径不存在: {path}")
        sys.exit(1)

    results = load_results(path)
    if not results:
        print("[Info] 没有找到已完成的 result.json")
        sys.exit(0)

    print(f"[Info] 找到 {len(results)} 个结果\n")

    # 确定输出目录
    if path.is_dir():
        out_dir = path / "token_analysis"
    else:
        out_dir = path.parent / "token_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 逐个分析
    all_reports = []
    log_lines = []  # 单独的文本 log
    DETAIL_FIELDS = ["input_text_tokens", "input_image_tokens", "input_video_tokens"]

    for label, data in results:
        per_round = data.get("token_usage", {}).get("per_round", [])
        if not per_round:
            print(f"  [{label}] 无 per_round 数据，跳过")
            log_lines.append(f"[{label}] 无 per_round 数据，跳过\n")
            continue

        # 1. 单调递增检查
        report = check_monotonic(per_round)
        all_ok = all(r["ok"] for r in report.values())
        status = "PASS" if all_ok else "FAIL"
        print(f"  [{label}] 单调性: {status}")
        log_lines.append(f"[{label}] 单调性: {status}")

        if not all_ok:
            for field, r in report.items():
                if not r["ok"]:
                    for v in r["violations"]:
                        line = f"    {field}: Round {v['round']} 下降 {v['prev']} -> {v['curr']} (diff={v['diff']})"
                        print(line)
                        log_lines.append(line)

        # 2. 检查每个 round 是否存在分项 token 字段
        missing_rounds = []
        for r in per_round:
            missing = [f for f in DETAIL_FIELDS if f not in r or r[f] is None]
            if missing:
                missing_rounds.append((r["round"], missing))

        if missing_rounds:
            log_lines.append(f"  缺失分项 token 字段的 round:")
            for rnd, fields in missing_rounds:
                log_lines.append(f"    Round {rnd}: 缺少 {', '.join(fields)}")
            print(f"  [{label}] {len(missing_rounds)} 个 round 缺少分项 token 字段")
        else:
            log_lines.append(f"  分项 token 字段: 全部存在")

        log_lines.append("")  # 空行分隔

        all_reports.append({"label": label, "monotonic": {k: v["ok"] for k, v in report.items()},
                            "violations": {k: v["violations"] for k, v in report.items() if not v["ok"]},
                            "missing_detail_fields": [{"round": rnd, "missing": fields} for rnd, fields in missing_rounds] if missing_rounds else []})

        # 3. 折线图
        plot_token_usage(per_round, label, out_dir)

    # 写汇总报告
    report_path = out_dir / "monotonic_report.json"
    report_path.write_text(json.dumps(all_reports, indent=2, ensure_ascii=False), encoding="utf-8")

    # 写文本 log
    log_path = out_dir / "check.log"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    # 过滤出有问题的，写到单独文件
    error_reports = [r for r in all_reports if not all(r["monotonic"].values()) or r.get("missing_detail_fields")]
    error_lines = []
    for r in error_reports:
        error_lines.append(f"{'='*60}")
        error_lines.append(f"[{r['label']}]")
        # 单调性问题
        for field, ok in r["monotonic"].items():
            if not ok:
                for v in r["violations"][field]:
                    error_lines.append(f"  单调性 {field}: Round {v['round']} 下降 {v['prev']} -> {v['curr']} (diff={v['diff']})")
        # 缺失字段
        for m in r.get("missing_detail_fields", []):
            error_lines.append(f"  缺失字段 Round {m['round']}: {', '.join(m['missing'])}")
        error_lines.append("")

    errors_path = out_dir / "errors.log"
    if error_lines:
        errors_path.write_text("\n".join(error_lines), encoding="utf-8")
    elif errors_path.exists():
        errors_path.unlink()

    print(f"\n[Done] 折线图和报告已保存到: {out_dir}")
    print(f"  图表: {len(results) * len(PLOT_GROUPS)} 张")
    print(f"  报告: {report_path}")
    print(f"  日志: {log_path}")
    if error_lines:
        print(f"  异常: {errors_path} ({len(error_reports)} 个)")
    else:
        print(f"  异常: 无")


if __name__ == "__main__":
    main()
