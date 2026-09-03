"""规则主动输出 (rule extraction) × 及格 (pass) 关联分析。

把两个**单侧**指标按 run join 起来，看「AI 在对话里主动输出视频规则的程度」与
「该 run 是否及格」之间的关联性。

两侧数据来源
------------
1. 规则主动输出：analyze_rule_extraction.py 产出的 run 级 rule_extraction.json
   （位于 rule_extraction_results/<...mirror...>/<game_safe>/<config>/run_N/）
   每个 run 取三个指标：
     - coverage              = rounds_with_evidence / total_rounds
     - avg_evidences_per_round = total_evidences / total_rounds
     - n_distinct_rules      = 该 run 命中的不同 delta_rule 数
2. 及格：eval_batch.py --format json 的输出（每条含 config_run / pass / score）

join key = config_run = "<game_safe>/<config>/run_N"，两侧天然一致。

⚠️ 新 batch_results 结构（batch_results/<model>/<ts>/...）下 analyze_rule_extraction
内置的 summary 聚合会因 batch_id 不含模型层而失效；本脚本**不依赖**那个 summary，
直接遍历 run 级 rule_extraction.json 自己算，故完全绕过该 bug。

用法
----
    # 1) 先出 pass（指到 timestamp 层，不要指模型层）
    python vcl.py eval-batch batch_results/seed-2_0-lite/<ts> --format json --out seed_pass.json
    # 2) 跑规则提取（同样指到 timestamp 层）
    python vcl.py rule-extract --scan batch_results/seed-2_0-lite/<ts> --no-auto-merge
    # 3) 关联
    python vcl.py rule-vs-pass \
        --rule-root rule_extraction_results/seed-2_0-lite/<ts> \
        --pass-json seed_pass.json
    # 便利：--batch-dir 自动推 rule-root（把 batch_results 换成 rule_extraction_results）
    python vcl.py rule-vs-pass --batch-dir batch_results/seed-2_0-lite/<ts> --pass-json seed_pass.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from tools._paths import ROOT  # 锚回仓库根

METRICS = ("coverage", "avg_evidences_per_round", "n_distinct_rules")


# ===================== 加载：规则主动输出（per run）=====================

def _rule_metrics_from_parsed(parsed: dict) -> dict:
    """从单个 rule_extraction.json 算三个指标。"""
    rounds = parsed.get("rounds", []) or []
    summary = parsed.get("summary") or {}
    total_rounds = summary.get("total_rounds", len(rounds))
    rounds_with = summary.get(
        "rounds_with_evidence",
        sum(1 for r in rounds if r.get("evidences")),
    )
    total_ev = summary.get(
        "total_evidences",
        sum(len(r.get("evidences") or []) for r in rounds),
    )
    distinct = set()
    for r in rounds:
        for ev in r.get("evidences") or []:
            rule = ev.get("rule_extracted", "")
            if rule:
                distinct.add(rule)
    cov = (rounds_with / total_rounds) if total_rounds else 0.0
    avg_ev = (total_ev / total_rounds) if total_rounds else 0.0
    return {
        "total_rounds": total_rounds,
        "rounds_with_evidence": rounds_with,
        "total_evidences": total_ev,
        "coverage": round(cov, 4),
        "avg_evidences_per_round": round(avg_ev, 4),
        "n_distinct_rules": len(distinct),
    }


def load_rule_runs(rule_root: Path) -> dict[str, dict]:
    """遍历 rule_root 下所有 run 级 rule_extraction.json。
    返回 {config_run: metrics}，config_run = "<game_safe>/<config>/run_N"。"""
    out: dict[str, dict] = {}
    for f in sorted(rule_root.glob("**/rule_extraction.json")):
        rel = f.parent.relative_to(rule_root).as_posix()  # game_safe/config/run_N
        try:
            parsed = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] 跳过无法解析的 {f}: {e}", file=sys.stderr)
            continue
        out[rel] = _rule_metrics_from_parsed(parsed)
    return out


# ===================== 加载：及格 (pass) =====================

def load_pass(pass_json: Path) -> dict[str, dict]:
    """读 eval_batch --format json 输出，返回 {config_run: {pass, score, game}}。"""
    data = json.loads(pass_json.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for rec in data:
        cr = rec.get("config_run")
        if not cr:
            continue
        out[cr] = {
            "pass": rec.get("pass"),
            "score": rec.get("score"),
            "game": rec.get("game"),
            "ending": rec.get("ending"),
            "error": rec.get("error"),
        }
    return out


# ===================== 统计工具 =====================

def pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    """Pearson 相关；point-biserial = pearson(metric, pass∈{0,1})。"""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def _mean(xs: list[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


# ===================== join + 汇总 =====================

def join_rows(rule_runs: dict, pass_runs: dict, target_games: set) -> list[dict]:
    """按 config_run 内联 join。只保留两侧都有的 run。"""
    rows = []
    for cr, m in rule_runs.items():
        p = pass_runs.get(cr)
        if p is None:
            continue
        game_safe = cr.split("/")[0]
        game = p.get("game") or game_safe.replace("_", "/", 1)
        if target_games and game not in target_games and game_safe not in target_games:
            continue
        rows.append({
            "config_run": cr,
            "game_safe": game_safe,
            "game": game,
            **m,
            "pass": p.get("pass"),
            "score": p.get("score"),
            "ending": p.get("ending"),
        })
    rows.sort(key=lambda r: r["config_run"])
    return rows


def global_stats(rows: list[dict]) -> dict:
    """全局：pass/fail 两组各指标均值 + point-biserial 相关。仅用 pass∈{True,False}。"""
    usable = [r for r in rows if r["pass"] in (True, False)]
    n_pass = sum(1 for r in usable if r["pass"] is True)
    n_fail = sum(1 for r in usable if r["pass"] is False)
    n_none = sum(1 for r in rows if r["pass"] not in (True, False))
    stats = {
        "n_joined": len(rows),
        "n_usable": len(usable),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_none_or_error": n_none,
        "metrics": {},
    }
    pass01 = [1.0 if r["pass"] else 0.0 for r in usable]
    for metric in METRICS:
        vals = [float(r[metric]) for r in usable]
        vals_pass = [float(r[metric]) for r in usable if r["pass"] is True]
        vals_fail = [float(r[metric]) for r in usable if r["pass"] is False]
        stats["metrics"][metric] = {
            "mean_pass": _mean(vals_pass),
            "mean_fail": _mean(vals_fail),
            "point_biserial_r": pearson(vals, pass01),
        }
    # 附带：coverage vs score（连续-连续）
    sc = [(float(r["coverage"]), float(r["score"]))
          for r in usable if isinstance(r.get("score"), (int, float))]
    if len(sc) >= 2:
        stats["coverage_vs_score_pearson"] = pearson(
            [a for a, _ in sc], [b for _, b in sc]
        )
    return stats


def per_game_stats(rows: list[dict]) -> list[dict]:
    by_game: dict[str, list[dict]] = {}
    for r in rows:
        by_game.setdefault(r["game_safe"], []).append(r)
    out = []
    for game_safe, items in sorted(by_game.items()):
        usable = [r for r in items if r["pass"] in (True, False)]
        n_pass = sum(1 for r in usable if r["pass"] is True)
        g = {
            "game_safe": game_safe,
            "n_runs": len(items),
            "n_usable": len(usable),
            "n_pass": n_pass,
            "pass_rate": round(n_pass / len(usable), 4) if usable else None,
        }
        for metric in METRICS:
            g[f"mean_{metric}"] = (
                round(_mean([float(r[metric]) for r in usable]), 4) if usable else None
            )
            g[f"mean_{metric}_pass"] = (
                round(_mean([float(r[metric]) for r in usable if r["pass"] is True]), 4)
                if n_pass else None
            )
            g[f"mean_{metric}_fail"] = (
                round(_mean([float(r[metric]) for r in usable if r["pass"] is False]), 4)
                if (len(usable) - n_pass) else None
            )
        out.append(g)
    return out


# ===================== 渲染 =====================

def _fmt(v, w=8, prec=3):
    if v is None:
        return f"{'-':>{w}}"
    if isinstance(v, float):
        return f"{v:>{w}.{prec}f}"
    return f"{str(v):>{w}}"


def render_table(rows, gstats, gamestats) -> str:
    L = []
    L.append("=" * 100)
    L.append("# 规则主动输出 × 及格 关联分析")
    L.append("=" * 100)

    # ---- 全局 ----
    L.append(
        f"\njoined={gstats['n_joined']}  usable(pass∈T/F)={gstats['n_usable']}  "
        f"pass={gstats['n_pass']}  fail={gstats['n_fail']}  none/err={gstats['n_none_or_error']}"
    )
    L.append("\n## 全局：及格组 vs 不及格组 指标均值 + point-biserial 相关")
    L.append(f"{'metric':<26}{'mean(pass)':>12}{'mean(fail)':>12}{'r(pb)':>10}")
    L.append("-" * 60)
    for metric in METRICS:
        m = gstats["metrics"][metric]
        L.append(
            f"{metric:<26}{_fmt(m['mean_pass'],12)}{_fmt(m['mean_fail'],12)}"
            f"{_fmt(m['point_biserial_r'],10)}"
        )
    if "coverage_vs_score_pearson" in gstats:
        L.append(f"\ncoverage vs score (Pearson): "
                 f"{_fmt(gstats['coverage_vs_score_pearson'],0)}")

    # ---- per-game ----
    L.append("\n## 按游戏汇总")
    hdr = (f"{'game':<24}{'runs':>5}{'pass':>5}{'rate':>7}"
           f"{'cov(p)':>9}{'cov(f)':>9}{'avgEv(p)':>10}{'avgEv(f)':>10}"
           f"{'nR(p)':>7}{'nR(f)':>7}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for g in gamestats:
        L.append(
            f"{g['game_safe']:<24}{g['n_runs']:>5}{g['n_pass']:>5}"
            f"{_fmt(g['pass_rate'],7,3)}"
            f"{_fmt(g['mean_coverage_pass'],9)}{_fmt(g['mean_coverage_fail'],9)}"
            f"{_fmt(g['mean_avg_evidences_per_round_pass'],10)}"
            f"{_fmt(g['mean_avg_evidences_per_round_fail'],10)}"
            f"{_fmt(g['mean_n_distinct_rules_pass'],7,1)}"
            f"{_fmt(g['mean_n_distinct_rules_fail'],7,1)}"
        )

    # ---- per-run 明细 ----
    L.append("\n## 明细（per-run）")
    hdr2 = (f"{'config_run':<46}{'pass':>6}{'cov':>7}{'avgEv':>8}"
            f"{'nRule':>7}{'score':>8}  ending")
    L.append(hdr2)
    L.append("-" * len(hdr2))
    for r in rows:
        L.append(
            f"{r['config_run']:<46}{str(r['pass']):>6}"
            f"{_fmt(r['coverage'],7,3)}{_fmt(r['avg_evidences_per_round'],8,3)}"
            f"{r['n_distinct_rules']:>7}{_fmt(r.get('score'),8,1)}  "
            f"{r.get('ending') or '-'}"
        )
    return "\n".join(L)


def render_csv(rows) -> str:
    cols = ["config_run", "game", "pass", "score", "ending",
            "total_rounds", "rounds_with_evidence", "total_evidences",
            "coverage", "avg_evidences_per_round", "n_distinct_rules"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    return "\n".join(lines)


# ===================== main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule-root", default=None,
                    help="rule_extraction_results 下的 batch 根（含 game_safe/config/run_N 子树）")
    ap.add_argument("--batch-dir", default=None,
                    help="便利项：batch_results 下的 timestamp 目录，自动推 rule-root")
    ap.add_argument("--pass-json", required=True,
                    help="eval_batch.py --format json 的输出文件")
    ap.add_argument("--game", action="append", default=None,
                    help="限定游戏（gym_id 或 game_safe），可多次")
    ap.add_argument("--format", choices=["table", "csv", "json"], default="table")
    ap.add_argument("--out", default=None, help="结果写文件而非 stdout")
    args = ap.parse_args()

    # 解析 rule_root
    if args.rule_root:
        rule_root = Path(args.rule_root)
    elif args.batch_dir:
        bd = Path(args.batch_dir).resolve()
        parts = list(bd.parts)
        if "batch_results" in parts:
            parts[parts.index("batch_results")] = "rule_extraction_results"
            rule_root = Path(*parts)
        else:
            ap.error("--batch-dir 路径里没有 'batch_results'，无法自动推 rule-root")
    else:
        ap.error("必须给 --rule-root 或 --batch-dir")
    if not rule_root.is_absolute():
        rule_root = ROOT / rule_root
    if not rule_root.is_dir():
        ap.error(f"rule-root 不存在: {rule_root}")

    pass_json = Path(args.pass_json)
    if not pass_json.is_absolute():
        pass_json = ROOT / pass_json
    if not pass_json.is_file():
        ap.error(f"pass-json 不存在: {pass_json}")

    target = set(args.game) if args.game else set()

    rule_runs = load_rule_runs(rule_root)
    pass_runs = load_pass(pass_json)
    rows = join_rows(rule_runs, pass_runs, target)

    # 未匹配诊断
    only_rule = set(rule_runs) - set(pass_runs)
    only_pass = set(pass_runs) - set(rule_runs)
    print(f"[join] rule_runs={len(rule_runs)} pass_runs={len(pass_runs)} "
          f"joined={len(rows)}  only_rule={len(only_rule)} only_pass={len(only_pass)}",
          file=sys.stderr)
    if only_rule:
        print(f"  [only in rule] 示例: {sorted(only_rule)[:3]}", file=sys.stderr)
    if only_pass:
        print(f"  [only in pass] 示例: {sorted(only_pass)[:3]}", file=sys.stderr)

    gstats = global_stats(rows)
    gamestats = per_game_stats(rows)

    if args.format == "csv":
        out = render_csv(rows)
    elif args.format == "json":
        out = json.dumps(
            {"global": gstats, "per_game": gamestats, "rows": rows},
            ensure_ascii=False, indent=2,
        )
    else:
        out = render_table(rows, gstats, gamestats)

    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"[done] 写入 {args.out}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
