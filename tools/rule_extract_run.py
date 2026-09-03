"""并发驱动 analyze_rule_extraction.run_one（复用其全部逻辑，仅并发执行）。

为什么需要它：analyze_rule_extraction --scan 是串行 await，每 run ~40s（pro 模型
thinking 量大），数百 run 要跑数小时；且对 API 偶发 502 只有 3 次内层重试，高并发下
会被击穿。本驱动做两件事：
  1. N 路并发（信号量限流）——实测代理在并发 4 下 0 502、并发 8 下 ~40% 502；
  2. 外层重试——每 run 最多 3 外 × 3 内 = 9 次 API 机会，对偶发 502 几乎必过。
只写 rule_extraction.json（关联分析够用；viewer md/html 可用
analyze_rule_extraction.py --rebuild-views 事后补）。

用法:
    python vcl.py rule-extract-run [batch_dir] [并发度]
    python vcl.py rule-extract-run batch_results/seed-2_0-lite 4
默认 batch_results/seed-2_0-lite、并发度 4。幂等：已有 rule_extraction.json 的 run 自动 skip，
失败的 run 重跑本脚本即可补齐。
"""
import asyncio
import json
import sys
from pathlib import Path

from tools._paths import ROOT  # 锚回仓库根 + 注入 sys.path
from tools import rule_extract as A  # noqa: E402

_arg_batch = sys.argv[1] if len(sys.argv) > 1 else "batch_results/seed-2_0-lite"
BATCH = Path(_arg_batch)
if not BATCH.is_absolute():
    BATCH = ROOT / BATCH
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 4


def build_todo():
    cfg = A.load_json(A.CONFIG_PATH)
    detailed = A.load_json(A.DETAILED_RULES_PATH)["games"]
    items = A.discover_from_batch_summary(BATCH)  # [(game_id, config, run_idx, conv)]
    todo = []
    skip_empty = skip_done = 0
    for game_id, cfgn, ri, conv in items:
        det = A._lookup(detailed, game_id)
        if det is None or not det.strip():
            skip_empty += 1
            continue
        out = A.rule_extraction_path_for(conv)
        if out.exists():
            skip_done += 1
            continue
        todo.append((game_id, conv, out))
    return cfg, todo, skip_empty, skip_done


async def main():
    cfg, todo, skip_empty, skip_done = build_todo()
    total = len(todo)
    print(f"[concurrent] batch={BATCH.name} total_to_run={total} skip_empty={skip_empty} "
          f"skip_existing={skip_done} conc={CONC}", flush=True)
    sem = asyncio.Semaphore(CONC)
    state = {"done": 0, "fail": 0}

    async def one(game_id, conv, out):
        async with sem:
            ok = False
            last = None
            for attempt in range(3):  # 外层重试（run_one 内部还有 3 次）→ 最多 9 次 API 机会
                try:
                    parsed = await A.run_one(game_id, conv, cfg)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(parsed, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
                    ok = True
                    break
                except Exception as e:
                    last = e
                    await asyncio.sleep(5 * (attempt + 1))
            if not ok:
                state["fail"] += 1
                print(f"[FAIL] {game_id} {Path(conv).parent.name}: {str(last)[:120]}",
                      flush=True)
            state["done"] += 1
            if state["done"] % 10 == 0 or state["done"] == total:
                print(f"  progress {state['done']}/{total} (fail {state['fail']})",
                      flush=True)

    await asyncio.gather(*[one(g, c, o) for g, c, o in todo])
    print(f"[concurrent] DONE done={state['done']}/{total} fail={state['fail']}",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
