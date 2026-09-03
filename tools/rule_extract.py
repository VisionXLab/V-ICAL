"""
rule_extract.py  (subcommand: `vcl rule-extract`)
=================================================

Quantitative pipeline for: "Did the AI player extract rules from the VIDEO
that the textual rules did NOT tell it?"

Input
-----
- A game id (e.g. "Taxi-v3")
- A conversation.json from a batch_results run

Defaults to AI-only text (filters out human prompts and image/video).

Output
------
JSON with:
- total_rounds
- rounds_with_evidence
- total_evidences
- per-round evidence breakdown (rule_extracted / quote / source_rule_in_detailed)

LLM config: re-uses ./config.json (the same Gemini Native config the main app uses).
"""
import argparse
import asyncio
import html
import json
import re
import sys
from pathlib import Path

from tools._paths import ROOT  # 锚回仓库根 + 注入 sys.path

from ai_backends.google_genai import call_gemini_native  # noqa: E402


CONFIG_PATH = ROOT / "config.json"
STANDARD_RULES_PATH = ROOT / "prompt" / "templates" / "game_rules.json"
DETAILED_RULES_PATH = ROOT / "prompt" / "templates" / "detailed_game_rules.json"
DEFAULT_OUT_ROOT = ROOT / "rule_extraction_results"


def default_output_path(conv_path: Path) -> Path:
    """Mirror the batch_results sub-structure under rule_extraction_results/.
    e.g. batch_results/20260408_140210/Taxi-v3/state1/run_0/conversation.json
      -> rule_extraction_results/20260408_140210/Taxi-v3/state1/run_0/rule_extraction.json

    Falls back to rule_extraction_results/<conv_stem>.json if conv_path is not
    under a 'batch_results' ancestor.
    """
    parts = conv_path.resolve().parts
    if "batch_results" in parts:
        idx = parts.index("batch_results")
        tail = parts[idx + 1 : -1]  # drop 'batch_results' and the conversation.json filename
        return DEFAULT_OUT_ROOT.joinpath(*tail) / "rule_extraction.json"
    return DEFAULT_OUT_ROOT / f"{conv_path.stem}_rule_extraction.json"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _canon(s: str) -> str:
    """规范化 game key：吸收 'procgen_:' vs 'procgen:' vs 'procgen_' 的写法差异。
    / 和 : 都归一为 _，连续 _ 坍缩成一个，再小写。三种 procgen 写法→同一 canonical。"""
    s = re.sub(r"[/:]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.lower()


def _lookup(table: dict, game_id: str):
    """先精确命中；未命中才走 canonical 容错（仅此时构建反查表，零影响其他游戏）。"""
    if game_id in table:
        return table[game_id]
    canon = {_canon(k): k for k in table}
    real = canon.get(_canon(game_id))
    return table[real] if real else None


def load_rules(game_id: str) -> tuple[str, str]:
    standard = _lookup(load_json(STANDARD_RULES_PATH)["games"], game_id)
    detailed = _lookup(load_json(DETAILED_RULES_PATH)["games"], game_id)
    if standard is None:
        raise KeyError(f"'{game_id}' not in {STANDARD_RULES_PATH}")
    if detailed is None:
        raise KeyError(f"'{game_id}' not in {DETAILED_RULES_PATH}")
    if not detailed.strip():
        raise ValueError(
            f"detailed_game_rules.json has an EMPTY rule for '{game_id}'. "
            "Curate it first before running rule-extraction analysis."
        )
    return standard, detailed


def extract_ai_responses(conv_path: Path) -> list[dict]:
    data = load_json(conv_path)
    out = []
    for r in data.get("rounds", []):
        out.append({
            "round": r.get("round"),
            "response": r.get("response", ""),
            "action": r.get("action_meaning"),
        })
    return out


def build_judge_prompt(
    game_id: str,
    standard_rules: str,
    detailed_rules: str,
    rounds: list[dict],
) -> str:
    rounds_block = []
    for r in rounds:
        rounds_block.append(
            f"--- Round {r['round']} (action chosen: {r['action']}) ---\n{r['response'].strip()}"
        )
    rounds_text = "\n\n".join(rounds_block)

    return f"""You are an evaluator measuring video-grounded rule learning in a vision-based game-playing AI.

The AI player was given only a SHORT textual rule (STANDARD_RULES below). Anything beyond that — passenger placement, walls, scoring numbers, illegal-action penalties, win/loss conditions, etc. — it would have to infer from the VIDEO of past gameplay.

We have a longer, ground-truth rulebook (DETAILED_RULES) describing what the game ACTUALLY does. Your job is to look at each round of the AI's chat output and decide: does the AI explicitly demonstrate knowledge of a rule that appears in DETAILED_RULES but is NOT mentioned in STANDARD_RULES?

GAME: {game_id}

================ STANDARD_RULES (what the AI was told in text) ================
{standard_rules}
==============================================================================

================ DETAILED_RULES (ground-truth, AI did NOT see this) ==========
{detailed_rules}
==============================================================================

================ AI PLAYER RESPONSES (round by round) ========================
{rounds_text}
==============================================================================

INSTRUCTIONS

STEP 1 — Derive `delta_rules` (STRICT, NO IMAGINATION)
- A delta_rule is a rule that is EXPLICITLY stated in DETAILED_RULES but NOT stated in STANDARD_RULES.
- Every delta_rule MUST be directly traceable to a specific phrase in DETAILED_RULES. Paraphrasing the wording is OK; introducing details is NOT.
- DO NOT use outside knowledge, common sense, genre conventions, or "what would obviously be true in this kind of game". If DETAILED_RULES does not say it, it is NOT a delta_rule — even if it looks obvious or you are confident it is true.
- Visual appearance (colors of sprites, exact shapes, on-screen labels, screen layout positions) is excluded UNLESS DETAILED_RULES explicitly states it.
- When in doubt, OMIT. The bar is: "I can point to a clause in DETAILED_RULES that says this."

STEP 2 — Find `evidences` (only against delta_rules from STEP 1)
- For EACH round, scan the AI's response.
- An "evidence" is a concrete statement in the AI response that demonstrates knowledge of ONE of the delta_rules you listed in STEP 1.
- The `rule_extracted` field of every evidence MUST be character-for-character identical to one of the items in `delta_rules`. Do not coin new rules at this stage.
- If the AI says something that is NOT covered by any delta_rule (e.g. it names a sprite color, claims a layout detail, or invokes outside knowledge), DO NOT record it as evidence — even if it sounds like a "video-derived" observation. Such statements are out of scope for this evaluation.
- Generic restatements that could come purely from STANDARD_RULES do NOT count. Mere action tokens (like `[Action: Up]`) do NOT count.
- Multiple evidences per round are allowed only if they map to DISTINCT delta_rules. Do not list the same delta_rule twice in one round as repeated restatement.

STEP 3 — Be conservative
- It is better to under-count than to over-count. If unsure whether a statement counts, leave it out.
- The goal is to measure how much DOCUMENTED rule content the AI extracted from the video; not to measure how much the AI talked about the game in general.

Output a SINGLE JSON object, no prose, no markdown fences, with this exact schema:

{{
  "delta_rules": ["short string per delta rule you identified"],
  "rounds": [
    {{
      "round": <int>,
      "evidences": [
        {{
          "rule_extracted": "which delta rule this evidence demonstrates (must match one item in delta_rules)",
          "quote": "<= 200 char verbatim snippet from the AI response",
          "explanation": "one short sentence on why this counts"
        }}
      ]
    }}
  ],
  "summary": {{
    "total_rounds": <int>,
    "rounds_with_evidence": <int>,
    "total_evidences": <int>
  }}
}}

Include EVERY round in "rounds" (even ones with zero evidences — use an empty list). Return the JSON object only.
"""


def parse_judge_json(text: str) -> dict:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in judge output:\n{text[:500]}")
    return json.loads(text[start : end + 1])


def recompute_summary(parsed: dict) -> dict:
    rounds = parsed.get("rounds", [])
    total_rounds = len(rounds)
    rounds_with = sum(1 for r in rounds if r.get("evidences"))
    total_ev = sum(len(r.get("evidences") or []) for r in rounds)
    parsed["summary"] = {
        "total_rounds": total_rounds,
        "rounds_with_evidence": rounds_with,
        "total_evidences": total_ev,
    }
    return parsed["summary"]


async def run_one(
    game_id: str,
    conv_path: Path,
    cfg: dict,
) -> dict:
    standard, detailed = load_rules(game_id)
    rounds = extract_ai_responses(conv_path)
    if not rounds:
        raise ValueError(f"No rounds found in {conv_path}")

    prompt = build_judge_prompt(game_id, standard, detailed, rounds)
    messages = [{"role": "user", "content": prompt}]

    ai = cfg["ai"]
    base_url = ai["base_url"]
    if base_url.rstrip("/").endswith("/v1") is False:
        base_url = base_url.rstrip("/") + "/v1"

    last_err = None
    for attempt in range(3):
        try:
            text, usage = await call_gemini_native(
                messages=messages,
                api_key=ai["api_key"],
                base_url=base_url,
                model=ai["model"],
                temperature=ai.get("temperature", 0.2),
                max_tokens=ai.get("max_tokens", 50000),
            )
            break
        except Exception as e:
            last_err = e
            wait = 2 ** (attempt + 1)
            print(f"[retry] attempt {attempt + 1}/3 failed ({e}); sleeping {wait}s")
            await asyncio.sleep(wait)
    else:
        raise RuntimeError(f"Gemini API failed after 3 retries: {last_err}")

    parsed = parse_judge_json(text)
    summary = recompute_summary(parsed)
    parsed["_meta"] = {
        "game_id": game_id,
        "conversation_path": str(conv_path),
        "judge_model": ai["model"],
        "judge_usage": usage,
        "ai_total_rounds_in_file": len(rounds),
    }
    print(
        f"[{game_id}] {conv_path}  →  rounds={summary['total_rounds']}  "
        f"with_evidence={summary['rounds_with_evidence']}  "
        f"total_evidences={summary['total_evidences']}"
    )
    return parsed


def write_ai_responses_md(rounds: list[dict], out_md: Path, game_id: str, conv_path: Path) -> None:
    """Plain-text Markdown unpack of every AI response (no images/videos)."""
    lines = [
        f"# AI Responses — {game_id}",
        "",
        f"Source: `{conv_path}`",
        f"Total rounds: {len(rounds)}",
        "",
        "---",
        "",
    ]
    for r in rounds:
        lines.append(f"## Round {r['round']}  ·  action = `{r['action']}`")
        lines.append("")
        lines.append(r["response"].strip() or "_(empty)_")
        lines.append("")
        lines.append("---")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


_PALETTE = [
    "#fde68a", "#bbf7d0", "#bae6fd", "#fbcfe8", "#ddd6fe",
    "#fed7aa", "#a7f3d0", "#fecaca", "#e9d5ff", "#fef08a",
]


def _highlight_response(text: str, quotes_with_color: list[tuple[str, str]]) -> str:
    """Wrap each quote (verbatim) in <mark style="background:color">…</mark>.
    Non-matching quotes are silently skipped; the response text is HTML-escaped first."""
    safe = html.escape(text)
    for quote, color in quotes_with_color:
        if not quote:
            continue
        q = quote.strip().strip('"').strip("'")
        if not q:
            continue
        q_esc = html.escape(q)
        if q_esc in safe:
            safe = safe.replace(
                q_esc,
                f'<mark style="background:{color};padding:0 2px;border-radius:2px">{q_esc}</mark>',
                1,
            )
    return safe


def write_viewer_html(
    rounds: list[dict],
    parsed: dict,
    out_html: Path,
    game_id: str,
    conv_path: Path,
    standard_rules: str,
    detailed_rules: str,
) -> None:
    delta_rules = parsed.get("delta_rules", [])
    summary = parsed.get("summary", {})
    rule_color = {rule: _PALETTE[i % len(_PALETTE)] for i, rule in enumerate(delta_rules)}

    ev_by_round: dict[int, list[dict]] = {}
    for r in parsed.get("rounds", []):
        ev_by_round[r["round"]] = r.get("evidences", []) or []

    def chip(rule: str) -> str:
        c = rule_color.get(rule, "#e5e7eb")
        return f'<span class="chip" style="background:{c}">{html.escape(rule)}</span>'

    delta_list_html = "".join(
        f'<li>{chip(r)}</li>' for r in delta_rules
    )

    round_cards = []
    for r in rounds:
        rnum = r["round"]
        action = r["action"]
        resp = r["response"]
        evs = ev_by_round.get(rnum, [])
        quotes_colors = [(ev.get("quote", ""), rule_color.get(ev.get("rule_extracted", ""), "#fde68a")) for ev in evs]
        resp_html = _highlight_response(resp, quotes_colors).replace("\n", "<br>")

        ev_html_parts = []
        if not evs:
            ev_html_parts.append('<div class="ev empty">— no evidence —</div>')
        for ev in evs:
            rule = ev.get("rule_extracted", "")
            color = rule_color.get(rule, "#e5e7eb")
            ev_html_parts.append(
                f'<div class="ev" style="border-left-color:{color}">'
                f'<div class="ev-rule">{html.escape(rule)}</div>'
                f'<div class="ev-quote">“{html.escape(ev.get("quote",""))}”</div>'
                f'<div class="ev-expl">{html.escape(ev.get("explanation",""))}</div>'
                f'</div>'
            )
        round_cards.append(
            f'<section class="round" id="round-{rnum}">'
            f'<div class="round-head"><span class="rnum">Round {rnum}</span>'
            f'<span class="action">action: <code>{html.escape(str(action))}</code></span>'
            f'<span class="evcount">{len(evs)} evidence{"s" if len(evs)!=1 else ""}</span></div>'
            f'<div class="round-body">'
            f'<div class="response">{resp_html}</div>'
            f'<aside class="evidences">{"".join(ev_html_parts)}</aside>'
            f'</div></section>'
        )

    nav_html = "".join(
        f'<a href="#round-{r["round"]}" class="nav-pill{"" if not ev_by_round.get(r["round"]) else " has-ev"}">{r["round"]}</a>'
        for r in rounds
    )

    css = """
    :root { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
    body { margin:0; background:#f8fafc; color:#0f172a; }
    header { position:sticky; top:0; z-index:10; background:#0f172a; color:#f8fafc; padding:14px 24px; box-shadow:0 1px 3px rgba(0,0,0,.2); }
    header h1 { margin:0 0 4px 0; font-size:17px; font-weight:600; }
    header .meta { font-size:12px; opacity:.75; }
    .summary { display:flex; gap:24px; margin-top:8px; font-size:13px; }
    .summary b { font-size:18px; display:block; }
    .nav { background:#1e293b; padding:8px 24px; display:flex; gap:6px; flex-wrap:wrap; }
    .nav-pill { color:#cbd5e1; background:#334155; padding:2px 8px; border-radius:10px; text-decoration:none; font-size:12px; }
    .nav-pill.has-ev { background:#facc15; color:#0f172a; font-weight:600; }
    .nav-pill:hover { background:#fbbf24; color:#0f172a; }
    main { display:grid; grid-template-columns: minmax(300px, 360px) 1fr; gap:0; }
    .side { padding:18px 18px 30px 24px; border-right:1px solid #e2e8f0; background:#fff; position:sticky; top:96px; align-self:start; max-height:calc(100vh - 96px); overflow:auto; }
    .side h3 { font-size:13px; margin:14px 0 6px; color:#334155; text-transform:uppercase; letter-spacing:.05em; }
    .side ul { padding-left:0; list-style:none; margin:0; }
    .side li { margin:4px 0; }
    .chip { display:inline-block; padding:2px 8px; border-radius:10px; font-size:12px; color:#0f172a; }
    .rules { white-space:pre-wrap; font-size:12.5px; color:#475569; background:#f1f5f9; padding:8px 10px; border-radius:6px; max-height:180px; overflow:auto; }
    .rounds { padding:16px 24px 64px; }
    .round { background:#fff; border:1px solid #e2e8f0; border-radius:8px; margin-bottom:14px; overflow:hidden; }
    .round-head { display:flex; align-items:center; gap:14px; padding:8px 14px; background:#f1f5f9; font-size:13px; border-bottom:1px solid #e2e8f0; }
    .rnum { font-weight:700; }
    .action code { background:#e2e8f0; padding:1px 6px; border-radius:4px; }
    .evcount { margin-left:auto; color:#64748b; font-size:12px; }
    .round-body { display:grid; grid-template-columns: 1fr 320px; }
    .response { padding:12px 16px; line-height:1.55; font-size:14px; }
    .evidences { background:#fafafa; border-left:1px solid #e2e8f0; padding:10px 12px; }
    .ev { border-left:3px solid #cbd5e1; padding:6px 8px; margin-bottom:8px; background:#fff; border-radius:0 4px 4px 0; }
    .ev.empty { color:#94a3b8; font-style:italic; border:none; background:transparent; padding:6px 0; }
    .ev-rule { font-weight:600; font-size:12.5px; margin-bottom:3px; }
    .ev-quote { font-size:12px; color:#475569; font-style:italic; margin-bottom:3px; }
    .ev-expl { font-size:11.5px; color:#64748b; }
    mark { color:#0f172a; }
    @media (max-width: 1100px) { main { grid-template-columns: 1fr; } .side { position:static; max-height:none; } .round-body { grid-template-columns: 1fr; } .evidences { border-left:none; border-top:1px solid #e2e8f0; } }
    """

    html_doc = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><title>{html.escape(game_id)} · rule-extraction viewer</title>
<style>{css}</style>
</head><body>
<header>
  <h1>{html.escape(game_id)} — rule-extraction viewer</h1>
  <div class="meta">{html.escape(str(conv_path))}</div>
  <div class="summary">
    <div><b>{summary.get('total_rounds',0)}</b>total rounds</div>
    <div><b>{summary.get('rounds_with_evidence',0)}</b>rounds w/ evidence</div>
    <div><b>{summary.get('total_evidences',0)}</b>total evidences</div>
    <div><b>{len(delta_rules)}</b>delta rules</div>
  </div>
</header>
<nav class="nav">{nav_html}</nav>
<main>
  <aside class="side">
    <h3>Delta rules (video-derived)</h3>
    <ul>{delta_list_html}</ul>
    <h3>Standard rules (given to AI)</h3>
    <div class="rules">{html.escape(standard_rules)}</div>
    <h3>Detailed rules (ground truth)</h3>
    <div class="rules">{html.escape(detailed_rules)}</div>
  </aside>
  <div class="rounds">
    {''.join(round_cards)}
  </div>
</main>
</body></html>
"""
    out_html.write_text(html_doc, encoding="utf-8")


# ===================== Aggregation helpers =====================

EMPTY_SUMMARY = {
    "n_runs": 0,
    "n_runs_succeeded": 0,
    "n_runs_failed": 0,
    "total_rounds": 0,
    "rounds_with_evidence": 0,
    "total_evidences": 0,
    "coverage": 0.0,
    "avg_evidences_per_round": 0.0,
    "delta_rule_hits": {},  # rule_text -> number of rounds across all runs that hit this rule
}


def _finalize(agg: dict) -> dict:
    tr = agg["total_rounds"]
    agg["coverage"] = round(agg["rounds_with_evidence"] / tr, 4) if tr else 0.0
    agg["avg_evidences_per_round"] = round(agg["total_evidences"] / tr, 4) if tr else 0.0
    # stable sort hits desc
    agg["delta_rule_hits"] = dict(
        sorted(agg["delta_rule_hits"].items(), key=lambda kv: (-kv[1], kv[0]))
    )
    return agg


def summarize_run(parsed: dict) -> dict:
    """Per-run summary in the unified shape (parents will fold these)."""
    s = parsed.get("summary") or {}
    out = {
        **{k: v for k, v in EMPTY_SUMMARY.items()},
        "n_runs": 1,
        "n_runs_succeeded": 1,
        "total_rounds": s.get("total_rounds", 0),
        "rounds_with_evidence": s.get("rounds_with_evidence", 0),
        "total_evidences": s.get("total_evidences", 0),
        "delta_rule_hits": {},
    }
    for r in parsed.get("rounds", []):
        seen = set()
        for ev in r.get("evidences") or []:
            rule = ev.get("rule_extracted", "")
            if not rule or rule in seen:
                continue
            seen.add(rule)
            out["delta_rule_hits"][rule] = out["delta_rule_hits"].get(rule, 0) + 1
    return _finalize(out)


def fold(children: list[dict]) -> dict:
    """Merge child summaries into a parent summary."""
    agg = {**{k: v for k, v in EMPTY_SUMMARY.items()}, "delta_rule_hits": {}}
    for c in children:
        agg["n_runs"] += c.get("n_runs", 0)
        agg["n_runs_succeeded"] += c.get("n_runs_succeeded", 0)
        agg["n_runs_failed"] += c.get("n_runs_failed", 0)
        agg["total_rounds"] += c.get("total_rounds", 0)
        agg["rounds_with_evidence"] += c.get("rounds_with_evidence", 0)
        agg["total_evidences"] += c.get("total_evidences", 0)
        for rule, hits in (c.get("delta_rule_hits") or {}).items():
            agg["delta_rule_hits"][rule] = agg["delta_rule_hits"].get(rule, 0) + hits
    return _finalize(agg)


# ===================== Batch scan & summary writing =====================

def discover_from_batch_summary(batch_dir: Path) -> list[tuple[str, str, int, Path]]:
    """Use batch_results/<id>/summary.json to enumerate (game_id, config_name, run_index, conv_path).
    Falls back to a filesystem walk if summary.json is missing or empty."""
    summary_path = batch_dir / "summary.json"
    out: list[tuple[str, str, int, Path]] = []
    if summary_path.exists():
        s = load_json(summary_path)
        for r in s.get("results", []):
            game_id = r["game_id"]
            game_safe = game_id.replace("/", "_").replace(":", "_")
            config_name = r["config_name"]
            run_index = r.get("run_index", 0)
            conv = batch_dir / game_safe / config_name / f"run_{run_index}" / "conversation.json"
            if conv.exists():
                out.append((game_id, config_name, run_index, conv))
        if out:
            return out
    # fallback: walk filesystem (game_id will need detailed_game_rules.json to disambiguate)
    detailed = load_json(DETAILED_RULES_PATH)["games"]
    safe_to_real = {gid.replace("/", "_").replace(":", "_"): gid for gid in detailed.keys()}
    for conv in sorted(batch_dir.glob("*/*/run_*/conversation.json")):
        rel = conv.relative_to(batch_dir).parts
        game_safe, config_name, run_dir = rel[0], rel[1], rel[2]
        run_index = int(run_dir.split("_")[1]) if "_" in run_dir else 0
        game_id = safe_to_real.get(game_safe, game_safe)
        out.append((game_id, config_name, run_index, conv))
    return out


def rule_extraction_path_for(conv_path: Path) -> Path:
    return default_output_path(conv_path)


def _load_merge_map(game_dir: Path) -> tuple[dict[str, str], list[dict]]:
    """Returns (raw_text -> canonical_text, groups list). Empty if no rules_merged.json."""
    p = game_dir / "rules_merged.json"
    if not p.exists():
        return {}, []
    data = load_json(p)
    groups = data.get("groups", [])
    m: dict[str, str] = {}
    for g in groups:
        canon = g.get("canonical", "")
        for member in g.get("members", []):
            m[member] = canon
    return m, groups


def _apply_merge_map(hits: dict[str, int], merge_map: dict[str, str]) -> dict[str, int]:
    if not merge_map:
        return dict(hits)
    out: dict[str, int] = {}
    for raw, n in hits.items():
        canon = merge_map.get(raw, raw)
        out[canon] = out.get(canon, 0) + n
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def write_summaries_for_batch(batch_id: str) -> dict:
    """Walk rule_extraction_results/<batch_id>/, write per-config/game/batch summary.json.
    If a game has rules_merged.json, apply that map to delta_rule_hits at game and above.
    Returns the batch-level summary dict."""
    batch_root = DEFAULT_OUT_ROOT / batch_id
    if not batch_root.exists():
        raise FileNotFoundError(batch_root)

    game_summaries = []
    for game_dir in sorted(p for p in batch_root.iterdir() if p.is_dir()):
        config_summaries = []
        for config_dir in sorted(p for p in game_dir.iterdir() if p.is_dir()):
            run_summaries = []
            for run_dir in sorted(p for p in config_dir.iterdir() if p.is_dir() and p.name.startswith("run_")):
                ext = run_dir / "rule_extraction.json"
                if not ext.exists():
                    continue
                try:
                    parsed = load_json(ext)
                except Exception:
                    continue
                run_summaries.append(summarize_run(parsed))
            if not run_summaries:
                continue
            csum = fold(run_summaries)
            csum["scope"] = "config"
            csum["config_name"] = config_dir.name
            (config_dir / "summary.json").write_text(
                json.dumps(csum, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            config_summaries.append(csum)
        if not config_summaries:
            continue
        gsum = fold(config_summaries)
        gsum["scope"] = "game"
        gsum["game_dir"] = game_dir.name
        gsum["n_configs"] = len(config_summaries)

        # Apply rule merge map (if available) at game level and propagate up.
        merge_map, groups = _load_merge_map(game_dir)
        if merge_map:
            raw_hits = gsum["delta_rule_hits"]
            merged_hits = _apply_merge_map(raw_hits, merge_map)
            gsum["delta_rule_hits_raw"] = raw_hits
            gsum["delta_rule_hits"] = merged_hits
            gsum["n_unique_rules_raw"] = len(raw_hits)
            gsum["n_unique_rules_merged"] = len(merged_hits)
            gsum["delta_rule_groups"] = groups

        (game_dir / "summary.json").write_text(
            json.dumps(gsum, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        game_summaries.append(gsum)

    bsum = fold(game_summaries) if game_summaries else {**EMPTY_SUMMARY, "delta_rule_hits": {}}
    bsum["scope"] = "batch"
    bsum["batch_id"] = batch_id
    bsum["n_games"] = len(game_summaries)
    bsum["games"] = [
        {
            "game_dir": g["game_dir"],
            "n_configs": g["n_configs"],
            "n_runs": g["n_runs"],
            "total_rounds": g["total_rounds"],
            "rounds_with_evidence": g["rounds_with_evidence"],
            "total_evidences": g["total_evidences"],
            "coverage": g["coverage"],
        }
        for g in game_summaries
    ]
    (batch_root / "summary.json").write_text(
        json.dumps(bsum, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bsum


# ===================== Rule merging (LLM) =====================

def _build_merge_prompt(game_id: str, rule_texts: list[str]) -> str:
    rule_list = "\n".join(f"- {r}" for r in rule_texts)
    return f"""You are normalizing a list of game rules that an LLM-as-judge extracted across multiple AI player conversations for the game: {game_id}.

Because the judge paraphrased the same underlying rule differently across different runs and rounds, the list below contains DUPLICATES that mean the same thing in different words. Your job is to GROUP semantically equivalent rules.

INPUT — list of rule texts (each was emitted by some judge call):
{rule_list}

INSTRUCTIONS
- Two texts belong to the SAME group iff they assert the same underlying rule about the game. Different specifics, different scope, or different polarity (positive vs negative) = DIFFERENT groups.
- Within each group, pick ONE `canonical` wording. Prefer the most informative member; do not invent new details that no member states.
- Every input text MUST appear in EXACTLY ONE group's `members` list, copied verbatim.
- A rule with no duplicates becomes a singleton group of size 1 (canonical = the rule itself).
- Be CONSERVATIVE: when unsure, keep them separate.

Output a SINGLE JSON object, no prose, no markdown fences:

{{
  "groups": [
    {{"canonical": "<chosen wording>", "members": ["<raw text 1>", "<raw text 2>", ...]}}
  ]
}}
"""


async def merge_rules_for_game(game_id: str, raw_hits: dict[str, int], cfg: dict) -> list[dict]:
    """Call LLM to group semantically equivalent delta_rule texts.
    Returns list of {canonical, members, hits} sorted by hits desc."""
    if not raw_hits:
        return []
    if len(raw_hits) == 1:
        only = next(iter(raw_hits))
        return [{"canonical": only, "members": [only], "hits": raw_hits[only]}]

    ai = cfg["ai"]
    base_url = ai["base_url"].rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    prompt = _build_merge_prompt(game_id, list(raw_hits.keys()))
    messages = [{"role": "user", "content": prompt}]

    last_err = None
    for attempt in range(3):
        try:
            text, _ = await call_gemini_native(
                messages=messages, api_key=ai["api_key"], base_url=base_url,
                model=ai["model"], temperature=ai.get("temperature", 0.2),
                max_tokens=ai.get("max_tokens", 50000),
            )
            break
        except Exception as e:
            last_err = e
            wait = 2 ** (attempt + 1)
            print(f"  [merge retry] attempt {attempt + 1}/3 failed ({e}); sleep {wait}s")
            await asyncio.sleep(wait)
    else:
        raise RuntimeError(f"merge LLM failed after 3 retries: {last_err}")

    parsed = parse_judge_json(text)
    groups_out: list[dict] = []
    assigned: set[str] = set()
    for g in parsed.get("groups", []):
        members = [m for m in g.get("members", []) if m in raw_hits and m not in assigned]
        if not members:
            continue
        assigned.update(members)
        canonical = g.get("canonical") or members[0]
        hits = sum(raw_hits[m] for m in members)
        groups_out.append({"canonical": canonical, "members": members, "hits": hits})
    # safety net: orphans become singletons
    for r in raw_hits:
        if r not in assigned:
            groups_out.append({"canonical": r, "members": [r], "hits": raw_hits[r]})
    groups_out.sort(key=lambda g: (-g["hits"], g["canonical"]))
    return groups_out


def _build_safe_to_real_map() -> dict[str, str]:
    detailed = load_json(DETAILED_RULES_PATH)["games"]
    return {gid.replace("/", "_").replace(":", "_"): gid for gid in detailed.keys()}


async def merge_rules_all_batches(cfg: dict, force: bool = False) -> int:
    """Scan rule_extraction_results/<batch>/<game>/summary.json,
    call LLM to merge raw delta_rule_hits, persist rules_merged.json.
    Returns number of games processed."""
    safe_to_real = _build_safe_to_real_map()
    n_done = 0
    for batch_dir in sorted(p for p in DEFAULT_OUT_ROOT.iterdir() if p.is_dir()):
        for game_dir in sorted(p for p in batch_dir.iterdir() if p.is_dir()):
            s_path = game_dir / "summary.json"
            if not s_path.exists():
                continue
            gsum = load_json(s_path)
            raw = gsum.get("delta_rule_hits_raw") or gsum.get("delta_rule_hits") or {}
            if not raw:
                continue
            merged_path = game_dir / "rules_merged.json"
            if merged_path.exists() and not force:
                print(f"[merge skip-existing] {batch_dir.name} / {game_dir.name}")
                continue
            game_id = safe_to_real.get(game_dir.name, game_dir.name)
            print(f"[merge] {batch_dir.name} / {game_dir.name}  raw_rules={len(raw)} ...")
            try:
                groups = await merge_rules_for_game(game_id, raw, cfg)
            except Exception as e:
                print(f"  [FAIL] {e}")
                continue
            merged_path.write_text(
                json.dumps({"game_id": game_id, "groups": groups}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  raw={len(raw)} -> merged={len(groups)}")
            n_done += 1
    return n_done


def write_global_index() -> dict:
    """Aggregate every <batch>/summary.json into rule_extraction_results/_index.json."""
    index = {"scope": "global", "batches": []}
    batch_summaries = []
    for batch_dir in sorted(p for p in DEFAULT_OUT_ROOT.iterdir() if p.is_dir()):
        s_path = batch_dir / "summary.json"
        if not s_path.exists():
            continue
        bsum = load_json(s_path)
        batch_summaries.append(bsum)
        index["batches"].append({
            "batch_id": bsum.get("batch_id", batch_dir.name),
            "n_games": bsum.get("n_games", 0),
            "n_runs": bsum.get("n_runs", 0),
            "total_rounds": bsum.get("total_rounds", 0),
            "rounds_with_evidence": bsum.get("rounds_with_evidence", 0),
            "total_evidences": bsum.get("total_evidences", 0),
            "coverage": bsum.get("coverage", 0.0),
        })
    if batch_summaries:
        index["aggregate"] = fold(batch_summaries)
    (DEFAULT_OUT_ROOT / "_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", help="game id, e.g. Taxi-v3 (required unless --scan / --rebuild-summaries)")
    ap.add_argument(
        "--conversation",
        action="append",
        default=None,
        help="path to conversation.json (may be repeated)",
    )
    ap.add_argument(
        "--scan",
        default=None,
        help="path to a batch_results/<run_id> directory; auto-discover all conversation.json, "
             "skip games whose detailed_game_rules.json entry is empty, write per-config/game/batch summaries.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="with --scan: skip LLM call for runs that already have rule_extraction.json",
    )
    ap.add_argument(
        "--rebuild-summaries",
        action="store_true",
        help="skip LLM; only recompute per-config/game/batch summary.json from existing rule_extraction.json. "
             "Use with --scan <batch> to target one batch, or with no path to rebuild every batch under rule_extraction_results/.",
    )
    ap.add_argument(
        "--merge-rules",
        action="store_true",
        help="for every <batch>/<game> with raw delta_rule_hits, call LLM to group semantically equivalent rules "
             "and persist rules_merged.json. Idempotent (skips games that already have rules_merged.json unless --force-merge).",
    )
    ap.add_argument(
        "--force-merge",
        action="store_true",
        help="with --merge-rules: redo merging even if rules_merged.json already exists",
    )
    ap.add_argument(
        "--no-auto-merge",
        action="store_true",
        help="with --scan: skip the auto rule-merge step at the end of scan",
    )
    ap.add_argument(
        "--out",
        default=None,
        help=(
            "output JSON path; default mirrors the conversation's batch_results sub-path "
            "under rule_extraction_results/ (kept out of .gitignore so analysis can be committed). "
            "When multiple --conversation are passed, --out is ignored in favour of the default mirror layout."
        ),
    )
    ap.add_argument(
        "--rebuild-views",
        action="store_true",
        help="skip LLM call; re-render ai_responses.md + viewer.html from existing rule_extraction.json",
    )
    args = ap.parse_args()

    cfg = load_json(CONFIG_PATH)

    # ----- Mode: merge raw delta_rule texts via LLM, then rebuild summaries -----
    if args.merge_rules:
        n = asyncio.run(merge_rules_all_batches(cfg, force=args.force_merge))
        print(f"[merge] processed {n} games; now rebuilding summaries to apply merge maps ...")
        for batch_dir in sorted(p for p in DEFAULT_OUT_ROOT.iterdir() if p.is_dir()):
            try:
                write_summaries_for_batch(batch_dir.name)
            except Exception as e:
                print(f"  [FAIL] {batch_dir.name}: {e}")
        write_global_index()
        print(f"  -> {DEFAULT_OUT_ROOT / '_index.json'}")
        return

    # ----- Mode: rebuild summaries only (no LLM) -----
    if args.rebuild_summaries:
        if args.scan:
            batch_id = Path(args.scan).resolve().name
            print(f"[rebuild-summaries] {batch_id}")
            bsum = write_summaries_for_batch(batch_id)
            print(f"  batch summary: {bsum['n_runs']} runs / "
                  f"{bsum['rounds_with_evidence']}/{bsum['total_rounds']} rounds w/ ev "
                  f"(coverage {bsum['coverage']:.2%})")
        else:
            for batch_dir in sorted(p for p in DEFAULT_OUT_ROOT.iterdir() if p.is_dir()):
                print(f"[rebuild-summaries] {batch_dir.name}")
                try:
                    write_summaries_for_batch(batch_dir.name)
                except Exception as e:
                    print(f"  [FAIL] {e}")
        write_global_index()
        print(f"  -> wrote {DEFAULT_OUT_ROOT / '_index.json'}")
        return

    # ----- Mode: scan a batch directory -----
    if args.scan:
        batch_dir = Path(args.scan).resolve()
        if not batch_dir.exists():
            ap.error(f"--scan path does not exist: {batch_dir}")
        batch_id = batch_dir.name
        all_items = discover_from_batch_summary(batch_dir)
        if not all_items:
            ap.error(f"no conversations found under {batch_dir}")
        detailed_map = load_json(DETAILED_RULES_PATH)["games"]
        kept, skipped_empty, skipped_existing = [], [], []
        for game_id, config_name, run_index, conv in all_items:
            if not (detailed_map.get(game_id) or "").strip():
                skipped_empty.append((game_id, config_name, run_index))
                continue
            if args.skip_existing and rule_extraction_path_for(conv).exists():
                skipped_existing.append((game_id, config_name, run_index))
                continue
            kept.append((game_id, config_name, run_index, conv))

        print(f"[scan] batch={batch_id}  total={len(all_items)}  "
              f"to_run={len(kept)}  skip_empty_detailed={len(skipped_empty)}  "
              f"skip_existing={len(skipped_existing)}")
        if skipped_empty:
            from collections import Counter
            cnt = Counter(t[0] for t in skipped_empty)
            for g, n in cnt.most_common():
                print(f"  [skip empty detailed_rules] {g}  ({n} runs)")

        async def _run_scan():
            for game_id, config_name, run_index, conv in kept:
                tag = f"{game_id} / {config_name} / run_{run_index}"
                try:
                    parsed = await run_one(game_id, conv, cfg)
                except Exception as e:
                    print(f"  [FAIL llm] {tag}: {e}")
                    continue
                try:
                    out_path = rule_extraction_path_for(conv)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
                    standard, detailed = load_rules(game_id)
                    rounds_data = extract_ai_responses(conv)
                    write_ai_responses_md(rounds_data, out_path.parent / "ai_responses.md", game_id, conv)
                    write_viewer_html(rounds_data, parsed, out_path.parent / "viewer.html",
                                      game_id, conv, standard, detailed)
                except Exception as e:
                    print(f"  [FAIL write] {tag}: {e}")

        asyncio.run(_run_scan())

        # First pass: write summaries so per-game delta_rule_hits exists for the merger to consume.
        print(f"\n[scan] writing initial summaries for {batch_id} ...")
        write_summaries_for_batch(batch_id)

        # Auto-merge: idempotent — only games without rules_merged.json get a LLM call.
        if not args.no_auto_merge:
            print("[scan] running auto rule-merge (idempotent) ...")
            try:
                n = asyncio.run(merge_rules_all_batches(cfg, force=False))
                if n:
                    print(f"[scan] merged {n} new game(s); re-applying merge maps to summaries ...")
            except Exception as e:
                print(f"  [FAIL auto-merge] {e}")

        # Second pass picks up the new merge maps (no-op if nothing was merged).
        bsum = write_summaries_for_batch(batch_id)
        write_global_index()
        print(f"  batch: n_games={bsum['n_games']}  n_runs={bsum['n_runs']}  "
              f"coverage={bsum['coverage']:.2%}  "
              f"avg_ev/round={bsum['avg_evidences_per_round']:.3f}")
        print(f"  -> {DEFAULT_OUT_ROOT / batch_id / 'summary.json'}")
        print(f"  -> {DEFAULT_OUT_ROOT / '_index.json'}")
        return

    # ----- Mode: rebuild views only (no LLM, needs --game --conversation) -----
    if args.rebuild_views:
        if not args.game or not args.conversation:
            ap.error("--rebuild-views requires --game and --conversation")
        for conv in args.conversation:
            conv_path = Path(conv)
            out_path = Path(args.out) if args.out else default_output_path(conv_path)
            if not out_path.exists():
                print(f"  [skip] no existing {out_path}")
                continue
            parsed = load_json(out_path)
            standard, detailed = load_rules(args.game)
            rounds_data = extract_ai_responses(conv_path)
            md_path = out_path.parent / "ai_responses.md"
            html_path = out_path.parent / "viewer.html"
            write_ai_responses_md(rounds_data, md_path, args.game, conv_path)
            write_viewer_html(rounds_data, parsed, html_path, args.game, conv_path, standard, detailed)
            print(f"  -> wrote {md_path}")
            print(f"  -> wrote {html_path}")
        return

    # ----- Mode: explicit --game --conversation list (original behaviour) -----
    if not args.game or not args.conversation:
        ap.error("must pass either --scan <batch>, --rebuild-summaries, or --game and --conversation")

    async def _run_all():
        results = []
        for conv in args.conversation:
            conv_path = Path(conv)
            try:
                parsed = await run_one(args.game, conv_path, cfg)
            except Exception as e:
                print(f"  [FAIL] {conv_path}: {e}")
                results.append((str(conv_path), {"error": str(e)}))
                continue
            try:
                out_path = Path(args.out) if args.out else default_output_path(conv_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=2)
                standard, detailed = load_rules(args.game)
                rounds_data = extract_ai_responses(conv_path)
                md_path = out_path.parent / "ai_responses.md"
                html_path = out_path.parent / "viewer.html"
                write_ai_responses_md(rounds_data, md_path, args.game, conv_path)
                write_viewer_html(rounds_data, parsed, html_path, args.game, conv_path, standard, detailed)
                print(f"  -> wrote {out_path}")
                print(f"  -> wrote {md_path}")
                print(f"  -> wrote {html_path}")
                results.append((str(conv_path), parsed["summary"]))
            except Exception as e:
                print(f"  [FAIL] writing outputs for {conv_path}: {e}")
                results.append((str(conv_path), {"error": str(e)}))
        return results

    results = asyncio.run(_run_all())

    print("\n========== AGGREGATE ==========")
    agg = {"runs": len(results), "succeeded": 0, "failed": 0, "rounds": 0, "rounds_with_evidence": 0, "total_evidences": 0}
    for path, s in results:
        if "error" in s:
            agg["failed"] += 1
            print(f"  {path}:  ERROR {s['error']}")
            continue
        agg["succeeded"] += 1
        agg["rounds"] += s["total_rounds"]
        agg["rounds_with_evidence"] += s["rounds_with_evidence"]
        agg["total_evidences"] += s["total_evidences"]
        print(f"  {path}:  {s}")
    print(f"\n  TOTAL: {agg}")


if __name__ == "__main__":
    main()
