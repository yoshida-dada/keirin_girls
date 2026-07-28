"""展開6パターンの発生確率と紐構造を、1レースぶんの予測dictから組み立てる（表示用）。

パターン定義（◎=モデル1着本命、B=主導権＝バック先頭）:
  ◎勝 ①◎逃げ切り / ②◎捲り / ③◎差し(前崩れ)
  ◎負 ④別選手の逃げ残り / ⑤捲り台頭 / ⑥差し/マーク決着

確率 = そのレースの top1_prob（較正検証済み） × 履歴の分岐比（ペース区分で条件付け）。
紐の内訳（2着/3着の印分布・◎の着順・主導権の絡み）は全5794レースのプール値。
統計は `dev_pattern_stats.json`（scripts/analyze_dev_patterns.py --emit が生成）から読む。
**DBに触らない**ので refresh_predictions.py（DB非依存）からも安全に呼べる。

回収率の主張はできない（過去検証で全手法ROI<100%）。展開の読み・紐選びの参考情報。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

STATS_PATH = Path(__file__).with_name("dev_pattern_stats.json")
MARKS = ["◎", "○", "▲", "△", "×"]

# 表示する構造的な読み（実測に基づく。穴を拾う判断で最も効くのは3着の割れ）。
INSIGHTS = [
    "◎-○の1-2着ペアが全パターンの軸。◎が勝てば2着は○が42〜54%、"
    "◎が負ければ勝つのは○が5〜7割で、その時◎は2着に残りやすい。",
    "◎を切る理由がない。◎が負けても3着以内に残るのが約8割（④71% / ⑤81% / ⑥82%）。"
    "◎フェードが成立しないという過去検証と一致する。",
    "最も割れるのは3着（▲約30% / ○約27% / 人気薄23〜26%）。"
    "点数を割くならここ＝穴を拾うなら3着候補を広げるのが筋。",
    "①◎逃げ切りだけは2着に人気薄が16%と他パターンより荒れる（他は11〜13%）。",
]


@lru_cache(maxsize=1)
def load_stats() -> dict:
    try:
        return json.loads(STATS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pace_key(level: str) -> str:
    """development.pace.level（"ハイ" "ミドル〜ハイ" 等）→ 履歴の区分へ寄せる。"""
    s = str(level or "")
    if "ハイ" in s:
        return "ハイ"
    if "スロー" in s:
        return "スロー"
    return "ミドル"


def _marks_to_cars(riders: list) -> dict[str, str]:
    """1着確率降順の riders → {印: "車番氏名"}。5番手以降は × にまとめる。"""
    out: dict[str, str] = {}
    for i, m in enumerate(MARKS):
        if i < len(riders):
            r = riders[i] or {}
            out[m] = f"{r.get('car','')}{r.get('name','')}"
    if len(riders) > len(MARKS):
        rest = ",".join(str((r or {}).get("car", "")) for r in riders[len(MARKS) - 1:])
        out["×"] = f"人気薄({rest})"
    return out


def _dist_list(d: dict | None, cars: dict[str, str], floor: float = 0.05) -> list:
    """{印:割合} → [{mark, car, p}] を割合降順で。floor未満は落とす。"""
    if not d:
        return []
    items = [(m, p) for m, p in d.items() if p and p >= floor]
    items.sort(key=lambda x: -x[1])
    return [{"mark": m, "car": cars.get(m, ""), "p": round(p, 3)} for m, p in items]


def build_dev_patterns(top1_prob: float | None, pace_level: str,
                       riders: list, top_k: int = 3) -> dict | None:
    """上位 top_k の展開パターンを返す。統計・入力が無ければ None。"""
    st = load_stats()
    if not st or not riders or top1_prob is None:
        return None
    rates = (st.get("pace_rates") or {}).get(_pace_key(pace_level))
    pats = st.get("patterns") or {}
    if not rates or not pats:
        return None
    win_keys = st.get("win_patterns") or []

    # ◎勝ち群／◎負け群それぞれの中での構成比（分岐比）へ正規化する
    wsum = sum(rates.get(k, 0) for k in win_keys)
    lose_keys = [k for k in pats if k not in win_keys]
    lsum = sum(rates.get(k, 0) for k in lose_keys)
    p1 = float(top1_prob)
    cars = _marks_to_cars(riders)

    rows = []
    for key, s in pats.items():
        is_win = key in win_keys
        share = (rates.get(key, 0) / wsum if wsum else 0) if is_win else \
                (rates.get(key, 0) / lsum if lsum else 0)
        prob = (p1 if is_win else 1 - p1) * share
        if prob <= 0:
            continue
        row = {"key": key, "prob": round(prob, 4), "fav_wins": is_win, "n": s.get("n"),
               "second": _dist_list(s.get("second"), cars),
               "third": _dist_list(s.get("third"), cars)}
        if is_win:
            row["winner"] = [{"mark": "◎", "car": cars.get("◎", ""), "p": 1.0}]
            if s.get("b_second"):
                row["b_second"] = round(s["b_second"], 3)
        else:
            row["winner"] = _dist_list(s.get("winner"), cars)
            if s.get("b_win") is not None:
                row["b_win"] = round(s["b_win"], 3)
            fp = s.get("fav_pos") or {}
            if fp:
                row["fav_pos"] = {k: round(v, 3) for k, v in fp.items()}
                row["fav_top3"] = round(fp.get("2", 0) + fp.get("3", 0), 3)
        rows.append(row)

    rows.sort(key=lambda r: -r["prob"])
    return {"top": rows[:top_k], "insights": INSIGHTS,
            "n_races": st.get("n_races"), "pace": _pace_key(pace_level),
            "note": "確率＝このレースの1着確率×履歴の分岐比（ペース区分で条件付け）。"
                    "紐の内訳は全レースのプール値。◎判定に学習データを含むため楽観側で、"
                    "分岐の目安として読む（回収率の根拠にはならない）。"}
