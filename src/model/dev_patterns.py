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

def _pct(x: float) -> str:
    return f"{round(x * 100)}%"


def _w_avg(rows: list, get) -> float:
    """表示中シナリオの発生確率で加重平均する。"""
    tot = sum(r["prob"] for r in rows)
    if not tot:
        return 0.0
    return sum(r["prob"] * (get(r) or 0) for r in rows) / tot


def _w_dist(rows: list, key: str) -> dict[str, float]:
    """表示中シナリオの確率加重で {印: 割合} を合成する（未フィルタの生分布を使う）。"""
    tot = sum(r["prob"] for r in rows)
    if not tot:
        return {}
    out = {m: 0.0 for m in MARKS}
    for r in rows:
        w = r["prob"] / tot
        for m, p in (r["_raw"][key] or {}).items():
            if m in out:
                out[m] += w * (p or 0)
    return out


def build_insights(top: list, allrows: list, cars: dict[str, str]) -> list[str]:
    """「紐選びの読み」をレースごとに組み立てる。

    top     : 画面に出す上位シナリオ。①の注意書きや3着加重はこれに即して書く
              （画面に無い形の話をしない＝表示との齟齬を防ぐ）。
    allrows : 全6形。「◎が負ける形では〜」のように一般化する文はこちらで加重する
              （上位だけで平均すると④を落として楽観側に振れるため）。
    """
    win = [r for r in top if r["fav_wins"]]
    lose_all = [r for r in allrows if not r["fav_wins"]]
    ins: list[str] = []
    maru, marum, ana_car = cars.get("◎", "◎"), cars.get("○", "○"), cars.get("×", "")

    if win:
        vals = [(r["_raw"]["second"] or {}).get("○", 0) for r in win]
        rng = (f"{_pct(min(vals))}〜{_pct(max(vals))}" if len(vals) > 1 and
               round(min(vals) * 100) != round(max(vals) * 100) else _pct(vals[0]))
        ins.append(f"◎{maru}が勝つ形（上位{len(win)}つ）では、2着は○{marum}が{rng}で最有力。"
                   f"◎-○の1-2着が軸になる。")
    if lose_all:
        wm = _w_avg(lose_all, lambda r: (r["_raw"].get("winner") or {}).get("○", 0))
        ft = _w_avg(lose_all, lambda r: r.get("fav_top3", 0))
        ins.append(f"◎{maru}が負ける形では勝ち頭は○{marum}が{_pct(wm)}。"
                   f"ただしその時も◎は3着以内に{_pct(ft)}残る＝◎を切る根拠にならない"
                   f"（◎フェードは過去検証でも不成立）。")

    # 最も割れる3着＝穴を拾う場所。表示中シナリオの加重で具体名を出す。
    d3 = _w_dist(top, "third")
    if d3:
        best = sorted(d3.items(), key=lambda kv: -kv[1])[:3]
        names = " / ".join(f"{m}{cars.get(m,'')} {_pct(p)}" for m, p in best if p >= 0.05)
        ins.append(f"最も割れるのは3着。上位3形の加重では {names}。"
                   f"穴を拾うならここで、{ana_car}が{_pct(d3.get('×', 0))}を占める。")

    # ①が上位に入っている時だけ触れる
    esc = next((r for r in top if r["key"].startswith("①") and not r["key"].startswith("①'")), None)
    if esc:
        x2 = (esc["_raw"]["second"] or {}).get("×", 0)
        ins.append(f"①◎逃げ切り（{_pct(esc['prob'])}）は2着に人気薄が{_pct(x2)}と"
                   f"他の形（11〜13%）より荒れる。ここだけは3着だけでなく2着も広げたい。")
    return ins


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
               "third": _dist_list(s.get("third"), cars),
               # 読みの合成に使う未フィルタの生分布（出力前に落とす）
               "_raw": {"second": s.get("second"), "third": s.get("third"),
                        "winner": s.get("winner")}}
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
    top = rows[:top_k]
    insights = build_insights(top, rows, cars)
    # 円グラフ用: 全6形を①〜⑥の定義順で（上位3だけでは全体像が見えないため）。
    order = {k: i for i, k in enumerate(
        ["①◎逃げ切り", "②◎捲り", "③◎差し", "④別の逃げ残り", "⑤捲り台頭", "⑥差し/マーク"])}
    allrows = sorted(rows, key=lambda r: order.get(r["key"], 99))
    for r in rows:                     # 生分布は読みの合成にだけ使う（出力には載せない）
        r.pop("_raw", None)
    return {"top": top, "insights": insights,
            "all": [{"key": r["key"], "prob": r["prob"], "fav_wins": r["fav_wins"]}
                    for r in allrows],
            "n_races": st.get("n_races"), "pace": _pace_key(pace_level),
            "note": "確率＝このレースの1着確率×履歴の分岐比（ペース区分で条件付け）。"
                    "紐の内訳は全レースのプール値。◎判定に学習データを含むため楽観側で、"
                    "分岐の目安として読む（回収率の根拠にはならない）。"}
