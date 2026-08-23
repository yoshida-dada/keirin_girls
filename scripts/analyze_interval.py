"""出走間隔（前走/前開催からの日数）× 成績の調査（男子7車）。

出走間隔=days_sinceは表示専用で特徴未使用。過去の選手ローリング(中何日含む)は線形で非採用
(ECE悪化)。ここでは「空きすぎ＝レース勘喪失」の非線形(U字)を確認する。能力交絡は
残差(通算平均着−当該着, +=通常より上位=良化)で除去。前開催からの間隔に注目するため
「開催初出走(前レースが別開催)」サブセットも別掲。

  PYTHONIOENCODING=utf-8 python scripts/analyze_interval.py
"""
from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR

DB = str(DATA_DIR / "keirin_men.sqlite")


def _adate(rid):
    return date(int(rid[2:6]), int(rid[6:8]), int(rid[8:10])) + timedelta(days=int(rid[10:12]) - 1)


def _bucket(g):
    if g <= 1:
        return "①1日(連続)"
    if g <= 6:
        return "②2-6日"
    if g <= 13:
        return "③7-13日"
    if g <= 20:
        return "④14-20日"
    if g <= 29:
        return "⑤21-29日"
    if g <= 44:
        return "⑥30-44日"
    if g <= 59:
        return "⑦45-59日"
    if g <= 89:
        return "⑧60-89日"
    return "⑨90日+"


ORDER = ["①1日(連続)", "②2-6日", "③7-13日", "④14-20日", "⑤21-29日",
         "⑥30-44日", "⑦45-59日", "⑧60-89日", "⑨90日+"]


def main():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    fs = {rid: n for rid, n in c.execute("SELECT race_id,field_size FROM races")}
    score = {(rid, car): s for rid, car, s in
             c.execute("SELECT race_id,car_number,racing_score FROM entries")}
    name = {(rid, car): nm for rid, car, nm in
            c.execute("SELECT race_id,car_number,rider_name FROM entries")}
    fin = {(rid, car): p for rid, car, p in
           c.execute("SELECT race_id,car_number,position FROM results") if p is not None}
    c.close()

    # 選手ごとに (date, rid, 着, 得点) を集約（7車のみ）
    byrider = defaultdict(list)
    for (rid, car), p in fin.items():
        if fs.get(rid) != 7:
            continue
        nm = name.get((rid, car))
        if nm:
            byrider[nm].append((_adate(rid), rid, p, score.get((rid, car))))
    # 通算平均着（能力ベースライン）
    cbase = {nm: sum(r[2] for r in rows) / len(rows) for nm, rows in byrider.items() if rows}

    agg = defaultdict(lambda: {"n": 0, "win": 0, "top3": 0, "sumfin": 0, "res": 0.0, "sc": 0.0})
    aggF = defaultdict(lambda: {"n": 0, "win": 0, "top3": 0, "sumfin": 0, "res": 0.0})  # 開催初出走
    for nm, rows in byrider.items():
        rows.sort()
        base = cbase[nm]
        for prev, cur in zip(rows, rows[1:]):
            gap = (cur[0] - prev[0]).days
            if gap <= 0:
                continue
            bk = _bucket(gap)
            f = cur[2]
            d = agg[bk]
            d["n"] += 1; d["win"] += int(f == 1); d["top3"] += int(f <= 3)
            d["sumfin"] += f; d["res"] += (base - f); d["sc"] += cur[3] or 0
            if cur[1][:10] != prev[1][:10]:              # 開催が変わった=前開催からの間隔
                e = aggF[bk]
                e["n"] += 1; e["win"] += int(f == 1); e["top3"] += int(f <= 3)
                e["sumfin"] += f; e["res"] += (base - f)

    print(f"男子7車: 出走間隔×成績（残差=通算平均着−当該着, +=通常より上位=良化）\n")
    print("【全レース】")
    print(f"   {'間隔':<12}{'n':>8}{'1着率':>8}{'top3率':>8}{'平均着':>8}{'残差':>8}{'平均得点':>9}")
    for bk in ORDER:
        d = agg.get(bk)
        if not d or d["n"] == 0:
            continue
        n = d["n"]
        print(f"   {bk:<12}{n:>8}{d['win']/n*100:>7.1f}%{d['top3']/n*100:>7.1f}%"
              f"{d['sumfin']/n:>8.2f}{d['res']/n:>+8.2f}{d['sc']/n:>9.1f}")

    print("\n【開催初出走のみ（=前開催からの間隔）】")
    print(f"   {'間隔':<12}{'n':>8}{'1着率':>8}{'top3率':>8}{'平均着':>8}{'残差':>8}")
    for bk in ORDER:
        d = aggF.get(bk)
        if not d or d["n"] == 0:
            continue
        n = d["n"]
        print(f"   {bk:<12}{n:>8}{d['win']/n*100:>7.1f}%{d['top3']/n*100:>7.1f}%"
              f"{d['sumfin']/n:>8.2f}{d['res']/n:>+8.2f}")
    print("\n※残差が間隔で単調でなくU字(短すぎ/長すぎで負)なら非線形の勘要素。"
          "平坦/得点交絡なら既存吸収。")


if __name__ == "__main__":
    main()
