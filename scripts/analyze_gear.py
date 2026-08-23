"""ギア倍数の調査（男子）: A)ギア×並び位置×成績  B)開催内の前日比ギア変更×成績。

ギアは既にPL_FEATURES_FULLの生特徴。ここでは「並び位置別のギア効果」と「開催中のギア変更効果」が
生ギアを超える信号を持つかを記述統計で確認（能力交絡は競走得点併記でチェック）。有望なら別途as-of
walk-forwardで純増検証する。並び位置=narabi(pos_in_line, 単騎=ライン人数1)、成績=results(着順)。

  PYTHONIOENCODING=utf-8 python scripts/analyze_gear.py
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


def _load():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    gear = {}      # (rid,car) -> gear
    score = {}     # (rid,car) -> racing_score
    for rid, car, g, s in c.execute(
            "SELECT race_id,car_number,gear_ratio,racing_score FROM entries"):
        if g is not None:
            gear[(rid, car)] = g
        score[(rid, car)] = s
    pos = {}       # (rid,car) -> pos_in_line ; size per line
    linesz = defaultdict(lambda: defaultdict(int))
    tmp = defaultdict(list)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        pos[(rid, car)] = (li, pi)
        linesz[rid][li] += 1
    fin = {}       # (rid,car) -> 着順
    for rid, car, p in c.execute("SELECT race_id,car_number,position FROM results"):
        if p is not None:
            fin[(rid, car)] = p
    c.close()
    return gear, score, pos, linesz, fin


def _posclass(rid, car, pos, linesz):
    pc = pos.get((rid, car))
    if pc is None:
        return None
    li, pi = pc
    if linesz[rid].get(li, 1) == 1:
        return "単騎"
    if pi == 0:
        return "先頭"
    if pi == 1:
        return "番手"
    return "三番手以降"


def main():
    gear, score, pos, linesz, fin = _load()
    keys = [k for k in gear if k in fin]
    print(f"男子: gear×着順の結合レコード {len(keys):,}\n")

    # 全体のギア分布→3バンド(低/中/高)を分位で
    gs = sorted(gear[k] for k in keys)
    q1, q2 = gs[len(gs) // 3], gs[2 * len(gs) // 3]
    def band(g): return "低" if g < q1 else ("中" if g < q2 else "高")
    print(f"ギアバンド: 低<{q1} / 中<{q2} / 高≥{q2}\n")

    # A) 並び位置 × ギアバンド → 1着率/top3率/平均得点(交絡確認)
    print("A) 並び位置 × ギア → 成績（平均得点は能力交絡の確認用）")
    agg = defaultdict(lambda: {"n": 0, "win": 0, "top3": 0, "sc": 0.0})
    for k in keys:
        pcl = _posclass(k[0], k[1], pos, linesz)
        if pcl is None:
            continue
        d = agg[(pcl, band(gear[k]))]
        d["n"] += 1
        d["win"] += int(fin[k] == 1)
        d["top3"] += int(fin[k] <= 3)
        d["sc"] += score.get(k) or 0
    print(f"   {'位置':<10}{'ギア':<5}{'n':>7}{'1着率':>8}{'top3率':>8}{'平均得点':>9}")
    for pcl in ["先頭", "番手", "三番手以降", "単騎"]:
        for b in ["低", "中", "高"]:
            d = agg.get((pcl, b))
            if not d or d["n"] == 0:
                continue
            print(f"   {pcl:<10}{b:<5}{d['n']:>7}{d['win']/d['n']*100:>7.1f}%"
                  f"{d['top3']/d['n']*100:>7.1f}%{d['sc']/d['n']:>9.1f}")
        print()

    # A2) 交絡を割るため得点帯を固定してギア効果を見る（先頭のみ・得点85-95）
    print("A2) 先頭・得点85〜95に固定した中でのギア効果（交絡除去の確認）")
    sub = defaultdict(lambda: {"n": 0, "win": 0, "top3": 0})
    for k in keys:
        if _posclass(k[0], k[1], pos, linesz) != "先頭":
            continue
        sc = score.get(k)
        if sc is None or not (85 <= sc <= 95):
            continue
        d = sub[band(gear[k])]
        d["n"] += 1; d["win"] += int(fin[k] == 1); d["top3"] += int(fin[k] <= 3)
    for b in ["低", "中", "高"]:
        d = sub.get(b)
        if d and d["n"]:
            print(f"   ギア{b}: n{d['n']:>6} 1着率{d['win']/d['n']*100:.1f}% top3率{d['top3']/d['n']*100:.1f}%")
    print()

    # B) 開催内・前日比ギア変更 × 成績（同一選手, 直近前走比）
    print("B) 開催内の前日比ギア変更 × 成績（同一開催・同一選手の連続走）")
    # 選手ごとに(開催キー, 日, gear, 着順)を集め、日順で連続ペアを作る
    byrider = defaultdict(list)  # rider(car名代わりにcar番号ではダメ→氏名) : use (rid,car) meta
    # 氏名で追う: entriesから名前を引く
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    name = {(rid, car): nm for rid, car, nm in
            c.execute("SELECT race_id,car_number,rider_name FROM entries")}
    c.close()
    for k in keys:
        rid, car = k
        meet = rid[:10]                      # 会場2+初日8
        day = int(rid[10:12])
        nm = name.get(k)
        if nm:
            byrider[(nm, meet)].append((day, gear[k], fin[k], score.get(k)))
    ch = defaultdict(lambda: {"n": 0, "win": 0, "top3": 0, "sumfin": 0, "base": 0.0})
    careerfin = defaultdict(list)
    for (nm, meet), rows in byrider.items():
        for _, _, f, _ in rows:
            careerfin[nm].append(f)
    cbase = {nm: sum(v) / len(v) for nm, v in careerfin.items() if v}
    for (nm, meet), rows in byrider.items():
        rows.sort()
        for a, b in zip(rows, rows[1:]):
            if b[0] - a[0] != 1:             # 連続日のみ
                continue
            dg = round(b[1] - a[1], 2)
            key = "上げた" if dg > 0 else ("下げた" if dg < 0 else "据置")
            d = ch[key]
            d["n"] += 1; d["win"] += int(b[2] == 1); d["top3"] += int(b[2] <= 3)
            d["sumfin"] += b[2]; d["base"] += cbase.get(nm, b[2])
    print(f"   {'変更':<6}{'n':>7}{'翌日1着率':>9}{'翌日top3率':>10}{'翌日平均着':>10}{'通算平均着':>10}{'残差':>7}")
    for key in ["上げた", "据置", "下げた"]:
        d = ch.get(key)
        if not d or d["n"] == 0:
            continue
        af = d["sumfin"] / d["n"]; bf = d["base"] / d["n"]
        print(f"   {key:<6}{d['n']:>7}{d['win']/d['n']*100:>8.1f}%{d['top3']/d['n']*100:>9.1f}%"
              f"{af:>10.2f}{bf:>10.2f}{bf-af:>+7.2f}")
    print("\n   ※残差>0＝通算平均着より上位(良化)。前日不振→変更で良化するかは次段(前日着別)で。")

    # B2) 前日の着順別に、翌日ギア変更の効果（前日不振ほど変えるか＋変えて良化するか）
    print("\nB2) 前日着順 × 翌日ギア変更 → 翌日top3率")
    b2 = defaultdict(lambda: {"n": 0, "top3": 0})
    for (nm, meet), rows in byrider.items():
        rows.sort()
        for a, b in zip(rows, rows[1:]):
            if b[0] - a[0] != 1:
                continue
            prevband = "前日1-3着" if a[2] <= 3 else ("前日4-6着" if a[2] <= 6 else "前日7着-")
            dg = b[1] - a[1]
            key = "上げ" if dg > 0 else ("下げ" if dg < 0 else "据置")
            d = b2[(prevband, key)]
            d["n"] += 1; d["top3"] += int(b[2] <= 3)
    for pb in ["前日1-3着", "前日4-6着", "前日7着-"]:
        line = f"   {pb:<9}"
        for key in ["上げ", "据置", "下げ"]:
            d = b2.get((pb, key))
            line += f" {key}:{(d['top3']/d['n']*100 if d and d['n'] else 0):.0f}%(n{d['n'] if d else 0})"
        print(line)


if __name__ == "__main__":
    main()
