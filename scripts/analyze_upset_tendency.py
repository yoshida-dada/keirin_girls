"""万車券レース(三連単払戻>=10000円)の決着傾向を ペース×クラス で分析し、買い目のポイントを探す。

仮説(ユーザー): ハイペース想定→ライン崩れ&番手/三番手/単騎が突っ込む、スロー想定→自力(先行)が残る。
A3(チャレンジ)は上位/下位の実力差が大きく傾向が異なるかも。

DBのみ(モデル不要)で:
  ペース = レース内 b_count(主導権回数, 相対)最多の40%以上の人数 → ハイ(>=4)/中(3)/スロー(<=2)
  クラス = レース出走の最頻 class_rank を S(SS/S1/S2)/A1/A2/A3 に集約
  各セル(ペース×クラス)で 万車券率(実払戻>=1万の割合) と、万車券時の勝者/2着の
    ライン位置(先頭/番手/三番手/単騎)・脚質(逃/両/追)・ライン決着率(1-2着同一ライン) を集計。

  PYTHONIOENCODING=utf-8 python scripts/analyze_upset_tendency.py           # 男子
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR

MAN = 10000


def _load(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    leg, cls = {}, {}
    for rid, car, lt, cr in c.execute("SELECT race_id,car_number,leg_type,class_rank FROM entries"):
        leg[(rid, car)] = lt
        cls.setdefault(rid, []).append(cr)
    bc = defaultdict(dict)
    for rid, car, b in c.execute("SELECT race_id,car_number,b_count FROM recent_form"):
        bc[rid][car] = b or 0
    pay = {}
    for rid, combo, p in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta"):
        pay[rid] = (tuple(int(x) for x in combo.split("-")), p)
    tmp = defaultdict(list)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        tmp[rid].append((li, pi, car))
    lines = {}
    for rid, rows in tmp.items():
        mx = max(li for li, _, _ in rows)
        ls = [[] for _ in range(mx + 1)]
        for li, pi, car in sorted(rows, key=lambda x: (x[0], x[1])):
            ls[li].append(car)
        lines[rid] = [x for x in ls if x]
    races = [r[0] for r in c.execute("SELECT race_id FROM races WHERE field_size=7")]
    c.close()
    return leg, cls, bc, pay, lines, races


def _cls_band(crs):
    m = Counter(crs).most_common(1)[0][0] if crs else None
    if m in ("SS", "S1", "S2"):
        return "S級"
    return m  # A1/A2/A3


def _pace(cars):
    mx = max(cars.values()) if cars else 0
    n = sum(1 for v in cars.values() if mx >= 2 and v >= 0.4 * mx)
    return "ハイ" if n >= 4 else ("中" if n == 3 else "スロー")


def _line_of(car, lines):
    for m in lines or []:
        if car in m:
            return m
    return [car]


def _pos(car, lines):
    m = _line_of(car, lines)
    if len(m) == 1:
        return "単騎"
    i = m.index(car)
    return "先頭" if i == 0 else ("番手" if i == 1 else "三番手+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    args = ap.parse_args()
    leg, cls, bc, pay, lines, races = _load(args.db)

    cells = defaultdict(lambda: {"n": 0, "man": 0,
                                 "wpos": Counter(), "wleg": Counter(),
                                 "spos": Counter(), "same_line": 0})
    for rid in races:
        if rid not in pay:
            continue
        cars = bc.get(rid, {})
        if len(cars) != 7:
            continue
        pace = _pace(cars)
        cl = _cls_band(cls.get(rid, []))
        key = (pace, cl)
        cell = cells[key]
        cell["n"] += 1
        combo, p = pay[rid]
        if p < MAN:
            continue
        cell["man"] += 1
        w, s = combo[0], combo[1]
        ln = lines.get(rid)
        cell["wpos"][_pos(w, ln)] += 1
        cell["wleg"][leg.get((rid, w))] += 1
        cell["spos"][_pos(s, ln)] += 1
        cell["same_line"] += int(w in _line_of(s, ln))    # 1-2着同一ライン

    def pct(cnt, tot):
        return f"{cnt/tot*100:.0f}%" if tot else "-"

    print(f"万車券(三連単>=1万)レースの決着傾向 ペース×クラス  DB={Path(args.db).stem}\n")
    order_cls = ["S級", "A1", "A2", "A3"]
    for pace in ["スロー", "中", "ハイ"]:
        for cl in order_cls:
            c = cells.get((pace, cl))
            if not c or c["n"] < 200:
                continue
            manr = c["man"] / c["n"] * 100
            m = c["man"]
            wp = c["wpos"]; wl = c["wleg"]
            print(f"■ {pace}ペース × {cl}  R数{c['n']}  万車券率{manr:.1f}%(n={m})")
            if m >= 40:
                print(f"    勝者位置: 先頭{pct(wp['先頭'],m)} 番手{pct(wp['番手'],m)} "
                      f"三番手+{pct(wp['三番手+'],m)} 単騎{pct(wp['単騎'],m)}")
                print(f"    勝者脚質: 逃{pct(wl.get('逃',0),m)} 両{pct(wl.get('両',0),m)} 追{pct(wl.get('追',0),m)}"
                      f"  ／ 1-2着同一ライン{pct(c['same_line'],m)}")
        print()


if __name__ == "__main__":
    main()
