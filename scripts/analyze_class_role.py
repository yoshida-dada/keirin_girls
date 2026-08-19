"""男子: クラス × 種別（×日目）で決着傾向を分析。展開分岐買い目の狙う選手をクラス別に精緻化する。

クラス: S級(SS/S1/S2) / A級(A1/A2) / チャレンジ(A3)。種別: 予選/準決勝/決勝/選抜/一般/特選。
先の知見(種別で先行/追込が変わる)がクラスでどう変わるか。特にチャレンジ(A3)は実力差が大きい。
勝者脚質(逃/両/追)・万車券率・勝者ライン位置(先頭/番手/単騎)を (クラス×種別) で見る。

  PYTHONIOENCODING=utf-8 python scripts/analyze_class_role.py
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


def _role(name):
    if not name:
        return "他"
    for kw in ["準決", "決勝", "予選", "選抜", "特選", "一般"]:
        if kw in name:
            return "準決勝" if kw == "準決" else kw
    return "他"


def _cls(crs):
    m = Counter(crs).most_common(1)[0][0] if crs else None
    if m in ("SS", "S1", "S2"):
        return "S級"
    if m in ("A1", "A2"):
        return "A級"
    if m == "A3":
        return "チャレンジ"
    return "他"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    args = ap.parse_args()
    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    role, cls = {}, defaultdict(list)
    for rid, nm, fs in c.execute("SELECT race_id,race_name,field_size FROM races"):
        if fs == 7:
            role[rid] = _role(nm)
    leg = {}
    for rid, car, lt, cr in c.execute("SELECT race_id,car_number,leg_type,class_rank FROM entries"):
        leg[(rid, car)] = lt
        cls[rid].append(cr)
    win, pay = {}, {}
    for rid, combo, p in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta"):
        win[rid] = int(combo.split("-")[0]); pay[rid] = p
    c.close()

    cell = defaultdict(lambda: {"n": 0, "man": 0, "leg": Counter()})
    for rid, r in role.items():
        if rid not in win:
            continue
        cl = _cls(cls.get(rid, []))
        s = cell[(cl, r)]
        s["n"] += 1
        s["leg"][leg.get((rid, win[rid]))] += 1
        if pay.get(rid, 0) >= MAN:
            s["man"] += 1

    print("男子7車: クラス × 種別 の決着傾向（狙う選手をクラス別に精緻化）\n")
    print(f"   {'クラス':<7}{'種別':<6}{'R数':>7}{'万車率':>8}{'勝者逃':>8}{'勝者両':>8}{'勝者追':>8}")
    for cl in ["S級", "A級", "チャレンジ"]:
        shown = False
        for r in ["予選", "準決勝", "決勝", "選抜", "一般", "特選"]:
            s = cell.get((cl, r))
            if not s or s["n"] < 80:
                continue
            m = s["n"]
            def pc(k): return f"{s['leg'].get(k,0)/m*100:.0f}%"
            print(f"   {cl:<7}{r:<6}{m:>7}{s['man']/m*100:>7.1f}%{pc('逃'):>8}{pc('両'):>8}{pc('追'):>8}")
            shown = True
        if shown:
            print()


if __name__ == "__main__":
    main()
