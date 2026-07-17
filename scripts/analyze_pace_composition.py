"""メンバー構成(先行型の数)→ 展開(ペース)→ 勝ち決まり手 の相関を検証。

仮説: 先行型(前に行きたい選手)が多いレースは主導権争いでハイペース化し、逃げが飛んで
差し/捲りが決まりやすい。逆に先行型が少ないと単騎逃げが残りやすい(逃げ/番手有利)。

girls は脚質がほぼ「両」で leg_type が使えないため、先行度は recent_form の
b_count(バック=主導権を取った回数, as-of) で測る。各レースの「先行型の数」= b_count>=閾値の人数。
それに対し 勝者の決まり手(逃/捲/差) と、2着の決まり手(ク=マーク2着 の増減) を見る。

  PYTHONIOENCODING=utf-8 python scripts/analyze_pace_composition.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--thr", type=int, default=3, help="先行型とみなす b_count 閾値")
    args = ap.parse_args()
    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    bc = defaultdict(dict)     # rid -> {car: b_count}
    for rid, car, b in c.execute("SELECT race_id,car_number,b_count FROM recent_form"):
        bc[rid][car] = b or 0
    win_km = {}                # rid -> 勝者の決まり手
    for rid, km in c.execute("SELECT race_id,kimarite FROM results WHERE position=1 AND kimarite IS NOT NULL AND kimarite!=''"):
        win_km[rid] = km
    races = [r[0] for r in c.execute("SELECT race_id FROM races WHERE field_size=7")]
    c.close()

    # 先行型の数 → 勝ち決まり手分布
    by_n = defaultdict(lambda: defaultdict(int))    # n_front -> {km: count}
    for rid in races:
        km = win_km.get(rid)
        if km is None:
            continue
        cars = bc.get(rid, {})
        if len(cars) != 7:
            continue
        n_front = sum(1 for v in cars.values() if v >= args.thr)
        by_n[n_front][km] += 1

    print(f"先行型(b_count>={args.thr})の人数 → 勝ち決まり手の割合\n")
    print(f"  {'先行型数':>7}{'R数':>6}{'逃%':>8}{'捲%':>8}{'差%':>8}")
    for n in sorted(by_n):
        d = by_n[n]
        tot = sum(d.values())
        if tot < 30:
            continue
        nige = d.get("逃", 0) / tot * 100
        maku = d.get("捲", 0) / tot * 100
        sashi = d.get("差", 0) / tot * 100
        print(f"  {n:>7}{tot:>6}{nige:>8.1f}{maku:>8.1f}{sashi:>8.1f}")

    # まとめ: 先行型「少(<=2) vs 多(>=4)」で比較
    def agg(cond):
        d = defaultdict(int)
        for n, dd in by_n.items():
            if cond(n):
                for k, v in dd.items():
                    d[k] += v
        return d
    lo, hi = agg(lambda n: n <= 2), agg(lambda n: n >= 4)
    print("\n【先行型 少(≤2) vs 多(≥4)】勝ち決まり手")
    for lbl, d in (("少≤2", lo), ("多≥4", hi)):
        tot = sum(d.values())
        if tot:
            print(f"  {lbl}(n={tot}): 逃{d.get('逃',0)/tot*100:.1f}% 捲{d.get('捲',0)/tot*100:.1f}% 差{d.get('差',0)/tot*100:.1f}%")


if __name__ == "__main__":
    main()
