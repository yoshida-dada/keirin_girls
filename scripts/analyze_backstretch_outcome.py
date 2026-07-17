"""ペース別に「主導権(B=最終バック先頭)を取った選手が勝ち切るか、垂れて2着以下か」を分析。

仮説: ハイペース(先行型が多い)ほど、主導権を取っても脚を使って垂れ、勝ち切れず2着以下に
なりやすい。スローなら主導権のまま押し切りやすい。
  ペース = レース内で b_count(recent,主導権回数)が最多の40%以上の人数（ペース読みと同一定義）
  B取得者 = results.sb に 'B' を含む一意の選手
  → ペース帯 × B取得者の着順(1着/2着/3着/着外) の割合。

  PYTHONIOENCODING=utf-8 python scripts/analyze_backstretch_outcome.py --db data/keirin.sqlite
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
    args = ap.parse_args()
    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    bc = defaultdict(dict)
    for rid, car, b in c.execute("SELECT race_id,car_number,b_count FROM recent_form"):
        bc[rid][car] = b or 0
    # B取得者と各選手の着順
    b_taker = {}
    pos = defaultdict(dict)
    for rid, car, p, sb in c.execute("SELECT race_id,car_number,position,sb FROM results"):
        pos[rid][car] = p
        if sb and "B" in sb:
            b_taker.setdefault(rid, []).append(car)
    b_taker = {rid: cs[0] for rid, cs in b_taker.items() if len(cs) == 1}
    races = [r[0] for r in c.execute("SELECT race_id FROM races WHERE field_size=7")]
    c.close()

    def pace_band(cars):
        mx = max(cars.values()) if cars else 0
        n = sum(1 for v in cars.values() if mx >= 2 and v >= 0.4 * mx)
        return ("スロー〜ミドル(≤2)" if n <= 2 else ("ミドル〜ハイ(3)" if n == 3 else "ハイ(≥4)")), n

    # ペース帯 × B取得者の着順
    tab = defaultdict(lambda: defaultdict(int))
    overall = defaultdict(int)
    for rid in races:
        bt = b_taker.get(rid)
        cars = bc.get(rid, {})
        if bt is None or len(cars) != 7:
            continue
        p = pos.get(rid, {}).get(bt)
        if p is None:
            continue
        band, _ = pace_band(cars)
        key = 1 if p == 1 else (2 if p == 2 else (3 if p == 3 else 9))
        tab[band][key] += 1
        overall[key] += 1

    ot = sum(overall.values())
    print(f"B(主導権)取得者の着順を ペース別に集計（対象 {ot}レース）\n")
    print("全体: B取得者の着順")
    print(f"  1着 {overall[1]/ot*100:.1f}% / 2着 {overall[2]/ot*100:.1f}% / "
          f"3着 {overall[3]/ot*100:.1f}% / 着外 {overall[9]/ot*100:.1f}%\n")
    print(f"  {'ペース帯':<18}{'R数':>6}{'1着':>8}{'2着':>8}{'3着':>8}{'着外':>8}{'連対(1-2着)':>12}")
    for band in ["スロー〜ミドル(≤2)", "ミドル〜ハイ(3)", "ハイ(≥4)"]:
        d = tab[band]
        tot = sum(d.values())
        if not tot:
            continue
        rentai = (d[1] + d[2]) / tot * 100
        print(f"  {band:<18}{tot:>6}{d[1]/tot*100:>7.1f}%{d[2]/tot*100:>7.1f}%"
              f"{d[3]/tot*100:>7.1f}%{d[9]/tot*100:>7.1f}%{rentai:>11.1f}%")
    print("\n（仮説: ハイペースほど B取得者の1着率↓・2着/着外率↑ なら『主導権でも垂れる』が実在）")


if __name__ == "__main__":
    main()
