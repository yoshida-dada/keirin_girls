"""2着決まり手「ク」(番手マークして2着に残る決着)が、将来の2着力を予測するかを as-of で検証。

各選手を race_id 昇順(≒時系列)に辿り、そのレース時点での過去成績:
  ku_rate  = 過去に「2着かつ決まり手=ク」だった割合（マークして2着に残せる率）
  top2_rate= 過去の2着以内率（既存特徴の近似・比較用）
を出し、当該レースで実際に2着になったかを記録。過去ku_rate帯ごとに実2着率を見て、
ku_rate が将来2着を当てるか（単調増加か）、既存のtop2_rateを超える情報かを判定。

  PYTHONIOENCODING=utf-8 python scripts/analyze_ku_mark.py --db data/keirin.sqlite
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
    ap.add_argument("--min-races", type=int, default=15)
    args = ap.parse_args()
    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    # 7車ガールズの結果を race_id 昇順で（race_id は日付を含み時系列近似）
    rows = c.execute("""
        SELECT r.race_id, r.rider_name, r.position, r.kimarite
        FROM results r JOIN races ra ON ra.race_id=r.race_id
        WHERE ra.field_size=7 AND r.position IS NOT NULL
        ORDER BY r.race_id""").fetchall()
    c.close()

    hist = defaultdict(lambda: {"n": 0, "ku": 0, "t2": 0})   # rider -> 過去累計
    ku_bins = defaultdict(lambda: [0, 0])    # ku_rate帯 -> [2着数, 母数]
    t2_bins = defaultdict(lambda: [0, 0])
    # ku_rate を固定して top2_rate で層別（kuが top2の内数でないか確認）
    pair = defaultdict(lambda: [0, 0])

    def band(x):
        for lo, hi in [(0, .05), (.05, .10), (.10, .15), (.15, .20), (.20, .30), (.30, 1.01)]:
            if lo <= x < hi:
                return f"{lo*100:.0f}-{hi*100:.0f}%"
        return "?"

    for rid, name, pos, km in rows:
        h = hist[name]
        if h["n"] >= args.min_races:
            kr = h["ku"] / h["n"]
            tr = h["t2"] / h["n"]
            is2 = int(pos == 2)
            kb = ku_bins[band(kr)]; kb[0] += is2; kb[1] += 1
            tb = t2_bins[band(tr)]; tb[0] += is2; tb[1] += 1
            # top2_rate中位(0.15-0.30)に絞って ku_rate の追加効果を見る
            if .15 <= tr < .30:
                p = pair["ku高" if kr >= .12 else "ku低"]; p[0] += is2; p[1] += 1
        # 更新
        h["n"] += 1
        if pos == 2:
            h["t2"] += 1
            if km == "ク":
                h["ku"] += 1

    print(f"as-of検証（過去{args.min_races}走以上の選手・レース）\n")
    print("【過去ク率 → 実2着率】(単調増加なら将来2着を予測)")
    for b in ["0-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30-100%"]:
        a, d = ku_bins.get(b, [0, 0])
        if d:
            print(f"  ク率 {b:>8}: 実2着率 {a/d*100:5.1f}%  (n={d})")
    print("\n【比較: 過去2着以内率 → 実2着率】(既存特徴の予測力)")
    for b in ["0-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30-100%"]:
        a, d = t2_bins.get(b, [0, 0])
        if d:
            print(f"  top2率 {b:>8}: 実2着率 {a/d*100:5.1f}%  (n={d})")
    print("\n【top2率0.15-0.30に固定して ク率で層別（純増の有無）】")
    for k in ("ku高", "ku低"):
        a, d = pair.get(k, [0, 0])
        if d:
            print(f"  {k}: 実2着率 {a/d*100:5.1f}%  (n={d})")


if __name__ == "__main__":
    main()
