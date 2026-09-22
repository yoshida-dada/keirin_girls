"""連戦・過密の疲労を調査（男子7車）。単一の前走間隔(gap_lin採用済)では長期ブランクしか
捉えていない。「開催が詰まってる疲れ」を ①開催内日進行(連戦) ②直近14/30日の累積出走数
③短間隔の細分 で確認する。能力交絡は残差(通算平均着−当該着, +=良化)で除去。

  PYTHONIOENCODING=utf-8 python scripts/analyze_fatigue.py
"""
from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR

DB = str(DATA_DIR / "keirin_men.sqlite")


def main():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    fs = {rid: n for rid, n in c.execute("SELECT race_id,field_size FROM races")}
    rd = {rid: d for rid, d in c.execute("SELECT race_id,race_date FROM races")}
    rows = c.execute(
        "SELECT ra.race_id, res.rider_name, res.position, ra.race_date"
        " FROM results res JOIN races ra ON res.race_id=ra.race_id"
        " WHERE res.position IS NOT NULL AND res.rider_name IS NOT NULL").fetchall()
    c.close()

    def pd(s):
        try:
            return date.fromisoformat(str(s))
        except (ValueError, TypeError):
            return None

    byrider = defaultdict(list)   # name -> [(date, rid, pos)]
    for rid, nm, pos, rdate in rows:
        if fs.get(rid) != 7:
            continue
        d = pd(rdate)
        if d:
            byrider[nm].append((d, rid, pos))
    cbase = {nm: sum(p for _, _, p in v) / len(v) for nm, v in byrider.items() if v}

    day_agg = defaultdict(lambda: {"n": 0, "res": 0.0, "top3": 0})     # 開催内日目
    cnt14 = defaultdict(lambda: {"n": 0, "res": 0.0, "top3": 0})       # 直近14日の出走数
    cnt30 = defaultdict(lambda: {"n": 0, "res": 0.0, "top3": 0})
    shortgap = defaultdict(lambda: {"n": 0, "res": 0.0, "top3": 0})    # 短間隔細分

    for nm, v in byrider.items():
        v.sort()
        base = cbase[nm]
        dates = [d for d, _, _ in v]
        for i, (d, rid, pos) in enumerate(v):
            r = base - pos
            t3 = int(pos <= 3)
            # ①開催内日目
            nn = int(rid[10:12])
            dk = nn if nn <= 4 else 5
            a = day_agg[dk]; a["n"] += 1; a["res"] += r; a["top3"] += t3
            # ②直近14/30日の出走数（当該レースは除く、過去のみ）
            n14 = sum(1 for pv in dates[:i] if 0 < (d - pv).days <= 14)
            n30 = sum(1 for pv in dates[:i] if 0 < (d - pv).days <= 30)
            b14 = n14 if n14 <= 5 else 6
            b30 = n30 if n30 <= 8 else 9
            e = cnt14[b14]; e["n"] += 1; e["res"] += r; e["top3"] += t3
            f = cnt30[b30]; f["n"] += 1; f["res"] += r; f["top3"] += t3
            # ③短間隔細分（前走からの日数, 2-13日を細かく）
            if i > 0:
                g = (d - dates[i - 1]).days
                if 1 <= g <= 13:
                    key = {1: "1日", 2: "2日", 3: "3日", 4: "4-6日", 5: "4-6日", 6: "4-6日"}.get(g, "7-13日")
                    s = shortgap[key]; s["n"] += 1; s["res"] += r; s["top3"] += t3

    def show(title, agg, order):
        print(f"\n{title}")
        print(f"   {'区分':<10}{'n':>9}{'残差':>8}{'top3率':>8}")
        for k in order:
            d = agg.get(k)
            if not d or d["n"] == 0:
                continue
            print(f"   {str(k):<10}{d['n']:>9}{d['res']/d['n']:>+8.2f}{d['top3']/d['n']*100:>7.1f}%")

    print("男子7車: 連戦・過密の疲労（残差=通算平均着−当該着, +=通常より上位=良化）")
    show("①開催内 日目（連戦の蓄積疲労）", day_agg, [1, 2, 3, 4, 5])
    print("   （日目が進むほど残差が下がるなら開催内の疲労。ただし勝ち上がり構造の交絡に注意）")
    show("②直近14日の出走数（過密度）", cnt14, [0, 1, 2, 3, 4, 5, 6])
    show("③直近30日の出走数（過密度）", cnt30, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print("   （出走数が多いほど残差が下がるなら過密疲労。0=直近ブランク→長期側で低下のはず）")
    show("④短間隔の細分（前走からの日数）", shortgap, ["1日", "2日", "3日", "4-6日", "7-13日"])


if __name__ == "__main__":
    main()
