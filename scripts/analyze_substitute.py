"""欠場補充などで開催途中(2日目/3日目)から参加した選手の成績を調査（男子7車）。

通常は全選手が初日(予選/特選)から出走し勝ち上がる。だが中途欠場の補充・流用選手は開催途中の
競走から挿入される（番組編成の要領 第2章 中途欠場選手の補充）。この「開催内 初出走日≥2日目」の
選手の、その開催での初戦成績を、初日から出た選手の初戦と比較する。能力交絡は残差(通算平均着−
当該着, +=良化)で除去。既存のgap特徴(前走からの日数)で説明できるかも併記。

  PYTHONIOENCODING=utf-8 python scripts/analyze_substitute.py
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

    # 全走(7車)を name 別に。career baseline と、直近前走日(gap算出)にも使う
    byrider = defaultdict(list)      # name -> [(date, rid, pos)]
    for rid, nm, pos, rdate in rows:
        if fs.get(rid) != 7:
            continue
        d = pd(rdate)
        if d:
            byrider[nm].append((d, rid, pos))
    cbase = {nm: sum(p for _, _, p in v) / len(v) for nm, v in byrider.items() if v}

    # 初出走日別に初戦残差を集計。gap(前走からの日数)も平均で併記
    agg = defaultdict(lambda: {"n": 0, "res": 0.0, "top3": 0, "win": 0, "gap": 0.0, "gapn": 0})
    for nm, v in byrider.items():
        v.sort()
        base = cbase[nm]
        # 各開催の初戦 index
        seen = {}
        for i, (d, rid, pos) in enumerate(v):
            meet = rid[:10]
            if meet in seen:
                continue
            seen[meet] = True
            nn = int(rid[10:12])
            dk = nn if nn <= 3 else 4
            a = agg[dk]
            a["n"] += 1
            a["res"] += base - pos
            a["top3"] += int(pos <= 3)
            a["win"] += int(pos == 1)
            if i > 0:
                g = (d - v[i - 1][0]).days
                if g > 0:
                    a["gap"] += g; a["gapn"] += 1

    print("男子7車: 開催内の初出走日目 × 初戦成績（残差=通算平均着−当該着, +=良化）\n")
    print(f"   {'初出走':<10}{'n':>9}{'1着率':>8}{'top3率':>8}{'残差':>8}{'平均gap日':>10}")
    labels = {1: "初日", 2: "2日目〜", 3: "3日目〜", 4: "4日目〜"}
    for k in [1, 2, 3, 4]:
        d = agg.get(k)
        if not d or d["n"] == 0:
            continue
        n = d["n"]
        gp = d["gap"] / d["gapn"] if d["gapn"] else 0
        print(f"   {labels[k]:<10}{n:>9}{d['win']/n*100:>7.1f}%{d['top3']/n*100:>7.1f}%"
              f"{d['res']/n:>+8.2f}{gp:>10.1f}")
    print("\n※2日目〜の初出走=補充/途中参加。残差が初日組より低ければ『途中参加のハンデ』。"
          "平均gapが大きければ既存のgap_lin特徴が一部説明。n が小さければ稀ケース。")


if __name__ == "__main__":
    main()
