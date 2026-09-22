"""初日の上がりタイムでデキを測り2日目に活かせるか（男子7車）。

上がりは展開/バンクに強く依存するので、生タイムでなく「同レース内の相対上がり=本人last_lap−その
レースの平均last_lap」を当日のデキ代理にする(負=速い=好調)。同一開催で初日→次走(2日目)の
ペアを作り、初日の相対上がりが2日目の成績残差(通算平均着−当該着,+=良化)を予測するか見る。
特に「初日着外だが上がり速い(展開負けだが好調)」が2日目に効くかを着順帯別に確認。

  PYTHONIOENCODING=utf-8 python scripts/analyze_agari_form.py
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
        "SELECT race_id,car_number,rider_name,position,last_lap FROM results"
        " WHERE position IS NOT NULL AND rider_name IS NOT NULL").fetchall()
    c.close()

    def pd(s):
        try:
            return date.fromisoformat(str(s))
        except (ValueError, TypeError):
            return None

    # レースごとの平均last_lap（7車のみ）
    race_ll = defaultdict(list)
    for rid, car, nm, pos, ll in rows:
        if fs.get(rid) == 7 and ll is not None:
            race_ll[rid].append(ll)
    race_mean = {rid: sum(v) / len(v) for rid, v in race_ll.items() if v}

    # 選手ごとに (date, rid, pos, agari_res) 。career baseline も
    byrider = defaultdict(list)
    finishes = defaultdict(list)
    for rid, car, nm, pos, ll in rows:
        if fs.get(rid) != 7:
            continue
        finishes[nm].append(pos)
        d = pd(rd.get(rid))
        ares = (ll - race_mean[rid]) if (ll is not None and rid in race_mean) else None
        if d:
            byrider[nm].append((d, rid, pos, ares))
    cbase = {nm: sum(v) / len(v) for nm, v in finishes.items() if v}

    def abk(a):
        if a is None:
            return None
        if a < -0.15:
            return "①速い(<-0.15)"
        if a < -0.05:
            return "②やや速(-0.15〜-0.05)"
        if a <= 0.05:
            return "③標準(±0.05)"
        if a <= 0.15:
            return "④やや遅(0.05〜0.15)"
        return "⑤遅い(>0.15)"

    ORDER = ["①速い(<-0.15)", "②やや速(-0.15〜-0.05)", "③標準(±0.05)", "④やや遅(0.05〜0.15)", "⑤遅い(>0.15)"]
    agg = defaultdict(lambda: {"n": 0, "res": 0.0, "top3": 0})           # 初日相対上がり→2日目
    agg_lose = defaultdict(lambda: {"n": 0, "res": 0.0, "top3": 0})      # 初日着外(4-7着)に限定

    for nm, v in byrider.items():
        v.sort()
        base = cbase[nm]
        # 同一開催内の連続走ペア（初日→2日目 等）
        for a, b in zip(v, v[1:]):
            if a[1][:10] != b[1][:10]:        # 同一開催のみ
                continue
            if int(b[1][10:12]) - int(a[1][10:12]) != 1:   # 連続日のみ
                continue
            bk = abk(a[3])
            if bk is None:
                continue
            r = base - b[2]                   # 2日目(翌日)の残差
            t3 = int(b[2] <= 3)
            d = agg[bk]; d["n"] += 1; d["res"] += r; d["top3"] += t3
            if a[2] >= 4:                     # 初日着外
                e = agg_lose[bk]; e["n"] += 1; e["res"] += r; e["top3"] += t3

    def show(title, agg):
        print(f"\n{title}")
        print(f"   {'初日の相対上がり':<24}{'n':>8}{'翌日残差':>9}{'翌日top3率':>11}")
        for bk in ORDER:
            d = agg.get(bk)
            if not d or d["n"] == 0:
                continue
            print(f"   {bk:<24}{d['n']:>8}{d['res']/d['n']:>+9.2f}{d['top3']/d['n']*100:>10.1f}%")

    print("男子7車: 初日の相対上がり(本人−レース平均, 負=速い=好調) → 翌日成績")
    show("【全体】初日相対上がり → 翌日残差", agg)
    show("【初日着外(4-7着)に限定】= 展開負けだが上がり速い が翌日効くか", agg_lose)
    print("\n※速い側(①②)ほど翌日残差/ top3 が高ければ『初日の上がりでデキ判定→翌日に活かせる』。"
          "平坦なら展開交絡でデキ信号は抽出できず。着外限定で効けば着順を超える上がり固有の情報。")


if __name__ == "__main__":
    main()
