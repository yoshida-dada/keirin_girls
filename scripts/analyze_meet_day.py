"""開催日別(初日/二日目/三日目)の傾向と、「初日に着を落とした選手は二日目に走る」仮説の検証。

競輪は原則3日開催。race_id=[会場2][初日8][NN日目2][R4]。開催キー=race_id[:10]、日目=race_id[10:12]。
決勝進出には初日・二日目の成績が重要＝勝ち上がり構造で、初日着外の選手は二日目に別条件の
レースへ回る/奮起する可能性。DBのみ(モデル不要)で:
  Part1: 日別に 万車券率(7車,三連単>=1万) と 勝者脚質(逃/両/追)。
  Part2: 同一開催で日1→日2に出た選手の、日1着順別 日2の3着内率/平均着順（選手の通算3着内率を基準に残差）。

  PYTHONIOENCODING=utf-8 python scripts/analyze_meet_day.py            # 男子
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    args = ap.parse_args()
    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")

    leg = {}
    for rid, car, lt in c.execute("SELECT race_id,car_number,leg_type FROM entries"):
        leg[(rid, car)] = lt
    fs = {rid: n for rid, n in c.execute("SELECT race_id,field_size FROM races")}
    pay = {}
    for rid, combo, p in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta"):
        pay[rid] = (tuple(int(x) for x in combo.split("-")), p)
    # 全レースの選手着順（開催内の選手追跡用）
    res = defaultdict(list)          # race_id -> [(name,pos)]
    for rid, name, pos in c.execute("SELECT race_id,rider_name,position FROM results WHERE position IS NOT NULL"):
        res[rid].append((name, pos))
    c.close()

    # ---- Part1: 日別 万車券率・勝者脚質（7車）----
    day_stat = defaultdict(lambda: {"n": 0, "man": 0, "wleg": Counter()})
    for rid, (combo, p) in pay.items():
        if fs.get(rid) != 7:
            continue
        d = rid[10:12]
        st = day_stat[d]
        st["n"] += 1
        if p >= MAN:
            st["man"] += 1
            st["wleg"][leg.get((rid, combo[0]))] += 1
    print(f"開催日別の傾向  DB={Path(args.db).stem}\n")
    print("【Part1: 日別 万車券率・勝者脚質(7車)】")
    print(f"   {'日目':>4}{'R数':>8}{'万車券率':>9}{'勝者逃':>8}{'勝者両':>8}{'勝者追':>8}")
    for d in ["01", "02", "03", "04"]:
        s = day_stat.get(d)
        if not s or s["n"] < 200:
            continue
        m = s["man"]
        wl = s["wleg"]
        def pc(k): return f"{wl.get(k,0)/m*100:.0f}%" if m else "-"
        print(f"   {int(d):>4}{s['n']:>8}{s['man']/s['n']*100:>8.1f}%{pc('逃'):>8}{pc('両'):>8}{pc('追'):>8}")

    # ---- Part2: 初日着落とし→二日目 ----
    # 選手の通算3着内率（基準）
    base_t3 = defaultdict(lambda: [0, 0])
    for rid, lst in res.items():
        for name, pos in lst:
            b = base_t3[name]
            b[0] += int(pos <= 3); b[1] += 1
    # 開催×選手ごとに 日目→着順
    meet_rider = defaultdict(dict)   # (meet,name) -> {day:pos}
    for rid, lst in res.items():
        meet, day = rid[:10], int(rid[10:12])
        for name, pos in lst:
            meet_rider[(meet, name)].setdefault(day, pos)

    def transition(d_from, d_to):
        buckets = defaultdict(lambda: {"n": 0, "t3": 0, "sumpos": 0, "resid": 0.0})
        for (meet, name), dd in meet_rider.items():
            if d_from not in dd or d_to not in dd:
                continue
            p1, p2 = dd[d_from], dd[d_to]
            key = "初日3着内" if p1 <= 3 else ("初日4-6着" if p1 <= 6 else "初日7着以下")
            bk = buckets[key]
            bk["n"] += 1
            bk["t3"] += int(p2 <= 3)
            bk["sumpos"] += p2
            bl = base_t3[name]
            base = bl[0] / bl[1] if bl[1] else 0
            bk["resid"] += int(p2 <= 3) - base
        return buckets

    print(f"\n【Part2: 初日→二日目（同一開催で両日出走した選手）】")
    print(f"   {'初日成績':<12}{'人数':>7}{'二日目3着内率':>13}{'二日目平均着':>12}{'通算比(残差)':>13}")
    for key in ["初日3着内", "初日4-6着", "初日7着以下"]:
        b = transition(1, 2).get(key)
        if not b or b["n"] < 100:
            continue
        n = b["n"]
        print(f"   {key:<12}{n:>7}{b['t3']/n*100:>12.1f}%{b['sumpos']/n:>12.2f}{b['resid']/n*100:>+12.1f}pt")

    print(f"\n【参考: 二日目→三日目】")
    print(f"   {'二日目成績':<12}{'人数':>7}{'三日目3着内率':>13}{'三日目平均着':>12}{'通算比(残差)':>13}")
    for key in ["初日3着内", "初日4-6着", "初日7着以下"]:
        b = transition(2, 3).get(key)
        if not b or b["n"] < 100:
            continue
        n = b["n"]
        lbl = key.replace("初日", "二日目")
        print(f"   {lbl:<12}{n:>7}{b['t3']/n*100:>12.1f}%{b['sumpos']/n:>12.2f}{b['resid']/n*100:>+12.1f}pt")


if __name__ == "__main__":
    main()
