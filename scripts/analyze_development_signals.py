"""男子の展開予想に使える信号を洗い出して実測する（棚卸し）。

**方針**: 外部サイトの「カマシ成功率」等を取りに行く前に、手元のDBから何が測れるかを尽くす。
記者の並び予想(脚質ラベル)＋着順＋決まり手＋S/Bマーカー＋ライン構成が揃っているので、
主導権・チギリ・飛びつき・競り・連携実績はすべて自前で計算できる。

測る対象:
  A 主導権(B)  … 脚質×ライン内位置。既存特徴(過去B回数)で層別してなお効くかも見る
  B チギリ     … 先頭が主導権を取ったのに番手が離れる率。選手別の「ちぎられ率」も出す
  C 飛びつき   … 1着ラインの外から2着に入る選手の脚質分布（ライン決着の裏側）
  D 競り       … 競りラベルの選手の着順分布
  E 連携実績   … 同じ(先頭,番手)ペアの過去成績が、今回のライン決着を予測するか（as-of）
  F 決まり手   … 主導権者の脚質別の決まり手構成（展開条件付き分布の材料）

  PYTHONIOENCODING=utf-8 python scripts/analyze_development_signals.py
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR

MIN_N = 150          # これ未満のセルは表示しない（偶然を拾わないため）


def load(db: str):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    nb = defaultdict(dict)          # race_id -> car -> (leg, line_id, pos_in_line)
    for rid, car, lg, li, pi in c.execute(
            "SELECT race_id,car_number,leg,line_id,pos_in_line FROM narabi"
            " WHERE line_id IS NOT NULL"):
        nb[rid][car] = (lg, li, pi)
    pos, kim, sbm = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for rid, p, car, k, s in c.execute(
            "SELECT race_id,position,car_number,kimarite,sb FROM results"):
        pos[rid][car] = p
        kim[rid][car] = k
        sbm[rid][car] = s
    name, cls = defaultdict(dict), defaultdict(dict)
    for rid, car, nm, cr in c.execute(
            "SELECT race_id,car_number,rider_name,class_rank FROM entries"):
        name[rid][car] = nm
        cls[rid][car] = cr
    bcnt = defaultdict(dict)
    for rid, car, b in c.execute("SELECT race_id,car_number,b_count FROM recent_form"):
        bcnt[rid][car] = b or 0
    dates = {rid: d for rid, d in c.execute("SELECT race_id,race_date FROM races")}
    c.close()
    return nb, pos, kim, sbm, name, cls, bcnt, dates


def _has_b(s) -> bool:
    return bool(s) and "B" in str(s)


def _lines_of(d: dict) -> dict[int, list[int]]:
    """race の narabi dict → {line_id: [車番(先頭順)]}。"""
    mem = defaultdict(list)
    for car, (lg, li, pi) in d.items():
        mem[li].append((pi, car))
    return {li: [c for _, c in sorted(v)] for li, v in mem.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="展開予想に使える信号の棚卸し")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    args = ap.parse_args()
    nb, pos, kim, sbm, name, cls, bcnt, dates = load(args.db)
    races = [r for r in nb if pos.get(r)]
    print(f"対象 {len(races):,}レース\n")

    # ---------- A 主導権(B) ----------
    print("=" * 70)
    print("A. 主導権(B)取得率  ライン内位置 × 脚質")
    agg = defaultdict(lambda: [0, 0, 0])
    for rid in races:
        d = nb[rid]
        L = _lines_of(d)
        for car, (lg, li, pi) in d.items():
            if car not in pos[rid]:
                continue
            sz = len(L[li])
            role = "単騎" if sz == 1 else ("先頭" if pi == 0 else ("番手" if pi == 1 else "3番手+"))
            a = agg[(role, lg)]
            a[0] += 1
            a[1] += int(_has_b(sbm[rid].get(car)))
            a[2] += int(pos[rid].get(car) == 1)
    print(f"{'位置':>7}{'脚質':>7}{'n':>8}{'主導権B':>9}{'1着':>8}")
    for role in ("先頭", "単騎", "番手", "3番手+"):
        rows = [(lg, a) for (r, lg), a in agg.items() if r == role and a[0] >= MIN_N]
        for lg, a in sorted(rows, key=lambda x: -x[1][1] / x[1][0]):
            print(f"{role:>7}{lg or '(空)':>7}{a[0]:>8,}{a[1]/a[0]*100:>8.1f}%{a[2]/a[0]*100:>7.1f}%")

    # ---------- B チギリ ----------
    print("\n" + "=" * 70)
    print("B. チギリ（先頭が主導権を取ったのに番手が離れる）")
    by_leg = defaultdict(lambda: [0, 0])
    by_rider = defaultdict(lambda: [0, 0])       # 番手選手の「ちぎられ率」
    by_head = defaultdict(lambda: [0, 0])        # 先頭選手の「ちぎり率」
    for rid in races:
        d, L = nb[rid], _lines_of(nb[rid])
        for li, mem in L.items():
            if len(mem) < 2:
                continue
            head, mate = mem[0], mem[1]
            if not _has_b(sbm[rid].get(head)):
                continue
            mp = pos[rid].get(mate)
            if mp is None:
                continue
            broke = int(mp >= 4)
            by_leg[d[head][0]][0] += 1
            by_leg[d[head][0]][1] += broke
            by_rider[name[rid].get(mate)][0] += 1
            by_rider[name[rid].get(mate)][1] += broke
            by_head[name[rid].get(head)][0] += 1
            by_head[name[rid].get(head)][1] += broke
    print(f"  {'先頭の脚質':>10}{'n':>8}{'チギリ率':>10}")
    for lg, a in sorted(by_leg.items(), key=lambda kv: -kv[1][0]):
        if a[0] >= MIN_N:
            print(f"  {lg or '(空)':>10}{a[0]:>8,}{a[1]/a[0]*100:>9.1f}%")
    rid_rates = [(v[1] / v[0], v[0], k) for k, v in by_rider.items() if v[0] >= 20 and k]
    if rid_rates:
        rid_rates.sort()
        print(f"\n  選手別「ちぎられ率」(20走以上 {len(rid_rates)}名): "
              f"最良 {rid_rates[0][0]*100:.0f}% / 中央 {rid_rates[len(rid_rates)//2][0]*100:.0f}% "
              f"/ 最悪 {rid_rates[-1][0]*100:.0f}%")
        print(f"    → 選手差 {(rid_rates[-1][0]-rid_rates[0][0])*100:.0f}pt。特徴量化の価値あり")
    hd = [(v[1] / v[0], v[0], k) for k, v in by_head.items() if v[0] >= 20 and k]
    if hd:
        hd.sort()
        print(f"  先頭別「ちぎり率」(20走以上 {len(hd)}名): "
              f"最良 {hd[0][0]*100:.0f}% / 中央 {hd[len(hd)//2][0]*100:.0f}% / 最悪 {hd[-1][0]*100:.0f}%")

    # ---------- C 飛びつき（1着ラインの外から2着） ----------
    print("\n" + "=" * 70)
    print("C. 1着ラインの外から2着に入る選手（＝ライン決着しない側）の脚質")
    out2 = defaultdict(lambda: [0, 0])
    tot_out = same = 0
    for rid in races:
        d = nb[rid]
        P = pos[rid]
        w = next((c for c, p in P.items() if p == 1), None)
        s2 = next((c for c, p in P.items() if p == 2), None)
        if w is None or s2 is None or w not in d or s2 not in d:
            continue
        tot_out += 1
        if d[w][1] == d[s2][1]:
            same += 1
            continue
        out2[d[s2][0]][0] += 1
    n_out = sum(v[0] for v in out2.values())
    print(f"  ライン決着 {same/tot_out*100:.1f}% / ライン外2着 {(tot_out-same)/tot_out*100:.1f}%"
          f"  (n={tot_out:,})")
    print(f"  {'ライン外2着の脚質':>16}{'件数':>8}{'構成比':>9}")
    for lg, a in sorted(out2.items(), key=lambda kv: -kv[1][0]):
        if a[0] >= MIN_N:
            print(f"  {lg or '(空)':>16}{a[0]:>8,}{a[0]/n_out*100:>8.1f}%")

    # ---------- D 競り ----------
    print("\n" + "=" * 70)
    print("D. 競りラベルの選手の着順")
    seri = [0, 0, 0, 0]
    base = [0, 0, 0, 0]
    for rid in races:
        for car, (lg, li, pi) in nb[rid].items():
            p = pos[rid].get(car)
            if p is None:
                continue
            t = seri if lg == "競り" else base
            t[0] += 1
            t[1] += int(p == 1)
            t[2] += int(p <= 2)
            t[3] += int(p <= 3)
    if seri[0]:
        print(f"  競り  n={seri[0]:,}  1着{seri[1]/seri[0]*100:.1f}% "
              f"連対{seri[2]/seri[0]*100:.1f}% 3着内{seri[3]/seri[0]*100:.1f}%")
        print(f"  他    n={base[0]:,}  1着{base[1]/base[0]*100:.1f}% "
              f"連対{base[2]/base[0]*100:.1f}% 3着内{base[3]/base[0]*100:.1f}%")

    # ---------- E 連携実績（as-of） ----------
    print("\n" + "=" * 70)
    print("E. 連携実績: 同じ(先頭,番手)ペアの過去成績が今回を予測するか（as-of・リーク無し）")
    print("   指標は2つ: pair12=そのペアで1-2着を占めた / mate_top3=番手が3着以内")
    ev12, ev3 = [], []
    hist = defaultdict(lambda: [0, 0, 0])     # (先頭名,番手名) -> [同走数, pair12, mate_top3]
    for rid in sorted(races, key=lambda r: (dates.get(r) or "", r)):
        L, P = _lines_of(nb[rid]), pos[rid]
        w = next((c for c, p in P.items() if p == 1), None)
        s2 = next((c for c, p in P.items() if p == 2), None)
        for li, mem in L.items():
            if len(mem) < 2:
                continue
            head, mate = mem[0], mem[1]
            key = (name[rid].get(head), name[rid].get(mate))
            if None in key:
                continue
            pair12 = int(w is not None and s2 is not None
                         and {w, s2} == {head, mate})
            mp = P.get(mate)
            mate3 = int(mp is not None and mp <= 3)
            h = hist[key]
            if h[0] >= 3:                     # 過去3走以上の実績があるペアだけ評価
                ev12.append((h[1] / h[0], pair12))
                ev3.append((h[2] / h[0], mate3))
            h[0] += 1
            h[1] += pair12
            h[2] += mate3
    for lbl, ev, base_lbl in (("ペアで1-2着を占める", ev12, "pair12"),
                              ("番手が3着以内", ev3, "mate_top3")):
        if not ev:
            continue
        overall = sum(v for _, v in ev) / len(ev)
        lo = [v for r, v in ev if r < 0.25]
        mid = [v for r, v in ev if 0.25 <= r < 0.5]
        hi = [v for r, v in ev if r >= 0.5]
        print(f"\n  【{lbl}】評価 {len(ev):,}件 / 全体 {overall*100:.1f}%")
        for l2, g in (("過去 <25%", lo), ("25-50%", mid), (">=50%", hi)):
            if len(g) >= 100:
                print(f"    {l2:>10}: 今回 {sum(g)/len(g)*100:5.1f}%  (n={len(g):,})")
        if len(lo) >= 100 and len(hi) >= 100:
            print(f"    → 差 {(sum(hi)/len(hi)-sum(lo)/len(lo))*100:+.1f}pt")

    # ---------- F 決まり手構成 ----------
    print("\n" + "=" * 70)
    print("F. 主導権者の脚質別 決まり手構成（展開条件付き分布の材料）")
    km = defaultdict(lambda: defaultdict(int))
    for rid in races:
        d = nb[rid]
        b = next((c for c in d if _has_b(sbm[rid].get(c))), None)
        w = next((c for c, p in pos[rid].items() if p == 1), None)
        if b is None or w is None:
            continue
        k = kim[rid].get(w) or "?"
        km[d[b][0]][k] += 1
        km[d[b][0]]["_n"] += 1
        km[d[b][0]]["_self"] += int(b == w)
    print(f"  {'主導権者の脚質':>14}{'n':>8}{'逃':>7}{'捲':>7}{'差':>7}{'ク':>7}{'B取者が1着':>11}")
    for lg, v in sorted(km.items(), key=lambda kv: -kv[1]["_n"]):
        n = v["_n"]
        if n < MIN_N:
            continue
        print(f"  {lg or '(空)':>14}{n:>8,}{v['逃']/n*100:>6.1f}%{v['捲']/n*100:>6.1f}%"
              f"{v['差']/n*100:>6.1f}%{v['ク']/n*100:>6.1f}%{v['_self']/n*100:>10.1f}%")


if __name__ == "__main__":
    main()
