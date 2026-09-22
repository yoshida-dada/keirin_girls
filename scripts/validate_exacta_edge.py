"""二車単 ◎→番手 × ◎弱め(混戦寄り) の「100%超セル」頑健性チェック(男子7車)。

ドリルで ◎勝率<0.30 の ◎→番手 が 100.3%(n=1245) と初の損益分岐超え。境界ノイズか本物か:
  A) 閾値スイープ … ◎勝率 上限を 0.24〜0.34 まで動かし ROI が閾値付近で安定か
  B) 時期分割     … 期間を前半/後半に割り、両方で ~100% を再現するか
  C) 併用フィルタ … <0.30 に「番手=2位」「ライン4車+」を重ねて厚くできるか
本番モデル、実払戻(payouts_exacta)。1点賭け(◎→番手)。

  PYTHONIOENCODING=utf-8 python scripts/validate_exacta_edge.py --days 540
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.feature_augment import augment_samples
from src.model.feature_sets import load_for


def _adate(rid):
    return date(int(rid[2:6]), int(rid[6:8]), int(rid[8:10])) + timedelta(days=int(rid[10:12]) - 1)


def _load(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    exa = {rid: (tuple(int(x) for x in combo.split("-")), p)
           for rid, combo, p in c.execute("SELECT race_id,combo,payout FROM payouts_exacta")}
    tmp = defaultdict(list)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        tmp[rid].append((li, pi, car))
    c.close()
    mate, llen = {}, {}
    for rid, rows in tmp.items():
        mx = max(li for li, _, _ in rows)
        ls = [[] for _ in range(mx + 1)]
        for li, pi, car in sorted(rows, key=lambda x: (x[0], x[1])):
            ls[li].append(car)
        ls = [x for x in ls if x]
        for ln in ls:
            for k in range(len(ln) - 1):
                mate[(rid, ln[k])] = ln[k + 1]
            for car in ln:
                llen[(rid, car)] = len(ln)
    return exa, mate, llen


def stat(bets):
    n = len(bets)
    if not n:
        return 0, 0, 0
    ret = sum(p for hit, p in bets if hit)
    return n, sum(1 for hit, _ in bets if hit) / n * 100, ret / (n * 100) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--days", type=int, default=540)
    args = ap.parse_args()
    exa, mate, llen = _load(args.db)
    model, elo, lbl = load_for(False)
    base = load_samples(args.db, field_size=[7], features=PL_FEATURES_FULL)
    samples = augment_samples(base, args.db, model.feature_names)
    dated = [(s, _adate(s.race_id)) for s in samples if s.race_id in exa]
    days = sorted({d for _, d in dated})[-args.days:]
    dayset = set(days)
    # ◎→番手が張れるレースだけ抽出。(date, pw, rank, llen, hit, pay)
    rows = []
    for s, d in dated:
        if d not in dayset:
            continue
        rid = s.race_id
        st = model.strengths(s.X, s.car_numbers)
        anchor = max(st, key=st.get)
        nb = mate.get((rid, anchor))
        if not nb:
            continue
        pw = st[anchor] / sum(st.values())
        rank = sorted(st, key=st.get, reverse=True).index(nb) + 1
        ll = llen.get((rid, anchor), 2)
        combo, pay = exa[rid]
        rows.append((d, pw, rank, ll, tuple(combo) == (anchor, nb), pay))
    rows.sort()
    print(f"男子7車 直近{args.days}日 {days[0]}〜{days[-1]}  ◎→番手が張れた {len(rows)}レース\n")

    print("A) ◎勝率 上限スイープ（cap未満だけ買う・1点◎→番手）")
    print(f"   {'◎勝率<':<10}{'点数':>7}{'的中率':>9}{'回収率':>9}")
    for cap in (0.24, 0.26, 0.28, 0.30, 0.32, 0.34):
        bets = [(h, p) for _, pw, _, _, h, p in rows if pw < cap]
        n, hr, roi = stat(bets)
        print(f"   {cap:<10.2f}{n:>7}{hr:>8.1f}%{roi:>8.1f}%{'  ★' if roi>=100 else ''}")
    print()

    print("B) ◎勝率<0.30 を時期3分割（各期間で~100%再現するか）")
    sub = [(d, h, p) for d, pw, _, _, h, p in rows if pw < 0.30]
    if sub:
        third = len(sub) // 3
        parts = [("前期", sub[:third]), ("中期", sub[third:2*third]), ("後期", sub[2*third:])]
        print(f"   {'期間':<8}{'範囲':<26}{'点数':>6}{'的中率':>9}{'回収率':>9}")
        for nm, part in parts:
            n, hr, roi = stat([(h, p) for _, h, p in part])
            rng = f"{part[0][0]}〜{part[-1][0]}"
            print(f"   {nm:<8}{rng:<26}{n:>6}{hr:>8.1f}%{roi:>8.1f}%{'  ★' if roi>=100 else ''}")
    print()

    print("C) ◎勝率<0.30 に併用フィルタ")
    filt = {
        "全体(<0.30)": lambda pw, rk, ll: True,
        "＋番手=2位": lambda pw, rk, ll: rk == 2,
        "＋ライン4車+": lambda pw, rk, ll: ll >= 4,
        "＋番手2位&ライン3車+": lambda pw, rk, ll: rk == 2 and ll >= 3,
    }
    print(f"   {'条件':<22}{'点数':>7}{'的中率':>9}{'回収率':>9}")
    for nm, fn in filt.items():
        bets = [(h, p) for _, pw, rk, ll, h, p in rows if pw < 0.30 and fn(pw, rk, ll)]
        n, hr, roi = stat(bets)
        print(f"   {nm:<22}{n:>7}{hr:>8.1f}%{roi:>8.1f}%{'  ★' if roi>=100 else ''}")


if __name__ == "__main__":
    main()
