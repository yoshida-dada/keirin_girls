"""記者の並び予想は「スタート隊列」か「最終バック(主導権)隊列」か、を実データで判定する。

仮説: 並び予想(narabi)は初手(S)ではなく、レース中盤〜バックの主導権(B)の隊列を表す。
検証: narabi予想先頭(pos0)が、実際の S(sb='S') と B(sb='B') のどちらとより一致するか。
併せて S取得者と B取得者が同一かも見る（別物なら初手と主導権は別フェーズ）。
予測力比較: B(主導権先頭)を narabi先頭 / b_count(as-of) / s_count(as-of) のどれが最も当てるか。

  PYTHONIOENCODING=utf-8 python scripts/analyze_narabi_phase.py --db data/keirin.sqlite
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

    ent = defaultdict(set)
    for rid, car in c.execute("SELECT race_id,car_number FROM entries"):
        ent[rid].add(car)
    scount = defaultdict(dict); bcount = defaultdict(dict)
    for rid, car, s, b in c.execute("SELECT race_id,car_number,s_count,b_count FROM recent_form"):
        scount[rid][car] = s or 0
        bcount[rid][car] = b or 0
    narabi = defaultdict(dict)
    for rid, car, pos in c.execute("SELECT race_id,car_number,position FROM narabi"):
        narabi[rid][car] = pos
    sb = defaultdict(dict)
    for rid, car, v in c.execute("SELECT race_id,car_number,sb FROM results"):
        sb[rid][car] = v or ""
    # 7車 & narabiあり のレース
    races = [r[0] for r in c.execute(
        "SELECT race_id FROM races WHERE field_size=7")]
    c.close()

    n = 0
    nf_S = 0; nf_S_den = 0        # narabi先頭==S
    nf_B = 0; nf_B_den = 0        # narabi先頭==B
    SeqB = 0; SB_den = 0          # S取得者==B取得者
    predB_narabi = 0; predB_b = 0; predB_s = 0; predB_den = 0
    for rid in races:
        cars = ent.get(rid, set())
        if len(cars) != 7:
            continue
        sbr = sb.get(rid, {})
        s_take = [car for car, v in sbr.items() if "S" in v]
        b_take = [car for car, v in sbr.items() if "B" in v]
        nb = narabi.get(rid, {})
        front = [car for car, p in nb.items() if p == 0]
        if not front:
            continue
        f = front[0]
        n += 1
        if len(s_take) == 1:
            nf_S_den += 1; nf_S += int(f == s_take[0])
        if len(b_take) == 1:
            nf_B_den += 1; nf_B += int(f == b_take[0])
            # B予測: narabi先頭 / b_count argmax / s_count argmax
            sc = scount.get(rid, {}); bc = bcount.get(rid, {})
            predB_den += 1
            predB_narabi += int(f == b_take[0])
            pb = sorted(cars, key=lambda x: (-bc.get(x, 0), x))[0]
            ps = sorted(cars, key=lambda x: (-sc.get(x, 0), x))[0]
            predB_b += int(pb == b_take[0])
            predB_s += int(ps == b_take[0])
        if len(s_take) == 1 and len(b_take) == 1:
            SB_den += 1; SeqB += int(s_take[0] == b_take[0])

    print(f"対象 {n}レース（7車・narabi予想先頭あり）\n")
    print("【記者の予想先頭は S と B のどちらと一致するか】")
    print(f"  予想先頭 == 実S(スタート先頭) : {nf_S}/{nf_S_den} = {nf_S/max(1,nf_S_den)*100:.1f}%")
    print(f"  予想先頭 == 実B(バック=主導権) : {nf_B}/{nf_B_den} = {nf_B/max(1,nf_B_den)*100:.1f}%")
    print(f"\n【S取得者 == B取得者（初手と主導権が同一選手か）】 {SeqB}/{SB_den} = {SeqB/max(1,SB_den)*100:.1f}%")
    print(f"\n【B(主導権先頭)を最も当てる予測は？（母数 {predB_den}）】")
    print(f"  記者の予想先頭   : {predB_narabi/max(1,predB_den)*100:.1f}%")
    print(f"  B回数 argmax     : {predB_b/max(1,predB_den)*100:.1f}%")
    print(f"  S回数 argmax     : {predB_s/max(1,predB_den)*100:.1f}%")


if __name__ == "__main__":
    main()
