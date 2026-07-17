"""スタート先頭(S=誘導員後ろ)の予測可能性と、記者の並び予想との乖離を探索する。

結果の results.sb='S'(スタートで前を取った選手) を「実際のスタート先頭」とみなし、
  (1) 車番 → S取得率（内の車番ほど好位置を取りやすい、の検証）
  (2) 事前S回数(recent_form.s_count, as-of) → S取得の予測力
  (3) 記者の予想先頭(narabi pos0) が実際のSと一致する率
  (4) 不一致(記者予想が外れた)時、実際のSは「内の車番/高S回数」か
  (5) 単純予測器「S=argmax(s_count, タイブレーク=内車番)」vs「S=記者予想先頭」の的中比較
を集計する。リーク無し（s_count/narabiは発走前確定, sbは結果）。

  PYTHONIOENCODING=utf-8 python scripts/analyze_start_position.py --db data/keirin.sqlite
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
    ap = argparse.ArgumentParser(description="スタート先頭(S)の予測と並び予想の乖離")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    args = ap.parse_args()
    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")

    # 対象: field_size=7 かつ S印のあるレース
    races = [r[0] for r in c.execute(
        "SELECT DISTINCT r.race_id FROM races r JOIN results re ON re.race_id=r.race_id "
        "WHERE r.field_size=7 AND re.sb LIKE '%S%'")]
    entries = defaultdict(dict)   # rid -> {car: leg}
    for rid, car, leg in c.execute("SELECT race_id,car_number,leg_type FROM entries"):
        entries[rid][car] = leg
    scount = defaultdict(dict)     # rid -> {car: s_count}
    for rid, car, s in c.execute("SELECT race_id,car_number,s_count FROM recent_form"):
        scount[rid][car] = s if s is not None else 0
    narabi = defaultdict(dict)     # rid -> {car: pos}
    for rid, car, pos in c.execute("SELECT race_id,car_number,position FROM narabi"):
        narabi[rid][car] = pos
    sb = defaultdict(dict)         # rid -> {car: sb}
    for rid, car, v in c.execute("SELECT race_id,car_number,sb FROM results"):
        sb[rid][car] = v or ""
    c.close()

    car_S = defaultdict(int); car_N = defaultdict(int)
    n = 0
    s_is_maxscount = 0; s_maxscount_denom = 0
    pred_scount_hit = 0; pred_narabi_hit = 0; pred_scount_inner_hit = 0
    narabi_match = 0; narabi_denom = 0
    div = 0; div_inner = 0; div_higher_s = 0
    div_scar = []; div_ncar = []

    for rid in races:
        ent = entries.get(rid, {})
        if len(ent) != 7:
            continue
        sbr = sb.get(rid, {})
        s_takers = [car for car, v in sbr.items() if "S" in v]
        if len(s_takers) != 1:            # Sが一意なレースのみ（曖昧を除外）
            continue
        s_taker = s_takers[0]
        sc = scount.get(rid, {})
        nb = narabi.get(rid, {})
        cars = list(ent.keys())
        n += 1
        for car in cars:
            car_N[car] += 1
        car_S[s_taker] += 1

        # (2) S取得者は最大s_countの選手か
        if sc:
            maxs = max(sc.get(car, 0) for car in cars)
            top_s_cars = [car for car in cars if sc.get(car, 0) == maxs and maxs > 0]
            if top_s_cars:
                s_maxscount_denom += 1
                if s_taker in top_s_cars:
                    s_is_maxscount += 1
                # (5) 予測器: argmax s_count, タイブレーク=内(小車番)
                pred = sorted(cars, key=lambda x: (-sc.get(x, 0), x))[0]
                pred_scount_hit += int(pred == s_taker)
                pred_scount_inner_hit += int(pred == s_taker)  # 同一(タイブレーク内蔵)

        # (3) 記者予想先頭 == 実S
        front = [car for car, p in nb.items() if p == 0]
        if front:
            narabi_denom += 1
            f = front[0]
            pred_narabi_hit += int(f == s_taker)
            if f == s_taker:
                narabi_match += 1
            else:
                # (4) 乖離の特徴
                div += 1
                div_scar.append(s_taker); div_ncar.append(f)
                if s_taker < f:                # 実Sの方が内
                    div_inner += 1
                if sc.get(s_taker, 0) > sc.get(f, 0):
                    div_higher_s += 1

    print(f"対象 {n}レース（S一意・7車・narabi/ scountあり）\n")
    print("【(1) 車番 → S取得率】内(小車番)ほど高いか")
    for car in range(1, 8):
        r = car_S[car] / car_N[car] * 100 if car_N[car] else 0
        bar = "#" * int(r / 2)
        print(f"  {car}番: {r:5.1f}%  (S{car_S[car]:>4}/{car_N[car]:>4})  {bar}")

    print(f"\n【(2) S取得者が“最大S回数”の選手だった率】 {s_is_maxscount}/{s_maxscount_denom} "
          f"= {s_is_maxscount/max(1,s_maxscount_denom)*100:.1f}%")
    print(f"【(3) 記者の予想先頭 == 実S】 {narabi_match}/{narabi_denom} "
          f"= {narabi_match/max(1,narabi_denom)*100:.1f}%")
    print(f"\n【(5) 実Sの的中率比較（同一母数 narabi_denom）】")
    print(f"  記者予想先頭で予測      : {pred_narabi_hit/max(1,narabi_denom)*100:.1f}%")
    print(f"  S回数argmax(内タイブレ) : {pred_scount_hit/max(1,s_maxscount_denom)*100:.1f}%  （母数 {s_maxscount_denom}）")

    print(f"\n【(4) 記者予想が外れた(乖離) {div}レースの特徴】")
    if div:
        print(f"  実Sの方が内(車番小)だった : {div_inner}/{div} = {div_inner/div*100:.1f}%")
        print(f"  実Sの方がS回数が多かった   : {div_higher_s}/{div} = {div_higher_s/div*100:.1f}%")
        print(f"  実Sの平均車番 {sum(div_scar)/len(div_scar):.2f} vs 予想先頭の平均車番 {sum(div_ncar)/len(div_ncar):.2f}")


if __name__ == "__main__":
    main()
