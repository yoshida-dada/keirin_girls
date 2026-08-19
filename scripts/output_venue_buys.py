"""指定会場・最新開催日の全レースで、現行/新の買い目・点数・的中・回収を出力(男子)。

現行: ◎頭固定 三連単6点。
新  : 種別別 三連単(準決勝/決勝=◎頭6 / 選抜・一般=全体top10 / 予選・特選=◎頭8) + 三連複2点(保険)。
本番モデル(load_for)+補正確率で予測、実払戻(payouts_trifecta/_trio)で決済。

  PYTHONIOENCODING=utf-8 python scripts/output_venue_buys.py --venue-code 45   # 豊橋(男子)
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.feature_augment import augment_samples
from src.model.feature_sets import load_for, men_features
from src.features.rider_narabi import compute_narabi_features
from src.features.venue_region import venue_name
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS


def _role(name):
    if not name:
        return "標"
    for kw, lab in [("準決", "堅"), ("決勝", "堅"), ("予選", "標"),
                    ("選抜", "荒"), ("一般", "荒"), ("特選", "標")]:
        if kw in name:
            return lab
    return "標"


def _load(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    rn = {rid: nm for rid, nm in c.execute("SELECT race_id,race_name FROM races")}
    tri, trio = {}, {}
    for rid, combo, p in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta"):
        tri[rid] = (tuple(int(x) for x in combo.split("-")), p)
    for rid, combo, p in c.execute("SELECT race_id,combo,payout FROM payouts_trio"):
        trio[rid] = (frozenset(int(x) for x in combo.replace("=", "-").split("-")), p)
    tmp = defaultdict(list)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        tmp[rid].append((li, pi, car))
    c.close()
    lines = {}
    for rid, rows in tmp.items():
        mx = max(li for li, _, _ in rows)
        ls = [[] for _ in range(mx + 1)]
        for li, pi, car in sorted(rows, key=lambda x: (x[0], x[1])):
            ls[li].append(car)
        lines[rid] = [x for x in ls if x]
    return rn, tri, trio, lines


def _cs(t):
    return "-".join(str(x) for x in t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue-code", default="45")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    args = ap.parse_args()
    rn, tri, trio, lines = _load(args.db)
    model, elo, lbl = load_for(False)

    base = load_samples(args.db, field_size=[7, 9], features=PL_FEATURES_FULL)
    samples = augment_samples(base, args.db, model.feature_names)
    narabi = compute_narabi_features(args.db)

    vc = args.venue_code
    vs = [s for s in samples if s.race_id[:2] == vc and s.race_id in tri]
    if not vs:
        print("該当レースなし"); return
    latest = max(s.race_id[2:12] for s in vs)          # 初日8+日目2 の最新
    day = [s for s in vs if s.race_id[2:12] == latest]
    day.sort(key=lambda s: int(s.race_id[12:]))
    vname = venue_name(vc) or vc
    print(f"{vname}競輪  開催 初日{latest[:8]} {int(latest[8:10])}日目  {len(day)}レース\n")

    def topk(dist, k, pred=None):
        it = [(o, p) for o, p in dist.items() if (pred is None or pred(o))]
        it.sort(key=lambda op: -op[1])
        return [o for o, _ in it[:k]]

    tot = {"cur_pts": 0, "cur_ret": 0, "cur_hit": 0,
           "new_pts": 0, "new_ret": 0, "new_hit": 0, "new_tri_hit": 0, "new_trio_hit": 0}
    for s in day:
        st = model.strengths(s.X, s.car_numbers)
        fav = max(st, key=st.get)
        npos = {cc: narabi.get((s.race_id, cc), {}).get("narabi_pos") for cc in s.car_numbers}
        dist = corrected_trifecta_probs(st, npos, MEN_PARAMS, lines=lines.get(s.race_id))
        trp = defaultdict(float)
        for (a, b, c), p in dist.items():
            trp[frozenset((a, b, c))] += p
        combo, pay = tri[s.race_id]
        r = _role(rn.get(s.race_id))
        rno = int(s.race_id[12:])
        # 現行
        cur = topk(dist, 6, lambda o: o[0] == fav)
        cur_hit = tuple(combo) in cur
        # 新: 三連単(種別別) + 三連複2
        if r == "荒":
            ntri = topk(dist, 10)
        elif r == "堅":
            ntri = topk(dist, 6, lambda o: o[0] == fav)
        else:
            ntri = topk(dist, 8, lambda o: o[0] == fav)
        ntrio = [o for o, _ in sorted(trp.items(), key=lambda kv: -kv[1])[:2]]
        ntri_hit = tuple(combo) in ntri
        trio_hit = (s.race_id in trio) and (trio[s.race_id][0] in ntrio)
        new_pts = len(ntri) + len(ntrio)
        new_ret = (pay if ntri_hit else 0) + (trio[s.race_id][1] if trio_hit else 0)
        # 集計
        tot["cur_pts"] += len(cur); tot["cur_ret"] += pay if cur_hit else 0; tot["cur_hit"] += cur_hit
        tot["new_pts"] += new_pts; tot["new_ret"] += new_ret
        tot["new_hit"] += int(ntri_hit or trio_hit); tot["new_tri_hit"] += ntri_hit; tot["new_trio_hit"] += trio_hit

        rlab = {"堅": "準決勝/決勝", "荒": "選抜/一般", "標": "予選/特選"}[r]
        print(f"── R{rno}  {rn.get(s.race_id,'')}  [{rlab}]  ◎{fav}  結果{_cs(combo)}({pay:,}円)")
        print(f"   現行(◎頭6): {' '.join(_cs(x) for x in cur)}")
        print(f"        → {'的中 '+format(pay,',')+'円' if cur_hit else '不的中'}")
        print(f"   新・三連単({len(ntri)}点): {' '.join(_cs(x) for x in ntri)}")
        _tset = trio[s.race_id][0] if s.race_id in trio else None
        print(f"   新・三連複(2点): {' '.join('='.join(map(str,sorted(x))) for x in ntrio)}"
              + (f"  結果{'='.join(map(str,sorted(_tset)))}({trio[s.race_id][1]:,}円)" if _tset else "  (三連複払戻データ無し)"))
        nres = []
        if ntri_hit: nres.append(f"三連単的中 {pay:,}円")
        if trio_hit: nres.append(f"三連複的中 {trio[s.race_id][1]:,}円")
        print(f"        → {'／'.join(nres) if nres else '不的中'}")
        print()

    n = len(day)
    def roi(pts, ret): return ret / (pts * 100) * 100 if pts else 0
    print("=" * 50)
    print("【集計】")
    print(f"  現行(◎頭6)     点数計{tot['cur_pts']:>3} 的中{tot['cur_hit']}/{n} "
          f"払戻計{tot['cur_ret']:>6,}円 回収率{roi(tot['cur_pts'],tot['cur_ret']):.1f}%")
    print(f"  新(種別別+複2)  点数計{tot['new_pts']:>3} 的中{tot['new_hit']}/{n}"
          f"(三連単{tot['new_tri_hit']}/三連複{tot['new_trio_hit']}) "
          f"払戻計{tot['new_ret']:>6,}円 回収率{roi(tot['new_pts'],tot['new_ret']):.1f}%")


if __name__ == "__main__":
    main()
