"""男子・中波乱(万車券率20-30%)で「◎頭外し(フェード)」の回収率が最も高くなる条件を層別探索。

analyze_upset_strategy(--men)で中波乱の◎頭外し8点が77.1%と全戦略中最良＝男子は中波乱で◎が
過剰人気。ここでは中波乱に絞り、フェード(◎頭外し top8)のROIを ◎の
  クラス(SS/S1/S2/A1/A2/A3) / 脚質(逃/両/追) / ラインでの位置(先頭/番手/後位/単騎) / 人気度(win_prob)
で層別し、100%に迫るゾーンがあるか探す。比較に◎頭固定8点も併記。

  PYTHONIOENCODING=utf-8 python scripts/analyze_men_fade.py
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
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.features.rider_narabi import compute_narabi_features
from src.model.feature_sets import men_features
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS
from src.model.upset import threshold_for
from src.backtest.walkforward import fold_boundaries


def _aux(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    leg, cls = {}, {}
    for rid, car, lt, cr in c.execute("SELECT race_id,car_number,leg_type,class_rank FROM entries"):
        leg[(rid, car)] = lt
        cls[(rid, car)] = cr
    payout = {}
    for rid, combo, pay in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta"):
        payout[rid] = (combo, pay)
    tmp = defaultdict(list)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        tmp[rid].append((li, pi, car))
    lines = {}
    for rid, rows in tmp.items():
        mx = max(li for li, _, _ in rows)
        ls = [[] for _ in range(mx + 1)]
        for li, pi, car in sorted(rows, key=lambda x: (x[0], x[1])):
            ls[li].append(car)
        lines[rid] = [x for x in ls if x]
    c.close()
    return leg, cls, payout, lines


def _line_pos(fav, lines):
    for m in lines or []:
        if fav in m:
            if len(m) == 1:
                return "単騎"
            i = m.index(fav)
            return "先頭" if i == 0 else ("番手" if i == 1 else "後位")
    return "単騎"


def _fade8(dist, fav):
    items = [(o, p) for o, p in dist.items() if o[0] != fav]
    items.sort(key=lambda op: -op[1])
    return set(o for o, _ in items[:8])


def _head8(dist, fav):
    items = [(o, p) for o, p in dist.items() if o[0] == fav]
    items.sort(key=lambda op: -op[1])
    return set(o for o, _ in items[:8])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    thr = threshold_for(False, 7)

    base = load_samples(args.db, field_size=7, features=PL_FEATURES_FULL)
    samples = augment_samples(base, args.db, men_features())
    narabi = compute_narabi_features(args.db)
    leg, cls, payout, lines = _aux(args.db)
    bounds = fold_boundaries(len(samples), n_folds=args.folds, warmup_frac=0.40, window="expanding")

    recs = []
    for a, b, c in bounds:
        model = train_gbdt(samples[a:b])
        for s in samples[b:c]:
            if s.race_id not in payout:
                continue
            st = model.strengths(s.X, s.car_numbers)
            fav = max(st, key=st.get)
            npos = {cc: narabi.get((s.race_id, cc), {}).get("narabi_pos") for cc in s.car_numbers}
            ln = lines.get(s.race_id)
            dist = corrected_trifecta_probs(st, npos, MEN_PARAMS, lines=ln)
            up = sum(p for p in dist.values() if p <= thr)
            if not (0.20 <= up < 0.30):        # 中波乱のみ
                continue
            wc = tuple(int(x) for x in payout[s.race_id][0].split("-"))
            pay = payout[s.race_id][1]
            recs.append({
                "fav": fav, "wc": wc, "pay": pay,
                "fade_hit": wc in _fade8(dist, fav), "head_hit": wc in _head8(dist, fav),
                "cls": cls.get((s.race_id, fav)), "leg": leg.get((s.race_id, fav)),
                "lpos": _line_pos(fav, ln), "wp": st[fav],
            })
    print(f"男子・中波乱(万車券率20-30%) {len(recs)}レース。◎頭外し(フェード)8点の回収率を層別\n")

    def roi(rs, hitkey):
        ret = sum(r["pay"] for r in rs if r[hitkey])
        return ret / (8 * 100 * len(rs)) * 100 if rs else 0

    print(f"全体: フェード {roi(recs,'fade_hit'):.1f}% (的中{sum(r['fade_hit'] for r in recs)/len(recs)*100:.1f}%) "
          f"vs ◎頭 {roi(recs,'head_hit'):.1f}%\n")

    def layer(name, keyfn, order=None):
        g = defaultdict(list)
        for r in recs:
            g[keyfn(r)].append(r)
        print(f"【{name}】 フェード回収率(的中) / 比較◎頭")
        keys = order or sorted(g, key=lambda k: -roi(g[k], "fade_hit"))
        for k in keys:
            rs = g.get(k, [])
            if len(rs) < 60:
                continue
            flag = "  ★" if roi(rs, "fade_hit") >= 90 else ""
            print(f"   {str(k):<8} n={len(rs):>4}  フェード {roi(rs,'fade_hit'):>6.1f}% "
                  f"(的中{sum(x['fade_hit'] for x in rs)/len(rs)*100:>4.1f}%)  ◎頭 {roi(rs,'head_hit'):>6.1f}%{flag}")
        print()

    layer("◎クラス", lambda r: r["cls"], ["SS", "S1", "S2", "A1", "A2", "A3"])
    layer("◎脚質", lambda r: r["leg"])
    layer("◎ライン位置", lambda r: r["lpos"], ["先頭", "番手", "後位", "単騎"])
    layer("◎人気度(win_prob)", lambda r: ("〜.25" if r["wp"] < .25 else ("‾.35" if r["wp"] < .35 else ".35〜")))


if __name__ == "__main__":
    main()
