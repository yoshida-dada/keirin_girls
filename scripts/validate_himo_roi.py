"""補正紐×市場過小評価ポケットで、絞った◎頭固定の回収率が100%を超えるか検証。

◎は市場に過小評価(予想先頭/500m/地元で+3〜4.5pt, analyze_upset_odds)。かつ補正紐で2着選定が
改善(validate_himo_adjust)。両者を合わせ、ポケット別に「◎1着固定・補正確率top-K点」を買った
ときの回収率・的中率・点数を確定オッズで測る。PL選定との差で補正の寄与も見る。

  PYTHONIOENCODING=utf-8 python scripts/validate_himo_roi.py --db data/keirin.sqlite
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
from src.features.tactics_features import TACTIC_NAMES
from src.features.rider_narabi import NARABI_KEYS, compute_narabi_features
from src.features import venue_region as vr
from src.features import venue_meta as vm
from src.model.himo_adjust import corrected_trifecta_probs, PL_PARAMS, DEFAULT_PARAMS
from src.backtest.walkforward import fold_boundaries


def _aux(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    pref = {}
    for rid, car, pf in c.execute("SELECT race_id,car_number,prefecture FROM entries"):
        pref[(rid, car)] = pf
    venue = {rid: v for rid, v in c.execute("SELECT race_id,venue_code FROM races")}
    payout = {}
    for rid, combo, pay in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta"):
        payout[rid] = (combo, pay)
    c.close()
    return pref, venue, payout


def _headfix_topk(dist, fav, k):
    """◎(fav)1着固定の補正確率top-K combo（tupleのリスト）。"""
    cand = [(o, p) for o, p in dist.items() if o[0] == fav]
    cand.sort(key=lambda op: -op[1])
    return [o for o, _ in cand[:k]]


def main():
    ap = argparse.ArgumentParser(description="補正紐×ポケットの回収率検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    feats = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES) + list(NARABI_KEYS)
    samples = augment_samples(base, args.db, feats)
    narabi = compute_narabi_features(args.db)
    pref, venue, payout = _aux(args.db)
    bounds = fold_boundaries(len(samples), n_folds=args.folds, warmup_frac=0.40, window="expanding")

    recs = []
    for a, b, c in bounds:
        model = train_gbdt(samples[a:b])
        for s in samples[b:c]:
            if s.race_id not in payout or len(s.order) < 3:
                continue
            st = model.strengths(s.X, s.car_numbers)
            fav = max(st, key=st.get)
            npos = {car: narabi.get((s.race_id, car), {}).get("narabi_pos") for car in s.car_numbers}
            v = venue.get(s.race_id, "")
            recs.append({
                "rid": s.race_id, "fav": fav, "order": tuple(s.order[:3]), "npos": npos, "st": st,
                "予想先頭": bool(narabi.get((s.race_id, fav), {}).get("narabi_lead")),
                "500m": vm.bank_length(v) == 500, "333m": vm.bank_length(v) == 333,
                "地元": vr.is_home_pref(pref.get((s.race_id, fav)), v),
            })

    # 補正確率とPL確率を各レース一度だけ計算
    for r in recs:
        r["dcx"] = corrected_trifecta_probs(r["st"], r["npos"], DEFAULT_PARAMS)
        r["dpl"] = corrected_trifecta_probs(r["st"], r["npos"], PL_PARAMS)

    def roi(rs, dist_key, k):
        stake = k * 100
        ret = hit = 0
        for r in rs:
            buys = set(_headfix_topk(r[dist_key], r["fav"], k))
            wc = tuple(int(x) for x in payout[r["rid"]][0].split("-"))
            if wc in buys:
                ret += payout[r["rid"]][1]
                hit += 1
        n = len(rs)
        return ret / (stake * n) * 100, hit / n * 100

    pockets = {
        "全体": recs,
        "予想先頭◎": [r for r in recs if r["予想先頭"]],
        "500m": [r for r in recs if r["500m"]],
        "地元◎": [r for r in recs if r["地元"]],
        "予想先頭 or 500m or 地元": [r for r in recs if r["予想先頭"] or r["500m"] or r["地元"]],
        "予想先頭&(500m or 地元)": [r for r in recs if r["予想先頭"] and (r["500m"] or r["地元"])],
    }
    print(f"検証 {len(recs)}レース。◎頭固定・補正確率top-K点の回収率(ROI)と的中率。\n")
    for name, rs in pockets.items():
        if len(rs) < 40:
            print(f"■ {name} (n={len(rs)}) 少数のため省略"); continue
        print(f"■ {name}  n={len(rs)}")
        print(f"   {'点数K':>5}{'ROI(補正)':>11}{'的中(補正)':>11}{'ROI(PL)':>10}{'的中(PL)':>10}")
        for k in (3, 4, 6, 8, 12):
            rc, hc = roi(rs, "dcx", k)
            rp, hp = roi(rs, "dpl", k)
            flag = "  ★>100%" if rc >= 100 else ""
            print(f"   {k:>5}{rc:>10.1f}%{hc:>10.1f}%{rp:>9.1f}%{hp:>9.1f}%{flag}")
        print()


if __name__ == "__main__":
    main()
