"""券種比較: 三連単 vs 三連複 の回収率(男子7車)。三連複は的中率↑・配当↓でROIが変わるかを実測。

三連複の的中確率 = 補正三連単分布を「順不同3人」に集約（{a,b,c}=6並びの確率和）。
払戻は payouts_trio。三連単と同一レースでwalk-forward(out-of-sample)比較する。

  PYTHONIOENCODING=utf-8 python scripts/analyze_bet_types.py
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.features.rider_narabi import compute_narabi_features
from src.model.feature_sets import men_features
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS
from src.backtest.walkforward import fold_boundaries


def _load(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    tri, trio = {}, {}
    for rid, combo, pay in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta"):
        tri[rid] = (tuple(int(x) for x in combo.split("-")), pay)
    for rid, combo, pay in c.execute("SELECT race_id,combo,payout FROM payouts_trio"):
        trio[rid] = (frozenset(int(x) for x in combo.replace("=", "-").split("-")), pay)
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
    return tri, trio, lines


def _trio_probs(dist):
    """順序付き三連単dist → 順不同3人の確率 {frozenset({a,b,c}): p}。"""
    out = defaultdict(float)
    for (a, b, c), p in dist.items():
        out[frozenset((a, b, c))] += p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    base = load_samples(args.db, field_size=7, features=PL_FEATURES_FULL)
    samples = augment_samples(base, args.db, men_features())
    narabi = compute_narabi_features(args.db)
    tri, trio, lines = _load(args.db)
    bounds = fold_boundaries(len(samples), n_folds=args.folds, warmup_frac=0.40, window="expanding")

    recs = []
    for a, b, c in bounds:
        model = train_gbdt(samples[a:b])
        for s in samples[b:c]:
            if s.race_id not in tri or s.race_id not in trio:
                continue
            st = model.strengths(s.X, s.car_numbers)
            npos = {cc: narabi.get((s.race_id, cc), {}).get("narabi_pos") for cc in s.car_numbers}
            dist = corrected_trifecta_probs(st, npos, MEN_PARAMS, lines=lines.get(s.race_id))
            recs.append({"dist": dist, "tp": _trio_probs(dist),
                         "tri": tri[s.race_id], "trio": trio[s.race_id]})
    print(f"男子7車・三連単/三連複 両方あり {len(recs)}レース（out-of-sample）\n")

    def tri_roi(rs, k):
        ret = hit = 0
        for r in rs:
            buys = [o for o, _ in sorted(r["dist"].items(), key=lambda kv: -kv[1])[:k]]
            if r["tri"][0] in buys:
                ret += r["tri"][1]; hit += 1
        n = len(rs)
        return ret / (k * 100 * n) * 100, hit / n * 100

    def trio_roi(rs, k):
        ret = hit = 0
        for r in rs:
            buys = [o for o, _ in sorted(r["tp"].items(), key=lambda kv: -kv[1])[:k]]
            if r["trio"][0] in buys:
                ret += r["trio"][1]; hit += 1
        n = len(rs)
        return ret / (k * 100 * n) * 100, hit / n * 100

    print("【三連単】点数→回収率(的中率)")
    for k in (4, 6, 8, 12):
        roi, h = tri_roi(recs, k)
        print(f"   {k:>2}点: {roi:>6.1f}%  (的中{h:.1f}%)")
    print("\n【三連複】点数→回収率(的中率)  ※7車の三連複は全35通り")
    for k in (2, 3, 4, 6, 10):
        roi, h = trio_roi(recs, k)
        print(f"   {k:>2}点: {roi:>6.1f}%  (的中{h:.1f}%)")

    # 全点買い(控除率確認): 三連単210 / 三連複35
    allt = sum(r["tri"][1] for r in recs) / (210 * 100 * len(recs)) * 100
    allc = sum(r["trio"][1] for r in recs) / (35 * 100 * len(recs)) * 100
    print(f"\n全点買いROI（控除率確認）: 三連単210点 {allt:.1f}% / 三連複35点 {allc:.1f}%")


if __name__ == "__main__":
    main()
