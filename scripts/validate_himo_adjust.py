"""条件付き紐補正(himo_adjust)の精度検証。PL vs 補正後を out-of-sample で比較。

walk-forward で各testレースの strengths/並び/実三連単を作り、
  - 補正パラメータ(t2,t3,mark)を fold0-2 の三連単log-lossで grid探索し確定
  - 残り fold3-4(hold-out)で PL vs 補正後を比較: 三連単log-loss / 2着marginal top-k / 三連単top-10
補正が hold-out で log-loss↓・的中↑ かつ fold一致すれば採用に足る。

  PYTHONIOENCODING=utf-8 python scripts/validate_himo_adjust.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.features.tactics_features import TACTIC_NAMES
from src.features.rider_narabi import NARABI_KEYS, compute_narabi_features
from src.model.himo_adjust import combo_logprob, corrected_trifecta_probs, PL_PARAMS
from src.backtest.walkforward import fold_boundaries


def _second_topk(dist, actual2, k):
    marg = {}
    for (a, b, c), p in dist.items():
        marg[b] = marg.get(b, 0.0) + p
    top = [x for x, _ in sorted(marg.items(), key=lambda kv: -kv[1])[:k]]
    return int(actual2 in top)


def _tri_topk(dist, actual3, k):
    top = [x for x, _ in sorted(dist.items(), key=lambda kv: -kv[1])[:k]]
    return int(tuple(actual3) in top)


def main():
    ap = argparse.ArgumentParser(description="条件付き紐補正の検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    feats = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES) + list(NARABI_KEYS)
    samples = augment_samples(base, args.db, feats)
    narabi = compute_narabi_features(args.db)
    bounds = fold_boundaries(len(samples), n_folds=args.folds, warmup_frac=0.40, window="expanding")

    recs = []   # (fold, strengths, npos, order3)
    for fi, (a, b, c) in enumerate(bounds):
        model = train_gbdt(samples[a:b])
        for s in samples[b:c]:
            if len(s.order) < 3:
                continue
            st = model.strengths(s.X, s.car_numbers)
            npos = {car: narabi.get((s.race_id, car), {}).get("narabi_pos") for car in s.car_numbers}
            recs.append((fi, st, npos, tuple(s.order[:3])))
    nf = len(bounds)
    tune = [r for r in recs if r[0] < max(1, nf - 2)]      # fold 0..nf-3
    hold = [r for r in recs if r[0] >= max(1, nf - 2)]     # 後半2fold
    print(f"records {len(recs)} / tune {len(tune)} / hold-out {len(hold)}\n")

    # --- grid tune by trifecta log-loss on tune ---
    grid = [{"t2": t2, "t3": t3, "mark": mk}
            for t2, t3, mk in product([1.0, 1.15, 1.3, 1.5, 1.7], [1.0, 1.2, 1.4], [0.0, 0.25, 0.5])]

    def mean_ll(rs, params):
        return -sum(combo_logprob(st, npos, o3, params) for _, st, npos, o3 in rs) / len(rs)

    best = min(grid, key=lambda p: mean_ll(tune, p))
    print(f"最良補正パラメータ(tune log-loss): t2={best['t2']} t3={best['t3']} mark={best['mark']}")
    print(f"  tune log-loss  PL {mean_ll(tune, PL_PARAMS):.4f} → 補正 {mean_ll(tune, best):.4f}\n")

    # --- hold-out 比較 ---
    print("【hold-out比較】PL vs 補正後")
    print(f"  三連単log-loss  PL {mean_ll(hold, PL_PARAMS):.4f} → 補正 {mean_ll(hold, best):.4f} "
          f"(Δ{mean_ll(hold, best)-mean_ll(hold, PL_PARAMS):+.4f})")

    # top-k 的中（full dist が要るので hold のみ2パス）
    agg = {"pl_2t1": 0, "pl_2t2": 0, "pl_2t3": 0, "pl_t10": 0,
           "cx_2t1": 0, "cx_2t2": 0, "cx_2t3": 0, "cx_t10": 0}
    per_fold = {}
    for fi, st, npos, o3 in hold:
        dpl = corrected_trifecta_probs(st, npos, PL_PARAMS)
        dcx = corrected_trifecta_probs(st, npos, best)
        a2 = o3[1]
        vals = {
            "pl_2t1": _second_topk(dpl, a2, 1), "cx_2t1": _second_topk(dcx, a2, 1),
            "pl_2t2": _second_topk(dpl, a2, 2), "cx_2t2": _second_topk(dcx, a2, 2),
            "pl_2t3": _second_topk(dpl, a2, 3), "cx_2t3": _second_topk(dcx, a2, 3),
            "pl_t10": _tri_topk(dpl, o3, 10), "cx_t10": _tri_topk(dcx, o3, 10),
        }
        for k, v in vals.items():
            agg[k] += v
        pf = per_fold.setdefault(fi, {k: 0 for k in agg} | {"n": 0})
        for k, v in vals.items():
            pf[k] += v
        pf["n"] += 1
    n = len(hold)
    def pct(k): return agg[k] / n * 100
    print(f"  2着 top1的中   PL {pct('pl_2t1'):.1f}% → 補正 {pct('cx_2t1'):.1f}%  (Δ{pct('cx_2t1')-pct('pl_2t1'):+.1f})")
    print(f"  2着 top2的中   PL {pct('pl_2t2'):.1f}% → 補正 {pct('cx_2t2'):.1f}%  (Δ{pct('cx_2t2')-pct('pl_2t2'):+.1f})")
    print(f"  2着 top3的中   PL {pct('pl_2t3'):.1f}% → 補正 {pct('cx_2t3'):.1f}%  (Δ{pct('cx_2t3')-pct('pl_2t3'):+.1f})")
    print(f"  三連単top10的中 PL {pct('pl_t10'):.1f}% → 補正 {pct('cx_t10'):.1f}%  (Δ{pct('cx_t10')-pct('pl_t10'):+.1f})")
    print("\n  per-fold Δ(補正−PL):")
    for fi, pf in sorted(per_fold.items()):
        m = pf["n"]
        print(f"    fold{fi} n={m}: 2着top1 {(pf['cx_2t1']-pf['pl_2t1'])/m*100:+.1f} / "
              f"2着top3 {(pf['cx_2t3']-pf['pl_2t3'])/m*100:+.1f} / tri10 {(pf['cx_t10']-pf['pl_t10'])/m*100:+.1f}")


if __name__ == "__main__":
    main()
