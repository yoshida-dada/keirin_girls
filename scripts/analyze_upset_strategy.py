"""万車券率(upset_prob)の帯ごとに、買い方(戦略)を変えると回収率が変わるかを実オッズで検証。

仮説: 固い(万車券率<20%)は点数を絞った◎頭固定が良く、荒れ(>30%)は◎を頭から外す
(ライン崩れ狙い)が良いかもしれない。ガールズ7車をwalk-forward(out-of-sample)で:
  各レースの万車券率= corrected三連単で p<=しきい(0.00919) の目の合計（本番と同一）
  帯 <20% / 20-30% / >=30% × 戦略の 回収率/的中率/点数 を比較。

戦略(各100円均等):
  tight6 : ◎頭固定・補正確率top6
  head8  : ◎頭固定・補正確率top8
  wide12 : 全体・補正確率top12（◎頭に限らない）
  anti8  : ◎を1着から外した目・補正確率top8（荒れ/ライン崩れ狙い）

  PYTHONIOENCODING=utf-8 python scripts/analyze_upset_strategy.py --db data/keirin.sqlite
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
from src.model.himo_adjust import corrected_trifecta_probs
from src.model.upset import threshold_for
from src.backtest.walkforward import fold_boundaries


def _load(db, ids):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    payout = {}
    for rid, combo, pay in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta"):
        if rid in ids:
            payout[rid] = (combo, pay)
    c.close()
    return payout


def _topk(dist, k, pred=None):
    items = [(o, p) for o, p in dist.items() if (pred is None or pred(o))]
    items.sort(key=lambda op: -op[1])
    return [o for o, _ in items[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    thr = threshold_for(True, 7)

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    feats = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES) + list(NARABI_KEYS)
    samples = augment_samples(base, args.db, feats)
    narabi = compute_narabi_features(args.db)
    bounds = fold_boundaries(len(samples), n_folds=args.folds, warmup_frac=0.40, window="expanding")

    recs = []
    for a, b, c in bounds:
        model = train_gbdt(samples[a:b])
        for s in samples[b:c]:
            st = model.strengths(s.X, s.car_numbers)
            fav = max(st, key=st.get)
            npos = {cc: narabi.get((s.race_id, cc), {}).get("narabi_pos") for cc in s.car_numbers}
            dist = corrected_trifecta_probs(st, npos)
            up = sum(p for p in dist.values() if p <= thr)
            recs.append({"rid": s.race_id, "fav": fav, "dist": dist, "up": up})
    payout = _load(args.db, {r["rid"] for r in recs})
    recs = [r for r in recs if r["rid"] in payout]

    strategies = {
        "tight6(◎頭6)": lambda r: _topk(r["dist"], 6, lambda o: o[0] == r["fav"]),
        "head8(◎頭8)": lambda r: _topk(r["dist"], 8, lambda o: o[0] == r["fav"]),
        "wide12(全体12)": lambda r: _topk(r["dist"], 12),
        "anti8(◎頭外し8)": lambda r: _topk(r["dist"], 8, lambda o: o[0] != r["fav"]),
    }
    bands = [("固い <20%", lambda u: u < .20), ("中 20-30%", lambda u: .20 <= u < .30),
             ("荒れ >=30%", lambda u: u >= .30)]

    print(f"ガールズ7車 {len(recs)}レース。万車券率(しきい{thr})帯 × 戦略の回収率/的中率\n")
    for bl, cond in bands:
        rs = [r for r in recs if cond(r["up"])]
        if not rs:
            continue
        print(f"■ {bl}  n={len(rs)}")
        print(f"   {'戦略':<16}{'点数':>5}{'回収率':>9}{'的中率':>9}")
        for name, fn in strategies.items():
            ret = hit = pts = 0
            for r in rs:
                buys = set(fn(r))
                pts += len(buys)
                wc = tuple(int(x) for x in payout[r["rid"]][0].split("-"))
                if wc in buys:
                    ret += payout[r["rid"]][1]; hit += 1
            n = len(rs)
            avgpts = pts / n
            roi = ret / (avgpts * 100 * n) * 100 if avgpts else 0
            print(f"   {name:<16}{avgpts:>5.1f}{roi:>8.1f}%{hit/n*100:>8.1f}%")
        print()


if __name__ == "__main__":
    main()
