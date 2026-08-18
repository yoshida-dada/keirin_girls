"""万車券率(upset_prob)の帯ごとに買い方(戦略)を変えると回収率が変わるかを実オッズで検証。

仮説: 固い(万車券率<20%)は点数を絞った◎頭固定、荒れ(>=30%)は◎/◎ラインを崩す狙いが良いか。
男子(--men)は**実際のライン**があるので「ライン信頼(固い)」「ライン崩れ(荒れ)」を本来の意味で検証できる。

  ガールズ7車 / 男子7車を walk-forward(out-of-sample) で:
    万車券率 = corrected三連単で p<=しきい の目の合計（本番と同一, 車立て別しきい）
    帯 <20% / 20-30% / >=30% × 戦略の 回収率/的中率/点数 を比較。

戦略(各100円均等):
  tight6  : ◎頭固定・補正top6
  head8   : ◎頭固定・補正top8
  wide12  : 全体・補正top12
  antiFav8: ◎を1着から外す・補正top8
  (男子のみ) lineTrust8 : ◎ラインで1-2着決着・補正top8（ライン信頼＝固い向け）
  (男子のみ) lineBreak8 : ◎ライン以外が1着・補正top8（ライン崩れ＝荒れ向け）

  PYTHONIOENCODING=utf-8 python scripts/analyze_upset_strategy.py --men
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
from src.model.feature_sets import men_features
from src.model.himo_adjust import corrected_trifecta_probs, DEFAULT_PARAMS, MEN_PARAMS
from src.model.upset import threshold_for
from src.backtest.walkforward import fold_boundaries


def _load_payout(db, ids):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    out = {}
    for rid, combo, pay in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta"):
        if rid in ids:
            out[rid] = (combo, pay)
    c.close()
    return out


def _load_lines(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    tmp = defaultdict(list)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        tmp[rid].append((li, pi, car))
    c.close()
    out = {}
    for rid, rows in tmp.items():
        mx = max(li for li, _, _ in rows)
        ls = [[] for _ in range(mx + 1)]
        for li, pi, car in sorted(rows, key=lambda x: (x[0], x[1])):
            ls[li].append(car)
        out[rid] = [x for x in ls if x]
    return out


def _topk(dist, k, pred=None):
    items = [(o, p) for o, p in dist.items() if (pred is None or pred(o))]
    items.sort(key=lambda op: -op[1])
    return [o for o, _ in items[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--men", action="store_true", help="男子で検証（既定=ガールズ）")
    ap.add_argument("--db")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    is_girls = not args.men
    db = args.db or str(DATA_DIR / ("keirin.sqlite" if is_girls else "keirin_men.sqlite"))
    params = DEFAULT_PARAMS if is_girls else MEN_PARAMS
    thr = threshold_for(is_girls, 7)
    print(f"{'ガールズ' if is_girls else '男子'}7車 / 万車券しきい {thr}\n")

    base = load_samples(db, field_size=7, features=PL_FEATURES_FULL)
    feats = (list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES) + list(NARABI_KEYS)
             if is_girls else men_features())
    samples = augment_samples(base, db, feats)
    narabi = compute_narabi_features(db)
    lines_of = {} if is_girls else _load_lines(db)
    bounds = fold_boundaries(len(samples), n_folds=args.folds, warmup_frac=0.40, window="expanding")

    recs = []
    for a, b, c in bounds:
        model = train_gbdt(samples[a:b])
        for s in samples[b:c]:
            st = model.strengths(s.X, s.car_numbers)
            fav = max(st, key=st.get)
            npos = {cc: narabi.get((s.race_id, cc), {}).get("narabi_pos") for cc in s.car_numbers}
            ln = lines_of.get(s.race_id)
            dist = corrected_trifecta_probs(st, npos, params, lines=ln)
            up = sum(p for p in dist.values() if p <= thr)
            fav_line = next((set(m) for m in (ln or []) if fav in m), {fav})
            recs.append({"rid": s.race_id, "fav": fav, "dist": dist, "up": up, "fline": fav_line})
    payout = _load_payout(db, {r["rid"] for r in recs})
    recs = [r for r in recs if r["rid"] in payout]

    strategies = {
        "tight6(◎頭6)": lambda r: _topk(r["dist"], 6, lambda o: o[0] == r["fav"]),
        "head8(◎頭8)": lambda r: _topk(r["dist"], 8, lambda o: o[0] == r["fav"]),
        "wide12(全体12)": lambda r: _topk(r["dist"], 12),
        "antiFav8(◎頭外し8)": lambda r: _topk(r["dist"], 8, lambda o: o[0] != r["fav"]),
    }
    if not is_girls:
        strategies["lineTrust8(ライン信頼)"] = lambda r: _topk(
            r["dist"], 8, lambda o: o[0] in r["fline"] and o[1] in r["fline"])
        strategies["lineBreak8(ライン崩れ)"] = lambda r: _topk(
            r["dist"], 8, lambda o: o[0] not in r["fline"])

    bands = [("固い <20%", lambda u: u < .20), ("中 20-30%", lambda u: .20 <= u < .30),
             ("荒れ >=30%", lambda u: u >= .30)]
    print(f"{len(recs)}レース。万車券率帯 × 戦略の回収率/的中率\n")
    for bl, cond in bands:
        rs = [r for r in recs if cond(r["up"])]
        if not rs:
            continue
        print(f"■ {bl}  n={len(rs)}")
        print(f"   {'戦略':<20}{'点数':>5}{'回収率':>9}{'的中率':>9}")
        for name, fn in strategies.items():
            ret = hit = pts = 0
            for r in rs:
                buys = set(fn(r))
                pts += len(buys)
                wc = tuple(int(x) for x in payout[r["rid"]][0].split("-"))
                if wc in buys:
                    ret += payout[r["rid"]][1]; hit += 1
            n = len(rs)
            avg = pts / n
            roi = ret / (avg * 100 * n) * 100 if avg else 0
            print(f"   {name:<20}{avg:>5.1f}{roi:>8.1f}%{hit/n*100:>8.1f}%")
        print()


if __name__ == "__main__":
    main()
