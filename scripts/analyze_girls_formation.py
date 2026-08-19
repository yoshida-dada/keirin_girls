"""ガールズの買い目「型別」実測（◎頭/◎2着/◎3着/全体）を walk-forward で測り、
formation_stats_girls.json に保存する。男子はライン展開分岐だが、ガールズはラインが無いので
◎軸中心のシンプルな構築（各型の top-K を固定点数で買った場合の的中率・回収率）。

型: ◎頭(◎を1着固定 top6) / ◎2着(◎を2着固定 top6) / ◎3着(◎を3着固定 top6) / 全体(補正top8)。
本番と同じ corrected_trifecta_probs（himo補正）で選定、実払戻で決済。

  PYTHONIOENCODING=utf-8 python scripts/analyze_girls_formation.py --emit
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.model.feature_sets import girls_features
from src.features.rider_narabi import compute_narabi_features
from src.model.himo_adjust import corrected_trifecta_probs, DEFAULT_PARAMS
from src.model.upset import threshold_for
from src.backtest.walkforward import fold_boundaries

KINDS = {"◎頭6": (6, lambda o, f: o[0] == f),
         "◎2着6": (6, lambda o, f: o[1] == f),
         "◎3着6": (6, lambda o, f: o[2] == f),
         "全体8": (8, lambda o, f: True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--emit", action="store_true")
    args = ap.parse_args()
    thr = threshold_for(True, 7)

    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    payout = {r: (tuple(int(x) for x in combo.split("-")), p)
              for r, combo, p in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta")}
    c.close()

    base = load_samples(args.db, field_size=7, features=PL_FEATURES_FULL)
    samples = augment_samples(base, args.db, girls_features())
    narabi = compute_narabi_features(args.db)
    bounds = fold_boundaries(len(samples), n_folds=args.folds, warmup_frac=0.40, window="expanding")

    agg = {k: {"hit": 0, "stake": 0, "ret": 0, "pts": 0} for k in KINDS}
    # 万車券率帯別の◎頭(固い絞る/荒れ広げる)も測る
    band = {"固<20": {"hit": 0, "stake": 0, "ret": 0}, "荒>=30": {"hit": 0, "stake": 0, "ret": 0}}
    n = 0
    for a, b, c in bounds:
        model = train_gbdt(samples[a:b])
        for s in samples[b:c]:
            if s.race_id not in payout:
                continue
            st = model.strengths(s.X, s.car_numbers)
            fav = max(st, key=st.get)
            npos = {cc: narabi.get((s.race_id, cc), {}).get("narabi_pos") for cc in s.car_numbers}
            dist = corrected_trifecta_probs(st, npos, DEFAULT_PARAMS)
            combo, pay = payout[s.race_id]
            n += 1
            for k, (kk, pred) in KINDS.items():
                buys = [o for o, _ in sorted(dist.items(), key=lambda kv: -kv[1])
                        if pred(o, fav)][:kk]
                agg[k]["pts"] += len(buys); agg[k]["stake"] += 100 * len(buys)
                if tuple(combo) in buys:
                    agg[k]["hit"] += 1; agg[k]["ret"] += pay
            up = sum(p for p in dist.values() if p <= thr) if thr else 0.25
            kk = 6 if up < 0.20 else (8 if up >= 0.30 else 7)
            hb = [o for o, _ in sorted(dist.items(), key=lambda kv: -kv[1]) if o[0] == fav][:kk]
            grp = "固<20" if up < 0.20 else ("荒>=30" if up >= 0.30 else None)
            if grp:
                band[grp]["stake"] += 100 * len(hb)
                if tuple(combo) in hb:
                    band[grp]["hit"] += 1; band[grp]["ret"] += pay

    print(f"ガールズ7車 {n}レース（out-of-sample）\n")
    print(f"{'型':<8}{'平均点数':>8}{'的中率':>8}{'回収率':>8}")
    emit = {}
    for k in KINDS:
        v = agg[k]
        hit = v["hit"] / n * 100
        roi = v["ret"] / v["stake"] * 100 if v["stake"] else 0
        print(f"  {k:<8}{v['pts']/n:>7.1f}{hit:>7.1f}%{roi:>7.1f}%")
        emit[k] = {"hit": round(v["hit"] / n, 4), "roi": round(v["ret"] / v["stake"], 4) if v["stake"] else 0,
                   "points": round(v["pts"] / n, 1), "n": n}
    print("\n万車券率帯別 ◎頭(固い=6点絞る/荒れ=8点広げる):")
    for g, v in band.items():
        if v["stake"]:
            print(f"  {g}: 回収率{v['ret']/v['stake']*100:.1f}% 的中{v['hit']}")
    if args.emit:
        p = Path(__file__).resolve().parents[1] / "src" / "model" / "formation_stats_girls.json"
        p.write_text(json.dumps(
            {"note": "walk-forward out-of-sample の実測。hit=無条件の的中率, roi=回収率(控除率25%で上限75%)。"
                     "ガールズは◎軸中心。買い目の推奨ではない。",
             "kinds": emit}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n保存: {p}")


if __name__ == "__main__":
    main()
