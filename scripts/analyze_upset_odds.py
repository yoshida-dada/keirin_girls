"""波乱ポケット（地元◎/500m等）に実回収率エッジがあるかを確定オッズで検証。

analyze_upset の walk-forward(out-of-sample)予測に、確定オッズ(odds_final_trifecta)と
確定配当(payouts_trifecta)を結合し、条件別に:
  - 実◎勝率 vs 市場implied◎勝率（オッズ逆算・控除補正）… ＋なら◎が市場に過小評価＝妙味
  - ◎頭固定30点均等流しの回収率（takeout込みの実回収率）
  - ベースライン(全210点均等=控除率確認)
を出す。市場対比で正なら初めて「回収率に効く条件」と言える。

  PYTHONIOENCODING=utf-8 python scripts/analyze_upset_odds.py --db data/keirin.sqlite
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
from src.backtest.walkforward import fold_boundaries


def _aux(db: str):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    leg, pref = {}, {}
    for rid, car, lt, pf in c.execute("SELECT race_id,car_number,leg_type,prefecture FROM entries"):
        leg[(rid, car)] = lt
        pref[(rid, car)] = pf
    venue = {rid: v for rid, v in c.execute("SELECT race_id,venue_code FROM races")}
    payout = {}
    for rid, combo, pay in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta"):
        payout[rid] = (combo, pay)
    c.close()
    return leg, pref, venue, payout


def _load_odds(db: str, ids: set[str]) -> dict[str, dict[str, float]]:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for rid, combo, odds in c.execute("SELECT race_id,combo,odds FROM odds_final_trifecta"):
        if rid in ids and odds and odds > 0:
            out[rid][combo] = odds
    c.close()
    return out


def _implied_fav_win(odds: dict[str, float], fav: int) -> float | None:
    """210点オッズ→控除補正した implied確率で P(◎が1着) を出す。"""
    if not odds:
        return None
    inv = {c: 1.0 / o for c, o in odds.items()}
    s = sum(inv.values())
    if s <= 0:
        return None
    return sum(v for c, v in inv.items() if c.split("-")[0] == str(fav)) / s


def main():
    ap = argparse.ArgumentParser(description="波乱ポケットの回収率検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    feats = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES) + list(NARABI_KEYS)
    samples = augment_samples(base, args.db, feats)
    leg, pref, venue, payout = _aux(args.db)
    narabi = compute_narabi_features(args.db)
    bounds = fold_boundaries(len(samples), n_folds=args.folds, warmup_frac=0.40, window="expanding")

    recs = []
    for a, b, c in bounds:
        model = train_gbdt(samples[a:b])
        for s in samples[b:c]:
            if s.race_id not in payout:
                continue
            st = model.strengths(s.X, s.car_numbers)
            fav = max(st, key=st.get)
            nb = narabi.get((s.race_id, fav), {})
            v = venue.get(s.race_id, "")
            recs.append({
                "rid": s.race_id, "fav": fav, "p_fav": st[fav],
                "won": int(s.order[0] == fav),
                "地元": vr.is_home_pref(pref.get((s.race_id, fav)), v),
                "予想先頭": bool(nb.get("narabi_lead")),
                "bank": vm.bank_length(v),
                "lt": leg.get((s.race_id, fav)),
            })
    ids = {r["rid"] for r in recs}
    odds = _load_odds(args.db, ids)
    print(f"検証対象 {len(recs)}レース（オッズ・配当あり）\n")

    def report(name, rs):
        n = len(rs)
        if n < 30:
            print(f"  {name:<16} n={n} (少数, 省略)")
            return
        win = sum(r["won"] for r in rs) / n
        # 市場implied ◎勝率
        imps = [_implied_fav_win(odds.get(r["rid"], {}), r["fav"]) for r in rs]
        imps = [x for x in imps if x is not None]
        imp = sum(imps) / len(imps) if imps else float("nan")
        # ◎頭固定30点均等流し ROI
        stake = 30 * 100
        ret = 0
        for r in rs:
            if r["won"]:                       # ◎が1着なら的中combо(30点内)の配当
                combo, pay = payout[r["rid"]]
                ret += pay                     # payは100円あたり配当
        roi_head = ret / (stake * n) * 100
        # ベースライン: 全210点均等（控除率確認）
        stake_all = 210 * 100
        ret_all = sum(payout[r["rid"]][1] for r in rs)   # 必ず1点的中
        roi_all = ret_all / (stake_all * n) * 100
        print(f"  {name:<16} n={n:>5}  実◎勝率{win*100:>5.1f}%  市場implied{imp*100:>5.1f}%  "
              f"差{(win-imp)*100:>+5.1f}pt  ◎頭流しROI{roi_head:>6.1f}%  全210ROI{roi_all:>5.1f}%")

    print("条件別: 実◎勝率 − 市場implied◎勝率（＋＝◎が市場に過小評価＝妙味）／ ◎頭固定流し回収率")
    report("全体", recs)
    report("地元◎", [r for r in recs if r["地元"]])
    report("非地元◎", [r for r in recs if not r["地元"]])
    report("500mバンク", [r for r in recs if r["bank"] == 500])
    report("400mバンク", [r for r in recs if r["bank"] == 400])
    report("333mバンク", [r for r in recs if r["bank"] == 333])
    report("予想先頭◎", [r for r in recs if r["予想先頭"]])
    report("地元&予想先頭", [r for r in recs if r["地元"] and r["予想先頭"]])
    report("p_fav>=0.55", [r for r in recs if r["p_fav"] >= 0.55])
    report("地元&p>=0.5", [r for r in recs if r["地元"] and r["p_fav"] >= 0.5])


if __name__ == "__main__":
    main()
