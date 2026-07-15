"""金額配分（均等 vs ドッチング/合成オッズ）と「低オッズ足切り」の効果を実測する。

ユーザーの狙い: 的中率20%・的中時回収800%(≈合成オッズ8倍)。そのためにオッズの低い買い目を
足切りして合成オッズを高める案。均等買いで見えた「高オッズほどROI低下(favorite-longshot bias)」
が金額配分でも同じか、そして 20%/800% の作動点が実現するか＆その時の実現ROIを検証する。

方式:
  各レースで「モデル1着確率(三連単model_prob)上位N点」を土台にし、オッズ下限 floor で低オッズを足切り。
  残った買い目に予算B円をドッチング配分（stake_i ∝ 1/odds_i）→ 的中すれば概ね B×合成オッズ を回収。
  合成オッズ = 1/Σ(1/odds)。実現回収は実払戻(payout)で計算（odds≒payout/100）。
  比較のため均等配分ROIも出す。out-of-sample・リーク無し（build_records）。

  PYTHONIOENCODING=utf-8 python scripts/analyze_stake_allocation.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.features.tactics_features import TACTIC_NAMES
from src.model.evaluate import time_split
from src.backtest.bucket_analysis import build_records
from src.backtest.selection import group_by_race

BUDGET = 1000   # 1レースあたり予算（円・ドッチング配分の基準）


def select_topN_floor(recs, top_n, odds_floor):
    """model_prob上位 top_n 点から、オッズ<odds_floor を足切りした買い目を返す。"""
    ranked = sorted(recs, key=lambda r: -r.model_prob)[:top_n]
    return [r for r in ranked if r.odds >= odds_floor and r.odds > 0]


def evaluate_alloc(records, top_n, odds_floor):
    """全レースで topN+足切りを選び、均等/ドッチングのROIと的中率・平均合成オッズを集計。"""
    n_races = n_hits = 0
    eq_stake = eq_ret = 0.0
    du_stake = du_ret = 0.0
    synth_sum = pts = 0.0
    for recs in group_by_race(records).values():
        bought = select_topN_floor(recs, top_n, odds_floor)
        if not bought:
            continue
        n_races += 1
        pts += len(bought)
        inv = sum(1.0 / r.odds for r in bought)
        synth = (1.0 / inv) if inv > 0 else 0.0
        synth_sum += synth
        win = next((r for r in bought if r.is_win), None)
        # 均等配分: 各点100円
        eq_stake += len(bought) * 100
        # ドッチング配分: 合計 BUDGET 円、stake_i ∝ 1/odds_i
        du_stake += BUDGET
        if win:
            n_hits += 1
            eq_ret += win.payout                       # 実払戻(100円あたり)＝均等1点の回収
            s_win = BUDGET * (1.0 / win.odds) / inv     # ドッチングの的中点への配分
            du_ret += s_win * (win.payout / 100.0)      # 実払戻で回収
    if n_races == 0:
        return None
    return {
        "top_n": top_n, "floor": odds_floor, "n_races": n_races,
        "hit_rate": n_hits / n_races,
        "avg_synth": synth_sum / n_races,
        "pts_per_race": pts / n_races,
        "roi_equal": eq_ret / eq_stake if eq_stake else 0.0,
        "roi_dutch": du_ret / du_stake if du_stake else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="金額配分＋低オッズ足切りの効果検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--test-frac", type=float, default=0.35)
    args = ap.parse_args()

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    feats31 = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
    samples = augment_samples(base, args.db, feats31)
    train, test = time_split(samples, args.test_frac)
    model = train_gbdt(train)
    records = build_records(args.db, model, [s.race_id for s in test])
    print(f"検証 {len(set(r.race_id for r in records))}レース / 31特徴lambdarank / out-of-sample\n")

    print("土台=model_prob上位N点、floor=低オッズ足切り閾値。ドッチング予算1000円/レース。")
    print(f"{'topN':>5}{'floor':>7}{'点/R':>7}{'的中率':>8}{'平均合成ｵｯｽﾞ':>13}{'ROI均等':>9}{'ROIドッチ':>10}")
    for top_n in (12, 20, 30):
        for floor in (1, 5, 10, 20, 50, 100):
            r = evaluate_alloc(records, top_n, floor)
            if r is None:
                continue
            print(f"{r['top_n']:>5}{r['floor']:>7}{r['pts_per_race']:>7.1f}"
                  f"{r['hit_rate']*100:>7.1f}%{r['avg_synth']:>12.1f}"
                  f"{r['roi_equal']*100:>8.1f}%{r['roi_dutch']*100:>9.1f}%")
        print()

    print("★ユーザー目標 的中率≈20% & 合成≈8倍(回収800%) の作動点を探索:")
    best = None
    for top_n in range(4, 40):
        for floor in (1, 3, 5, 8, 10, 15, 20, 30, 50):
            r = evaluate_alloc(records, top_n, floor)
            if r and abs(r["hit_rate"] - 0.20) < 0.03:
                if best is None or abs(r["avg_synth"] - 8) < abs(best["avg_synth"] - 8):
                    best = r
    if best:
        print(f"  最も近い作動点: topN={best['top_n']} floor={best['floor']} → "
              f"的中率{best['hit_rate']*100:.1f}% / 平均合成{best['avg_synth']:.1f}倍 / "
              f"ROI均等{best['roi_equal']*100:.1f}% ROIドッチ{best['roi_dutch']*100:.1f}%")
        print("  ※ 的中率20%×合成8倍が『名目160%』でも、実現ROIがそれに届くかがポイント。")
    else:
        print("  的中率≈20%の作動点が見つからず（探索範囲外）。")


if __name__ == "__main__":
    main()
