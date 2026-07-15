"""市場ブレンド／エッジ比で「モデルが市場を上回る買い目」だけを買い、実現ROIを測る。

②帯(中オッズ・中確率)で、生EV(model_prob×odds)ではなく「モデル確率 vs 市場フェア確率」の乖離で
選定すればROIが100%に近づくかを検証する。市場フェア確率 q は implied_trifecta_probs で控除率除去済み。
  edge_ratio = model_prob / q   … これが 1/(1−控除率)≈1.33 を十分上回る買い目＝真に+EV候補。
  blend EV   = blend_loglinear(model,q,alpha)×odds … モデルの過信を市場へ収縮させた版。
オッズ帯で絞り込み、均等/ドッチングの実現ROIと的中率・買い目数(信頼度)を出す。out-of-sample。

  PYTHONIOENCODING=utf-8 python scripts/analyze_market_blend.py --db data/keirin.sqlite
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
from src.ev.market import implied_trifecta_probs, blend_loglinear

BUDGET = 1000


def _settle(chosen):
    """買い目(ComboRecord list)を均等/ドッチングで決済。信頼度のため n_bets/n_hits も返す。"""
    by_race = group_by_race(chosen)
    n_races = n_hits = n_bets = 0
    eq_stake = eq_ret = du_stake = du_ret = 0.0
    odds_sum = 0.0
    for recs in by_race.values():
        if not recs:
            continue
        n_races += 1
        n_bets += len(recs)
        odds_sum += sum(r.odds for r in recs)
        inv = sum(1.0 / r.odds for r in recs)
        eq_stake += len(recs) * 100
        du_stake += BUDGET
        win = next((r for r in recs if r.is_win), None)
        if win:
            n_hits += 1
            eq_ret += win.payout
            du_ret += (BUDGET * (1.0 / win.odds) / inv) * (win.payout / 100.0)
    if n_races == 0:
        return None
    return {"n_races": n_races, "n_bets": n_bets, "n_hits": n_hits,
            "hit_rate": n_hits / n_races, "pts": n_bets / n_races,
            "avg_odds": odds_sum / n_bets if n_bets else 0,
            "roi_eq": eq_ret / eq_stake if eq_stake else 0,
            "roi_du": du_ret / du_stake if du_stake else 0}


def _by_race_maps(records):
    """race_id → (recs, implied q dict)。市場フェア確率をレースごとに1回だけ計算。"""
    out = {}
    for rid, recs in group_by_race(records).items():
        odds = {r.combo: r.odds for r in recs}
        out[rid] = (recs, implied_trifecta_probs(odds))
    return out


def _pr(label, s):
    if s is None:
        print(f"{label:<34}(該当なし)"); return
    print(f"{label:<34}{s['n_bets']:>7}{s['pts']:>7.1f}{s['hit_rate']*100:>7.1f}%"
          f"{s['avg_odds']:>8.1f}{s['roi_eq']*100:>8.1f}%{s['roi_du']*100:>8.1f}%")


def main():
    ap = argparse.ArgumentParser(description="市場ブレンド/エッジ比の買い目選定ROI検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--test-frac", type=float, default=0.35)
    args = ap.parse_args()

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    feats31 = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
    samples = augment_samples(base, args.db, feats31)
    train, test = time_split(samples, args.test_frac)
    model = train_gbdt(train)
    records = build_records(args.db, model, [s.race_id for s in test])
    rmaps = _by_race_maps(records)
    print(f"検証 {len(rmaps)}レース / 31特徴lambdarank / out-of-sample")
    print("控除率除去後の市場フェア確率 q に対する乖離で選定。")
    print(f"{'戦略':<34}{'買い目':>7}{'点/R':>7}{'的中率':>8}{'平均odds':>8}{'ROI均等':>8}{'ROIドッチ':>9}")

    def collect(pred):
        out = []
        for recs, q in rmaps.values():
            out.extend([r for r in recs if pred(r, q.get(r.combo, 0.0))])
        return out

    # 生EV（ベースライン）
    _pr("生EV≥1.10（全オッズ）", _settle(collect(lambda r, q: r.ev >= 1.10)))

    # ブレンドEV（α掃引）: blendはレース単位なので別処理
    for alpha in (0.9, 0.7, 0.5):
        chosen = []
        for recs, q in rmaps.values():
            model_p = {r.combo: r.model_prob for r in recs}
            bl = blend_loglinear(model_p, q, alpha)
            odds = {r.combo: r.odds for r in recs}
            for r in recs:
                if bl.get(r.combo, 0.0) * odds.get(r.combo, 0.0) >= 1.10:
                    chosen.append(r)
        _pr(f"ブレンドEV≥1.10 (α={alpha})", _settle(chosen))

    # エッジ比 model_prob/q（全オッズ）
    for ratio in (1.3, 1.5, 2.0, 3.0):
        _pr(f"エッジ比 p/q≥{ratio}（全オッズ）",
            _settle(collect(lambda r, q, t=ratio: q > 0 and r.model_prob / q >= t)))

    # エッジ比 × 中オッズ帯（②帯の精査）
    print("  --- 中オッズ帯に限定（②帯: odds 8〜60）---")
    for ratio in (1.3, 1.5, 2.0):
        _pr(f"エッジ比 p/q≥{ratio} & odds∈[8,60]",
            _settle(collect(lambda r, q, t=ratio: 8 <= r.odds <= 60 and q > 0 and r.model_prob / q >= t)))
    print("  --- 足切り20倍以上（大穴寄り）---")
    for ratio in (1.5, 2.0, 3.0):
        _pr(f"エッジ比 p/q≥{ratio} & odds≥20",
            _settle(collect(lambda r, q, t=ratio: r.odds >= 20 and q > 0 and r.model_prob / q >= t)))

    # 最良候補のレースタイプ層別
    print("\n=== 参考: エッジ比 p/q≥1.5 & odds∈[8,60] のレースタイプ層別 ===")
    for rt in ("軸堅", "標準", "混戦"):
        sub = []
        for recs, q in rmaps.values():
            if recs and recs[0].race_type == rt:
                sub.extend([r for r in recs if 8 <= r.odds <= 60 and q.get(r.combo, 0) > 0
                            and r.model_prob / q[r.combo] >= 1.5])
        _pr(f"  {rt}", _settle(sub))


if __name__ == "__main__":
    main()
