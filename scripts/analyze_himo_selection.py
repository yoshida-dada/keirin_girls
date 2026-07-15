"""三連単「紐選定戦略」の実現ROI/点数/的中率を比較評価する（S5・課題A派生）。

学習期間でモデルを訓練し、検証期間（未来・out-of-sample）で build_records → records を作り、
selection.py の各戦略×パラメータ（EV閾値/紐点数/紐の並べ方 order）を回して
 全体ROI・点数・的中率 と レースタイプ層別（軸堅/標準/混戦） を表化する。

主な問い:
  1. 現行ベースライン（EV≥thr 全買い）に対し、頭固定フォーメーションで点数効率は上がるか
  2. **紐の並べ方 order='ev'（高EV紐）vs order='prob'（高確率紐）で回収率はどう変わるか**（ユーザー仮説）
  3. **同確率帯で高オッズ紐を選ぶ効果**（確率が近い候補群に絞った high-odds vs low-odds 比較）
  4. 高オッズ紐のEV条件付き取り込み（longshot_himo）はベースを改善するか

  # Windows は文字化け回避のため PYTHONIOENCODING=utf-8 を付ける
  set PYTHONIOENCODING=utf-8 && python scripts/analyze_himo_selection.py --db data/keirin.sqlite

★控除率≈25%のためROIは1.0未満が基本。目的は「どの戦略が相対的に高いROI/点数効率か」を示すこと。
 本体は同じ呼び出しでモデルを31特徴のものへ差し替えて再実行する（末尾コメント参照）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # Windows のコンソール文字化け対策
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES, PL_FEATURES_FULL
from src.model.train_pl import train_pl
from src.model.evaluate import time_split
from src.model.race_type import JIKU, STANDARD, CHAOS
from src.backtest.bucket_analysis import build_records
from src.backtest import selection as sel

RACE_TYPES = [JIKU, STANDARD, CHAOS]
MIN_BETS = 30  # これ未満は母数不足として注記する


# --------------------------------------------------------------------------
# 表示ヘルパ
# --------------------------------------------------------------------------
def _fmt_roi(v) -> str:
    return f"{v*100:5.1f}%" if v is not None else "   -  "


def _fmt_rate(v) -> str:
    return f"{v*100:5.1f}%" if v is not None else "   -  "


def _row(label: str, res: dict) -> str:
    note = "  ※母数少" if 0 < res["n_bets"] < MIN_BETS else ""
    return (f"{label:<26}{res['n_bets']:>7}{res['n_races']:>7}"
            f"{res['points_per_race'] or 0:>8.2f}{res['n_hits']:>6}"
            f"{_fmt_rate(res['hit_rate']):>9}{_fmt_roi(res['roi']):>10}{note}")


def _header(title: str) -> None:
    print(f"\n{title}")
    print(f"{'戦略':<26}{'点数':>7}{'R数':>7}{'点/R':>8}{'的中':>6}{'的中率':>9}{'ROI':>10}")


def _eval_all_types(records, label, strategy, **kwargs) -> None:
    """全体＋レースタイプ層別で1行ずつ表示する。"""
    print(_row(label, sel.run_strategy(records, strategy, **kwargs)))
    for rt in RACE_TYPES:
        res = sel.run_strategy(records, strategy, race_type=rt, **kwargs)
        if res["n_bets"] == 0:
            continue
        print(_row(f"   └ {rt}", res))


# --------------------------------------------------------------------------
# 同確率帯 × 高オッズ紐 の効果分析
# --------------------------------------------------------------------------
def equal_prob_himo_analysis(records, n_pbins: int = 5) -> None:
    """頭=top1固定の紐（combo[0]=最有力車）を対象に、model_prob を等件数ビンに分け、
    各ビン内をオッズ中央値で low/high に割って実現ROIを比較する。

    「確率が近い（同じprobビン内）なら高オッズ紐の方がEVが高い」というユーザー仮説を、
    実現ROI（favorite-longshot bias 込み）で定量化する。
    """
    # 各レースで頭=1着確率最大の車を決め、その車が頭の買い目だけ集める
    himo: list = []
    for recs in sel.group_by_race(records).values():
        fp = sel._first_place_probs(recs)
        if not fp:
            continue
        head = max(fp, key=fp.get)
        himo.extend(r for r in recs if r.combo[0] == head)
    if not himo:
        print("  紐候補なし")
        return
    himo.sort(key=lambda r: r.model_prob)
    n = len(himo)
    print(f"\n=== 同確率帯 × 高オッズ紐 の効果（頭=top1固定・紐 {n} 点を確率{n_pbins}分位） ===")
    print("  各probビン内をオッズ中央値でlow/highに二分し、実現ROIを比較（highが高ければ仮説を支持）")
    print(f"{'probビン':<10}{'prob範囲':>16}{'側':>6}{'点数':>7}{'的中':>6}"
          f"{'平均odds':>10}{'ROI':>10}")
    for i in range(n_pbins):
        lo, hi = i * n // n_pbins, (i + 1) * n // n_pbins
        chunk = himo[lo:hi]
        if not chunk:
            continue
        p_lo, p_hi = chunk[0].model_prob, chunk[-1].model_prob
        odds_sorted = sorted(r.odds for r in chunk)
        med = odds_sorted[len(odds_sorted) // 2]
        low = [r for r in chunk if r.odds <= med]
        high = [r for r in chunk if r.odds > med]
        for side_name, side in (("low", low), ("high", high)):
            if not side:
                continue
            stake = len(side) * 100
            ret = sum(r.payout for r in side if r.is_win)
            hits = sum(1 for r in side if r.is_win)
            avg_odds = sum(r.odds for r in side) / len(side)
            roi = ret / stake if stake else None
            label = f"P{i+1}" if side_name == "low" else ""
            rng = f"{p_lo:.4f}-{p_hi:.4f}" if side_name == "low" else ""
            print(f"{label:<10}{rng:>16}{side_name:>6}{len(side):>7}{hits:>6}"
                  f"{avg_odds:>10.1f}{_fmt_roi(roi):>10}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="三連単 紐選定戦略の比較評価")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--haircut", type=float, default=1.0)
    ap.add_argument("--full", action="store_true",
                    help="拡張特徴量(PL_FEATURES_FULL・recent_form必須)で学習")
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args()

    feats = PL_FEATURES_FULL if args.full else PL_FEATURES
    samples = load_samples(args.db, features=feats)
    train, test = time_split(samples, args.test_frac)
    if not train or not test:
        print("サンプル不足。DB/特徴量を確認してください。")
        return
    print(f"特徴量: {'FULL(' + str(len(feats)) + ')' if args.full else 'BASE(' + str(len(feats)) + ')'}")
    print(f"サンプル {len(samples)}（train {len(train)} / test {len(test)}）")
    print(f"検証期間: {test[0].date} 〜 {test[-1].date}")

    model = train_pl(train, l2=args.l2)
    test_ids = [s.race_id for s in test]
    records = build_records(args.db, model, test_ids, haircut=args.haircut)
    n_races = len({r.race_id for r in records})
    print(f"検証買い目レコード: {len(records)}（{n_races}レース × 各210点）")
    print("\n凡例: 点/R=1レース平均点数, 的中率=的中レース/買ったレース, ROI=回収/投票")

    # --- 1. ベースライン: EV≥thr 全買い ---
    _header("【1】ベースライン select_all_ev（EV≥thr を全210点から全買い）")
    for thr in (1.05, 1.10, 1.20, 1.30):
        _eval_all_types(records, f"all_ev thr={thr}", sel.select_all_ev, thr=thr)

    # --- 2. 頭固定フォーメーション: EV順 vs 確率順（ユーザー仮説の中核） ---
    _header("【2】頭固定フォーメーション select_head_fixed（頭=top1・2着×3着=n×n）")
    for n in (3, 4, 5):
        for order in ("prob", "ev"):
            _eval_all_types(records, f"head_fixed {n}x{n} order={order}",
                            sel.select_head_fixed, n_second=n, n_third=n, order=order)
        print()

    # --- 3. 頭ボックス（上位2車を頭に） ---
    _header("【3】頭ボックス select_head_box（上位2車を頭・2着×3着=4×4）")
    for order in ("prob", "ev"):
        _eval_all_types(records, f"head_box heads=2 order={order}",
                        sel.select_head_box, heads=2, n_second=4, n_third=4, order=order)

    # --- 4. 高オッズ紐のEV条件付き取り込み ---
    _header("【4】高オッズ紐取り込み select_longshot_himo（prob順4x4 + odds≥50 & EV≥thr）")
    _eval_all_types(records, "base(prob 4x4)", sel.select_head_fixed,
                    n_second=4, n_third=4, order="prob")
    for ev_thr in (1.20, 1.50):
        _eval_all_types(records, f"+longshot EV≥{ev_thr}", sel.select_longshot_himo,
                        n_second=4, n_third=4, base_order="prob",
                        ev_thr=ev_thr, min_odds=50.0)

    # --- 5. 同確率帯 × 高オッズ紐 ---
    equal_prob_himo_analysis(records)

    # --- 結論の材料（EV順 vs 確率順を4x4で直接対比） ---
    print("\n=== 結論の材料: 頭固定4x4 EV順 vs 確率順（全体） ===")
    r_prob = sel.run_strategy(records, sel.select_head_fixed,
                              n_second=4, n_third=4, order="prob")
    r_ev = sel.run_strategy(records, sel.select_head_fixed,
                            n_second=4, n_third=4, order="ev")
    print(_row("order=prob", r_prob))
    print(_row("order=ev  ", r_ev))
    if r_prob["roi"] is not None and r_ev["roi"] is not None:
        diff = (r_ev["roi"] - r_prob["roi"]) * 100
        print(f"→ ROI差(ev - prob) = {diff:+.1f}pt（正なら高EV紐選定が有利＝ユーザー仮説を支持）")


if __name__ == "__main__":
    main()
