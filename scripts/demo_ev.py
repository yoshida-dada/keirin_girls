"""S4デモ: ダミーの選手強さと市場オッズから三連単210通りのEVを算出し、購入対象を表示する。

実データが揃う前にEVエンジン（PL確率→控除率除去→logit blend→EV→Kelly）を体感するための
スクリプト。実際のオッズ/確率はS1・S3が供給する。

  python scripts/demo_ev.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.plackett_luce import all_trifecta_probs, normalize_strengths
from src.ev.market import implied_trifecta_probs, blend_loglinear
from src.ev.ev_engine import build_trifecta_ev_table, format_combo
from src.ev import staking

TAKEOUT = 0.25


def main() -> None:
    # --- ダミー入力（本番では S3=モデル確率, S1=実オッズ が供給） ---
    strengths = normalize_strengths({1: 3.0, 2: 2.4, 3: 2.0, 4: 1.6, 5: 1.3, 6: 1.0, 7: 0.8})
    model = all_trifecta_probs(strengths)

    # 市場は平坦化（favorite-longshot bias を模擬）→ 本命筋に +EV が出る
    market_dist = {c: p ** 0.6 for c, p in model.items()}
    z = sum(market_dist.values())
    market_dist = {c: v / z for c, v in market_dist.items()}
    odds = {c: round((1 - TAKEOUT) / q, 1) for c, q in market_dist.items()}

    # --- パイプライン ---
    implied = implied_trifecta_probs(odds)                 # 控除率25%を除去
    blended = blend_loglinear(model, implied, alpha=0.8)   # モデル重み0.8でlogit blend
    # 市場縮小はブレンドで済ませたのでガードの shrink_to_market は無効化（二重適用を避ける）。
    # min_prob / max_odds の尾部ガードのみ効かせる。
    guards = {"shrink_to_market": 0.0, "min_prob": 0.005, "max_odds": 500.0}
    table = build_trifecta_ev_table(blended, odds, ev_threshold=1.10, guards=guards)

    print(f"候補{len(table['all'])}点 / 購入対象{len(table['buy'])}点（EV閾値1.10, quarter Kelly, 資金10万円）\n")
    print(f"{'買い目':<10}{'モデル%':>8}{'市場%':>8}{'オッズ':>9}{'EV':>7}{'帯':>9}{'賭金(理論)':>13}")
    print("-" * 66)
    bankroll = 100_000
    for r in table["buy"][:15]:
        theo = staking.kelly_stake_yen(bankroll, r.prob, r.odds)
        stake = staking.round_to_unit(theo)
        print(f"{format_combo(r.combo):<10}{r.model_prob*100:>7.2f}{r.market_prob*100:>8.2f}"
              f"{r.odds:>9.1f}{r.ev_gross:>7.2f}{r.odds_bucket:>9}{stake:>7}({theo:>4.0f})")
    print("  ※賭金0=quarter Kellyの理論額が100円未満（低エッジ×高オッズは資金比が小さい）")

    print("\nオッズ帯別サマリ:")
    for bucket, info in sorted(table["by_bucket"].items()):
        print(f"  {bucket:>9}: 候補{info['n_candidates']:>3}  購入{info['n_buy']:>3}  最大EV {info['max_ev']}")


if __name__ == "__main__":
    main()
