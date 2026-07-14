"""S4 EVエンジンのエンドツーエンド検証（ダミー確率で210通りEV算出を通す）。"""
import math

import pytest

from src.model.plackett_luce import all_trifecta_probs, normalize_strengths, trifecta_prob
from src.ev.market import implied_trifecta_probs, blend_loglinear
from src.ev.ev_engine import build_trifecta_ev_table, odds_bucket_label
from src.ev import staking

TAKEOUT = 0.25


def _dummy_strengths():
    # 7車。車番1が最強、以降なだらかに弱くなる。
    raw = {1: 3.0, 2: 2.4, 3: 2.0, 4: 1.6, 5: 1.3, 6: 1.0, 7: 0.8}
    return normalize_strengths(raw)


def test_trifecta_probs_cover_210_and_sum_to_one():
    probs = all_trifecta_probs(_dummy_strengths())
    assert len(probs) == 210            # 7*6*5
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-9)
    assert all(p > 0 for p in probs.values())


def test_favorite_combo_more_likely_than_longshot():
    s = _dummy_strengths()
    assert trifecta_prob(s, 1, 2, 3) > trifecta_prob(s, 7, 6, 5)


def test_implied_probs_remove_takeout():
    # 真の市場確率 q から控除率25%込みのオッズを作る: odds = (1-takeout)/q
    s = _dummy_strengths()
    q_true = all_trifecta_probs(s)
    odds = {c: (1 - TAKEOUT) / q for c, q in q_true.items()}
    # プールの取り込み率 Σ(1/odds) = 1/(1-takeout) ≈ 1.333
    assert math.isclose(sum(1 / o for o in odds.values()), 1 / (1 - TAKEOUT), rel_tol=1e-9)
    implied = implied_trifecta_probs(odds)
    assert math.isclose(sum(implied.values()), 1.0, rel_tol=1e-9)
    # 控除率を除いた implied は元の q に一致
    for c in q_true:
        assert math.isclose(implied[c], q_true[c], rel_tol=1e-9)


def test_blend_endpoints():
    model = {("a",): 0.6, ("b",): 0.4}
    market = {("a",): 0.3, ("b",): 0.7}
    assert blend_loglinear(model, market, 1.0)[("a",)] == pytest.approx(0.6)   # α=1 → モデル
    assert blend_loglinear(model, market, 0.0)[("a",)] == pytest.approx(0.3)   # α=0 → 市場


def test_ev_table_flags_plus_ev_combo():
    # モデルは真の分布。市場は分布が平坦化している（favorite-longshot bias: 穴が過剰に売れ
    # 本命が過小評価される = 課題A）。平坦化で本命筋の市場確率 q < モデル確率 p となり、
    # 本命側に +EV が実在する。
    s = _dummy_strengths()
    model = all_trifecta_probs(s)
    beta = 0.6                                  # <1 で分布を平坦化
    market_dist = {c: p ** beta for c, p in model.items()}
    z = sum(market_dist.values())
    market_dist = {c: v / z for c, v in market_dist.items()}
    odds = {c: (1 - TAKEOUT) / q for c, q in market_dist.items()}

    table = build_trifecta_ev_table(
        model, odds, ev_threshold=1.10,
        guards={"shrink_to_market": 0.0, "min_prob": 0.0, "max_odds": None},
    )
    assert len(table["all"]) == 210
    assert len(table["buy"]) > 0
    assert all(r.ev_gross >= 1.10 for r in table["buy"])
    # 平坦化で過小評価される本命筋(1→2→3)が +EV 側に来る
    assert (1, 2, 3) in {r.combo for r in table["buy"]}


def test_odds_bucket_label():
    assert odds_bucket_label(5) == "0-10"
    assert odds_bucket_label(25) == "10-30"
    assert odds_bucket_label(1000) == "300+"


def test_kelly_and_hard_limits():
    # p*odds=1.2 の +EV。quarter Kelly の理論額 → 100円単位へ丸め
    stake = staking.kelly_stake_yen(bankroll_yen=100_000, prob=0.12, odds=10.0)
    assert stake > 0
    state = staking.BankrollState()
    capped, reason = staking.apply_hard_limits(stake, "R1", state,
                                               limits={"max_points_per_race": 30,
                                                       "max_stake_per_bet_yen": 3000,
                                                       "max_stake_per_race_yen": 3000,
                                                       "max_stake_per_day_yen": 30000,
                                                       "max_consecutive_losses": 15})
    assert capped % 100 == 0
    assert capped <= 3000


def test_no_ev_when_model_equals_market():
    # モデル=市場ならEV=1-takeout<1（控除率の分だけ必ず負け）→ 買い目ゼロ
    s = _dummy_strengths()
    model = all_trifecta_probs(s)
    odds = {c: (1 - TAKEOUT) / p for c, p in model.items()}
    table = build_trifecta_ev_table(model, odds, ev_threshold=1.0,
                                    guards={"shrink_to_market": 0.0, "min_prob": 0.0,
                                            "max_odds": None})
    assert len(table["buy"]) == 0
    for r in table["all"]:
        assert r.ev_gross == pytest.approx(1 - TAKEOUT, rel=1e-9)
