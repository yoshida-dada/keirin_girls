"""レース全体特徴量＋相対特徴量のテスト（実出走表フィクスチャ＋合成データ）。"""
from pathlib import Path

import pytest

from src.collect.gamboo_racecard import parse_race_card, Entry
from src.features.race_features import race_context, rider_relative

FX = Path(__file__).parent / "fixtures"


def _entries():
    return parse_race_card((FX / "gamboo_racecard_7car.html").read_text(encoding="utf-8"))


def test_race_context_basic():
    ctx = race_context(_entries())
    assert ctx.field_size == 7
    # 実測得点: 79.88,89.66,91.00,84.15,90.33,78.19,89.59 → top=91.00
    assert ctx.top_score == 91.00
    assert ctx.mean_score == pytest.approx(86.114, abs=0.01)
    assert ctx.std_score > 0
    assert ctx.top_gap == pytest.approx(91.00 - 90.33, abs=1e-6)   # トップ−2位


def test_relative_features_consistent():
    entries = _entries()
    rel = rider_relative(entries)
    ctx = race_context(entries)
    # 最高得点(3番=91.00)は rel_score_max=0 かつ score_rank=1
    top_car = max(rel, key=lambda c: -rel[c].score_rank)
    assert rel[3].rel_score_max == 0.0
    assert rel[3].score_rank == 1
    # rel_score_mean = 得点 − 平均
    e3 = next(e for e in entries if e.car_number == 3)
    assert rel[3].rel_score_mean == pytest.approx(e3.racing_score - ctx.mean_score, abs=0.01)
    # 順位は1..7の全単射
    assert sorted(r.score_rank for r in rel.values()) == [1, 2, 3, 4, 5, 6, 7]


def test_leg_composition_and_escape_count():
    # 逃3・捲1・差2・マ1 の構成を合成
    entries = [
        _mk(1, 55, "逃"), _mk(2, 54, "逃"), _mk(3, 53, "逃"),
        _mk(4, 52, "捲"), _mk(5, 51, "差"), _mk(6, 50, "差"), _mk(7, 49, "マーク"),
    ]
    ctx = race_context(entries)
    assert ctx.escape_count == 3
    assert ctx.dash_count == 1
    assert ctx.leg_composition["逃"] == 3


def test_std_reflects_spread():
    tight = [_mk(i, 55.0, "逃") for i in range(1, 8)]        # 全員同点＝混戦
    spread = [_mk(i, 40.0 + i * 5, "逃") for i in range(1, 8)]  # ばらつき大
    assert race_context(tight).std_score == 0.0
    assert race_context(spread).std_score > race_context(tight).std_score


def test_too_few_scores_returns_none():
    assert race_context([_mk(1, 55, "逃")]) is None
    assert rider_relative([]) == {}


def _mk(car, score, leg):
    return Entry(car_number=car, bracket_number=car, rider_name=f"r{car}", prefecture="東京",
                 age=25, term=120, class_rank="L1", leg_type=leg, gear_ratio=3.9,
                 racing_score=score)
