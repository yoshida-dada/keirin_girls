"""派生特徴量のテスト（実出走表フィクスチャ＋合成データ）。"""
from pathlib import Path

import pytest

from src.collect.gamboo_racecard import parse_race_card, parse_recent_form, Entry, RecentForm
from src.features.derived_features import derived_features

FX = Path(__file__).parent / "fixtures"


def _real():
    html = (FX / "gamboo_racecard_7car.html").read_text(encoding="utf-8")
    return parse_race_card(html), parse_recent_form(html)


def test_derived_on_real_card():
    entries, recent = _real()
    d = derived_features(entries, recent)
    assert set(d.keys()) == {1, 2, 3, 4, 5, 6, 7}
    for car, f in d.items():
        assert f.ability_gap is not None
        if f.recent_avg_finish is not None:
            assert 1.0 <= f.recent_avg_finish <= 7.0     # 7車なら着順代表値の範囲内
        if f.escape_ev is not None:
            assert f.escape_ev >= 0


def test_escape_ev_scales_with_escape_count():
    # 逃げ率100%の選手。逃げ人数が多いほど逃げ期待値が上がる。
    entries_few = [_mk(1, 55, "逃"), _mk(2, 54, "差"), _mk(3, 53, "差"),
                   _mk(4, 52, "差"), _mk(5, 51, "差"), _mk(6, 50, "差"), _mk(7, 49, "差")]
    entries_many = [_mk(1, 55, "逃"), _mk(2, 54, "逃"), _mk(3, 53, "逃"),
                    _mk(4, 52, "逃"), _mk(5, 51, "差"), _mk(6, 50, "差"), _mk(7, 49, "差")]
    recent = {1: _rf(1, escape=10, first=10, out=0)}   # starts=10, 逃げ率=1.0
    d_few = derived_features(entries_few, recent)[1]
    d_many = derived_features(entries_many, recent)[1]
    assert d_few.escape_ev == pytest.approx(1.0 * 1)     # 逃げ人数1
    assert d_many.escape_ev == pytest.approx(1.0 * 4)    # 逃げ人数4


def test_stability_lower_for_consistent():
    entries = [_mk(i, 55, "逃") for i in range(1, 8)]
    steady = {1: _rf(1, first=0, second=10, third=0, out=0)}   # 常に2着＝安定
    erratic = {2: _rf(2, first=5, second=0, third=0, out=5)}   # 1着と着外が半々
    d_steady = derived_features(entries, steady)[1]
    d_erratic = derived_features(entries, erratic)[2]
    assert d_steady.stability == 0.0
    assert d_erratic.stability > d_steady.stability


def test_no_recent_form_graceful():
    entries = [_mk(i, 55, "逃") for i in range(1, 8)]
    d = derived_features(entries, {})           # 直近成績なし
    assert d[1].ability_gap is not None          # 得点由来は出る
    assert d[1].escape_ev is None                # 直近由来はNone


def _mk(car, score, leg):
    return Entry(car_number=car, bracket_number=car, rider_name=f"r{car}", prefecture="東京",
                 age=25, term=120, class_rank="L1", leg_type=leg, gear_ratio=3.9,
                 racing_score=score)


def _rf(car, escape=0, first=0, second=0, third=0, out=0):
    return RecentForm(car_number=car, escape=escape, first=first, second=second,
                      third=third, out=out)
