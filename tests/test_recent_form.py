"""出走表同梱の直近4ヶ月成績（as-osスタッツ）パーサのテスト。

fixtures/gamboo_racecard_7car.html の1番=竹内(実測: 1着1/2着0/3着5/着外21, 勝率3.7%, 3連対22.2%)。
"""
from pathlib import Path

import pytest

from src.collect.gamboo_racecard import parse_recent_form

FX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def form():
    return parse_recent_form((FX / "gamboo_racecard_7car.html").read_text(encoding="utf-8"))


def test_all_cars_present(form):
    assert set(form.keys()) == {1, 2, 3, 4, 5, 6, 7}


def test_car1_recent(form):
    f = form[1]
    assert (f.first, f.second, f.third, f.out) == (1, 0, 5, 21)
    assert f.starts == 27
    assert f.win_rate == pytest.approx(0.037)      # 3.7%
    assert f.top3_rate == pytest.approx(0.222)      # 22.2%
    assert f.s_count == 3 and f.b_count == 0


def test_rates_consistent(form):
    # 勝率 ≈ 1着/出走 が概ね整合（表示は小数第1位丸め）
    for f in form.values():
        if f.starts and f.starts > 0 and f.win_rate is not None:
            assert abs(f.win_rate - f.first / f.starts) < 0.02


def test_kimarite_counts_nonnegative(form):
    for f in form.values():
        for v in (f.escape, f.dash, f.closing, f.mark):
            assert v is None or v >= 0
