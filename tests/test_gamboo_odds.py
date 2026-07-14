"""GambooBET三連単オッズパーサのテスト（保存フィクスチャでオフライン検証）。"""
from pathlib import Path

import pytest

from src.collect.gamboo_odds import parse_trifecta_odds, build_odds_url
from src.collect.base import detect_missing_trifecta

FIXTURE = Path(__file__).parent / "fixtures" / "gamboo_trifecta_sample.html"


@pytest.fixture(scope="module")
def odds():
    return parse_trifecta_odds(FIXTURE.read_text(encoding="utf-8"))


def test_field_size_and_completeness(odds):
    cars = {c for combo in odds for c in combo}
    n = len(cars)
    assert n >= 7                                   # 実サンプルは9車
    expected = n * (n - 1) * (n - 2)
    assert len(odds) == expected                    # 全点取得（的中以外も）
    assert detect_missing_trifecta(odds, n) == []   # 欠損なし


def test_known_odds_values(odds):
    # フィクスチャ実測値（1着1固定テーブル）
    assert odds[(1, 2, 3)] == 67.4
    assert odds[(1, 3, 2)] == 70.2
    assert odds[(1, 4, 2)] == 187.5


def test_no_self_pairs(odds):
    for a, b, c in odds:
        assert len({a, b, c}) == 3                  # 同一車番の組合せは無い


def test_all_odds_positive(odds):
    assert all(o > 0 for o in odds.values())


def test_url_builder():
    url = build_odds_url("3520251228", "35202512280100", 11)
    assert url.endswith("/3520251228/35202512280100/11/3rentan/")
