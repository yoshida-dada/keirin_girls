"""GambooBET 結果パーサのテスト（保存フィクスチャでオフライン検証）。

fixtures/gamboo_result_sample.html は平塚 2025-12-28 R11（9車）。実測値を固定する。
"""
from pathlib import Path

import pytest

from src.collect.gamboo_result import (
    parse_results, parse_trifecta_payout, build_result_url,
)

FX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def html():
    return (FX / "gamboo_result_sample.html").read_text(encoding="utf-8")


def test_results_order_and_fields(html):
    rows = parse_results(html)
    assert len(rows) == 9
    assert [r.position for r in rows] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    winner = rows[0]
    assert winner.car_number == 3
    assert winner.rider_name == "中石 湊"
    assert winner.last_lap == 11.0            # 上がりタイム
    assert winner.kimarite == "捲"
    # 2着は差し・上り11.1
    assert rows[1].car_number == 9 and rows[1].last_lap == 11.1


def test_last_lap_present_for_all(html):
    rows = parse_results(html)
    assert all(r.last_lap and r.last_lap > 0 for r in rows)


def test_trifecta_payout(html):
    p = parse_trifecta_payout(html)
    assert p is not None
    assert p.combo == (3, 9, 6)               # 着順 1着3・2着9・3着6 と整合
    assert p.payout == 74450                   # 74,450円
    assert p.popularity == 269


def test_payout_matches_result_order(html):
    rows = parse_results(html)
    p = parse_trifecta_payout(html)
    top3 = tuple(r.car_number for r in rows[:3])
    assert p.combo == top3                     # 払戻の的中組合せ＝着順上位3


def test_result_url():
    url = build_result_url("3520251228", "35202512280100", 11)
    assert url.endswith("/race-card/result/3520251228/35202512280100/11/")
