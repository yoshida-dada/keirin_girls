"""競輪ステーション 選手詳細パーサのテスト（保存フィクスチャでオフライン検証）。

fixtures/ks_player_detail_sample.html は登録番号013140（小倉竜二・S級男子）。
girls(L級)も同一DOM構造。実測値をハードコードで固定する。
"""
from pathlib import Path

import pytest

from src.collect.keirin_station import parse_player_detail, build_player_url

FX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def ps():
    html = (FX / "ks_player_detail_sample.html").read_text(encoding="utf-8")
    return parse_player_detail(html, rider_id="013140")


def test_profile(ps):
    assert ps.rider_id == "013140"
    assert ps.name == "小倉 竜二"
    assert ps.prefecture == "徳島県"                # 生値（県付き）
    from src.collect.keirin_station import normalize_prefecture
    assert normalize_prefecture(ps.prefecture) == "徳島"   # 突合用の正規化


def test_recent_4m_rates(ps):
    assert ps.starts == 30
    assert ps.win_rate == pytest.approx(0.133)     # 13.30%
    assert ps.top2_rate == pytest.approx(0.333)
    assert ps.top3_rate == pytest.approx(0.433)


def test_recent_4m_score(ps):
    assert ps.score_avg == pytest.approx(110.57)
    assert ps.score_max == pytest.approx(111.21)
    assert ps.score_min == pytest.approx(109.96)


def test_finish_counts(ps):
    assert (ps.first, ps.second, ps.third) == (4, 6, 3)
    assert ps.out == 13
    assert ps.dnf == 3 and ps.dsq == 1


def test_kimarite_rates(ps):
    assert ps.escape_rate == pytest.approx(0.0)     # 逃げ 0%
    assert ps.dash_rate == pytest.approx(0.0)       # 捲り 0%
    assert ps.closing_rate == pytest.approx(0.5)    # 差し 50%
    assert ps.mark_rate == pytest.approx(0.5)       # マーク 50%


def test_career_totals(ps):
    assert ps.career["出走数"]["通算"] == 2599
    assert ps.career["優勝"]["通算"] == 54
    assert ps.career["1着"]["本年"] == 8


def test_url_builder():
    assert build_player_url("013140").endswith("/player/detail/013140/")
