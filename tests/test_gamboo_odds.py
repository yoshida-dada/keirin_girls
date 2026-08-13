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


# ---- 開催格・レース名（オッズページから追加フェッチ0で取る） ----

RACECARD_FIXTURE = Path(__file__).parent / "fixtures" / "gamboo_racecard_7car.html"


def test_race_meta_grade_from_fullwidth_text():
    """h1の全角表記（Ｆ１）を第一情報源にする。GPは実フィクスチャで確認済み。"""
    from src.collect.gamboo_odds import parse_race_meta
    m = parse_race_meta(RACECARD_FIXTURE.read_text(encoding="utf-8"))
    assert m["grade"] == "F1"
    assert m["venue"] == "函館競輪"
    assert m["race_name"] == "Ａ級予選"          # 生文字列（正規化は表示側）
    gp = parse_race_meta(FIXTURE.read_text(encoding="utf-8"))
    assert gp["grade"] == "GP"                   # 競輪グランプリ
    assert gp["race_name"] == "ヤンググランプリ"


def test_race_meta_falls_back_to_icon_class():
    """全角表記が無いページでも icon_grade クラスから拾う（grN→格）。"""
    from src.collect.gamboo_odds import parse_race_meta
    html = '<h2 class="title"><span class="icon_grade gr5"></span></h2>'
    assert parse_race_meta(html)["grade"] == "G1"


def test_race_meta_missing_is_none():
    """要素が無ければNone。取れないことを空文字で誤魔化さない。"""
    from src.collect.gamboo_odds import parse_race_meta
    m = parse_race_meta("<html><body></body></html>")
    assert m == {"grade": None, "venue": None, "meet_name": None, "race_name": None}
