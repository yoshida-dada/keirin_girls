"""競輪ステーション 選手検索＋名前突合のテスト（保存フィクスチャでオフライン）。

fixtures/ks_girls_search.html は girls_flag=1 の検索結果（現役ガールズ一覧の1ページ目）。
"""
from pathlib import Path

from src.collect.keirin_station import (
    parse_search_results, match_rider, build_search_url, RiderCandidate,
)

FX = Path(__file__).parent / "fixtures"


def _cands():
    return parse_search_results((FX / "ks_girls_search.html").read_text(encoding="utf-8"))


def test_search_results_parsed():
    cands = _cands()
    assert len(cands) >= 20                       # 1ページ目に多数
    aoki = next(c for c in cands if c.rider_id == "015485")
    assert aoki.name == "青木美保"                 # 空白除去済み
    assert aoki.prefecture == "埼玉"
    assert aoki.term == 118
    assert aoki.class_rank.startswith("Ｌ")        # ガールズ


def test_match_by_name_unique():
    cands = _cands()
    m = match_rider("青木 美保", "埼玉", 118, cands)
    assert m is not None and m.rider_id == "015485"


def test_match_disambiguates_by_pref_term():
    # 同姓同名を人工的に作り、府県＋期別で絞れることを確認
    cands = [
        RiderCandidate("000001", "山田花子", "東京", 110, "Ｌ級１班"),
        RiderCandidate("000002", "山田花子", "大阪", 115, "Ｌ級２班"),
    ]
    assert match_rider("山田 花子", "大阪", 115, cands).rider_id == "000002"
    # 府県も期別も一致しなければ確定できない → None
    assert match_rider("山田 花子", "福岡", 120, cands) is None


def test_no_match_returns_none():
    assert match_rider("存在 しない", "北海道", 100, _cands()) is None


def test_search_url_has_girls_flag():
    url = build_search_url("青木", "美保", girls=True)
    assert "girls_flag" in url and "name_1" in url and "name_2" in url
