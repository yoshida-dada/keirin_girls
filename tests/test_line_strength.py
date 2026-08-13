"""男子のライン別強さ（表示用）のテスト。

数値がモデル分布と整合していること＝表示のライン評価と買い目確率が食い違わないことを見る。
"""
import pytest

from src.model.line_strength import build_lines
from src.model.plackett_luce import all_trifecta_probs


@pytest.fixture(scope="module")
def race():
    """3-2-2 の三分戦。強さは 1>2>… の順に置く。"""
    lines = [[1, 2, 3], [4, 5], [6, 7]]
    strengths = {1: 0.30, 2: 0.15, 3: 0.05, 4: 0.20, 5: 0.10, 6: 0.14, 7: 0.06}
    probs = all_trifecta_probs(strengths)
    return lines, strengths, probs


def test_girls_returns_none():
    """ガールズは lines が空。ラインの概念が無いので None（表示しない）。"""
    assert build_lines([], {1: 1.0}, None) is None


def test_p_win_is_sum_of_members(race):
    lines, strengths, probs = race
    got = build_lines(lines, strengths, probs)
    assert [round(x["p_win"], 4) for x in got["lines"]] == [0.50, 0.30, 0.20]
    # 1着はどこかのラインから必ず出る＝合計は1
    assert sum(x["p_win"] for x in got["lines"]) == pytest.approx(1.0, abs=1e-6)


def test_rank_is_by_win_prob(race):
    lines, strengths, probs = race
    got = build_lines(lines, strengths, probs)
    assert [x["rank"] for x in got["lines"]] == [1, 2, 3]


def test_settle_prob_matches_direct_count(race):
    """ライン決着確率が三連単分布からの直接集計と一致する（別物差しを使っていない）。"""
    lines, strengths, probs = race
    got = build_lines(lines, strengths, probs)
    line_of = {c: i for i, mem in enumerate(lines) for c in mem}
    direct = sum(p for (a, b, _), p in probs.items() if line_of[a] == line_of[b])
    assert got["settle_prob"] == pytest.approx(direct, abs=1e-4)
    assert 0.0 < got["settle_prob"] < 1.0


def test_top3_any_bounded_and_ordered(race):
    lines, strengths, probs = race
    got = build_lines(lines, strengths, probs)
    for x in got["lines"]:
        assert 0.0 <= x["p_top3_any"] <= 1.0
        assert x["p_top3_any"] >= x["p_win"]      # 1着はいずれ3着以内に含まれる
    # 3名ラインは誰かが3着内に入りやすい
    assert got["lines"][0]["p_top3_any"] > got["lines"][2]["p_top3_any"]


def test_solo_labeled_as_tanki():
    """単騎は「先頭」ではなく単騎と出す（1人ラインを先頭と書くと誤読される）。"""
    got = build_lines([[1, 2], [3]], {1: 0.5, 2: 0.3, 3: 0.2}, None)
    assert got["lines"][1]["cars"][0]["pos_label"] == "単騎"
    assert [c["pos_label"] for c in got["lines"][0]["cars"]] == ["先頭", "番手"]


def test_without_probs_line_metrics_are_none():
    """三連単確率が無ければ p_12/p_top3_any は None。0で埋めない（無いことを隠さない）。"""
    got = build_lines([[1, 2]], {1: 0.6, 2: 0.4}, None)
    assert got["lines"][0]["p_12"] is None
    assert got["lines"][0]["p_top3_any"] is None
    assert got["settle_prob"] is None
    assert got["lines"][0]["p_win"] == 1.0


def test_carries_display_fields():
    got = build_lines([[1, 2]], {1: 0.6, 2: 0.4}, None,
                      names={1: "山田", 2: "鈴木"}, scores={1: 100.0, 2: 90.0},
                      legs={1: "先行", 2: "追込"}, classes={1: "S1", 2: "S2"})
    c0 = got["lines"][0]["cars"][0]
    assert (c0["name"], c0["leg"], c0["class_rank"]) == ("山田", "先行", "S1")
    assert got["lines"][0]["score_avg"] == 95.0
