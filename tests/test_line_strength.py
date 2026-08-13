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


# ---- 番手の判定（記者の並び予想のラインを正とする） ----

from src.model.himo_adjust import marker_of, corrected_trifecta_probs


def test_marker_uses_line_not_flat_order():
    """男子: 番手は同一ライン内の直後。隊列の直後（＝次ラインの先頭）を拾わない。

    表示例 5-2-1 / 6 / 3-7-4 のとき、隊列は 5,2,1,6,3,7,4。
    1番はラインの最後尾なので番手はいない。隊列直後の6番は別ラインの単騎＝敵。
    """
    lines = [[5, 2, 1], [6], [3, 7, 4]]
    npos = {c: i for i, c in enumerate([5, 2, 1, 6, 3, 7, 4])}
    assert marker_of(5, npos, lines) == 2          # ライン先頭の番手
    assert marker_of(2, npos, lines) == 1          # 番手の後ろは3番手
    assert marker_of(1, npos, lines) is None       # 最後尾＝番手なし（6番を拾わない）
    assert marker_of(6, npos, lines) is None       # 単騎＝番手なし（3番を拾わない）
    assert marker_of(3, npos, lines) == 7
    assert marker_of(4, npos, lines) is None
    # ライン情報が無ければ従来どおり隊列の直後（ガールズ）
    assert marker_of(1, npos, None) == 6


def test_marker_absent_car_returns_none():
    assert marker_of(9, {1: 0}, [[1, 2]]) is None
    assert marker_of(None, {1: 0}, [[1, 2]]) is None


def test_correction_targets_the_line_mate():
    """加点が同一ラインの番手に向き、次ラインの先頭には向かないこと。"""
    lines = [[1], [2, 3]]
    st = {1: 0.5, 2: 0.3, 3: 0.2}
    npos = {1: 0, 2: 1, 3: 2}
    params = {"t2": 1.0, "t3": 1.0, "mark": 1.0}
    # ◎=1 は単騎。ライン基準なら番手なし＝補正は掛からず素のPLと一致する
    with_lines = corrected_trifecta_probs(st, npos, params, lines=lines)
    plain = corrected_trifecta_probs(st, npos, {"t2": 1.0, "t3": 1.0, "mark": 0.0})
    for k in plain:
        assert with_lines[k] == pytest.approx(plain[k], abs=1e-9)
    # ライン情報が無いと隊列直後の2番へ加点され、素のPLから乖離する（＝従来の誤り方）
    flat = corrected_trifecta_probs(st, npos, params)
    assert flat[(1, 2, 3)] > plain[(1, 2, 3)]
