"""展開分岐（主導権で場合分けした着順分布と買い目）のテスト。"""
import pytest

from src.model.development_branches import (role_of, branch_trifecta, _formation,
                                            build_branches, load_stats)

LINES = [[5, 2, 1], [6], [3, 7, 4]]
ST = {5: 0.40, 2: 0.15, 1: 0.05, 6: 0.05, 3: 0.20, 7: 0.10, 4: 0.05}


def test_role_is_relative_to_the_b_taker():
    """役割は主導権者から見た相対位置。同じ車でもBが変われば役割が変わる。"""
    assert role_of(5, 5, LINES) == "B本人"
    assert role_of(2, 5, LINES) == "B番手"
    assert role_of(1, 5, LINES) == "B同ライン他"
    assert role_of(3, 5, LINES) == "他ライン先頭"
    assert role_of(7, 5, LINES) == "他ライン番手"
    assert role_of(4, 5, LINES) == "他ライン3番手+"
    assert role_of(6, 5, LINES) == "単騎"
    # Bが3に変われば、5の側が「他ライン先頭」になる
    assert role_of(5, 3, LINES) == "他ライン先頭"
    assert role_of(2, 3, LINES) == "他ライン番手"


def test_branch_distribution_is_a_probability():
    d = branch_trifecta(ST, 5, LINES)
    assert d
    assert sum(d.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(p > 0 for p in d.values())


def test_branch_lifts_the_b_line():
    """主導権ラインを条件にすると、そのラインの1・2着が素のPLより上がる。

    実測倍率は B番手の1着1.53倍・他ライン番手0.51倍。ライン結束が反映されること。
    """
    from src.model.plackett_luce import all_trifecta_probs
    raw = all_trifecta_probs(ST)
    d5 = branch_trifecta(ST, 5, LINES)
    settle = lambda dd, mem: sum(p for (a, b, _), p in dd.items()
                                 if a in mem and b in mem)
    assert settle(d5, {5, 2, 1}) > settle(raw, {5, 2, 1})


def test_formation_does_not_waste_a_slot_on_the_fixed_first():
    """1着を1頭に固定したとき、その車を2着欄に入れて枠を無駄にしない。"""
    f = _formation(branch_trifecta(ST, 5, LINES), n1=1, n2=3, n3=5)
    assert len(f["first"]) == 1
    assert f["first"][0] not in f["second"]
    assert f["points"] >= 3 * 3          # 1×3×(5-重複) で概ね9点以上
    assert 0.0 < f["cover"] <= 1.0


def test_formation_text_is_compact_notation():
    f = _formation(branch_trifecta(ST, 5, LINES), n1=1, n2=2, n3=3)
    a, b, c = f["text"].split("-")
    assert a == "".join(str(x) for x in sorted(f["first"]))
    assert len(b) == len(f["second"]) and len(c) == len(f["third"])


def test_girls_returns_none():
    """ラインが無い（ガールズ）なら分岐は作らない。"""
    assert build_branches(ST, [], {5: 1.0}) is None
    assert build_branches(ST, LINES, None) is None


def test_branches_sorted_and_thresholded():
    got = build_branches(ST, LINES, {5: 0.70, 3: 0.25, 6: 0.02}, top_k=3, min_prob=0.05)
    assert got is not None
    probs = [b["prob"] for b in got["branches"]]
    assert probs == sorted(probs, reverse=True)
    assert all(p >= 0.05 for p in probs)      # 0.02 の 6番は落ちる
    assert got["covered"] == pytest.approx(0.95, abs=1e-6)
    labels = [b["label"] for b in got["branches"]]
    assert "5番ラインが主導権" in labels[0]


def test_solo_branch_labeled_as_tanki():
    got = build_branches(ST, LINES, {6: 0.9}, top_k=1)
    assert got["branches"][0]["is_solo"] is True
    assert "単騎" in got["branches"][0]["label"]


def test_stats_file_has_all_roles():
    st = load_stats()
    assert st.get("weights"), "branch_stats_men.json が無い"
    for k in ("1", "2", "3"):
        for r in ("B本人", "B番手", "他ライン番手"):
            assert r in st["weights"][k]
