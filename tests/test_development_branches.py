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


def test_formation_has_no_dead_slot():
    """どの枠のどの車も、少なくとも1点の有効な買い目に寄与すること。

    枠に入れても1点も生まないなら選び方が間違っている（以前、1着に固定した車を
    2着欄にも入れて3点しか買えないフォーメーションが出た）。
    複数頭を1着に置く場合、2着欄との重複自体は正常（同一車の組は除外されるだけ）。
    """
    f = _formation(branch_trifecta(ST, 5, LINES), budget=18)
    combos = [(a, b, c) for a in f["first"] for b in f["second"] for c in f["third"]
              if len({a, b, c}) == 3]
    for slot, idx in (("first", 0), ("second", 1), ("third", 2)):
        for car in f[slot]:
            assert any(cb[idx] == car for cb in combos), f"{slot} の {car} が死に枠"
    assert 0.0 < f["cover"] <= 1.0
    assert f["cover_cond"] is not None and f["cover"] <= f["cover_cond"] + 1e-9


def test_formation_respects_the_points_budget():
    """点数は予算以内。予算を増やせばカバーは減らない（貪欲に広げるので単調）。"""
    d = branch_trifecta(ST, 5, LINES)
    small = _formation(d, budget=8)
    big = _formation(d, budget=24)
    assert small["points"] <= 8 and big["points"] <= 24
    assert big["cover"] >= small["cover"]


def test_formation_adapts_points_to_the_distribution():
    """尖った分岐と割れた分岐で点数/カバーが変わる（固定点数ではない）。"""
    sharp = {1: 0.80, 2: 0.10, 3: 0.04, 4: 0.03, 5: 0.02, 6: 0.005, 7: 0.005}
    flat = {c: 1 / 7 for c in range(1, 8)}
    a = _formation(branch_trifecta(sharp, 5, LINES), budget=18)
    b = _formation(branch_trifecta(flat, 5, LINES), budget=18)
    assert a["cover"] != b["cover"]


def test_formation_text_is_compact_notation():
    f = _formation(branch_trifecta(ST, 5, LINES), budget=12)
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


# ---- 「Aが勝つときの2着・3着」（条件付き着順） ----

from src.model.development_branches import conditional_orders


def test_conditional_is_a_proper_conditional():
    """2着分布は P(2着|1着=A) で、合計1になる（同時分布の周辺化そのもの）。"""
    d = branch_trifecta(ST, 5, LINES)
    got = conditional_orders(d, LINES, top_win=3, top_n=99)
    assert got
    for g in got:
        # top_n=99 で全員拾っているので合計はほぼ1（probは小数4桁に丸めてあるので誤差を許容）
        assert sum(x["prob"] for x in g["second"]) == pytest.approx(1.0, abs=1e-3)
        assert sum(x["prob"] for x in g["third"]) == pytest.approx(1.0, abs=1e-3)


def test_conditional_matches_manual_marginalization():
    """手で周辺化した値と一致すること（別経路で計算していない）。"""
    d = branch_trifecta(ST, 5, LINES)
    g = conditional_orders(d, LINES, top_win=1, top_n=99)[0]
    w = g["car"]
    manual = {}
    for (a, b, _c), p in d.items():
        if a == w:
            manual[b] = manual.get(b, 0.0) + p
    z = sum(manual.values())
    for x in g["second"]:
        assert x["prob"] == pytest.approx(manual[x["car"]] / z, abs=1e-4)


def test_conditional_roles_are_relative_to_the_winner():
    """役割は勝者基準。主導権者基準ではない（買い目は勝者から読むため）。"""
    d = branch_trifecta(ST, 5, LINES)
    g = next(x for x in conditional_orders(d, LINES, top_win=3, top_n=99) if x["car"] == 5)
    role = {x["car"]: x["role"] for x in g["second"]}
    assert role[2] == "番手"          # 5の直後
    assert role[1] == "同ライン"      # 5と同じラインだが直後ではない
    assert role[3] == "別線先頭"
    assert role[7] == "別線番手"
    assert role[6] == "単騎"


def test_second_is_more_inline_than_third():
    """実測の非対称性（2着はライン内56%・3着は他ライン中心）が分布にも出ること。"""
    d = branch_trifecta(ST, 5, LINES)
    for g in conditional_orders(d, LINES, top_win=2):
        assert g["second_inline"] > g["third_inline"]


def test_branches_carry_conditional():
    got = build_branches(ST, LINES, {5: 0.7, 3: 0.3})
    assert got is not None
    for b in got["branches"]:
        assert b["conditional"], "分岐に conditional が無い"
        assert b["conditional"][0]["second"]
