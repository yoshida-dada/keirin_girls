"""出走表パーサのテスト。構造は実7車フィクスチャ、ガールズ判定は合成データで検証。

fixtures/gamboo_racecard_7car.html は 2026-07-14 会場11 R1（7車のA級男子戦）。ガールズと車立て
構造は同一（級班と脚質の値のみ異なる）。実L級レースのフィクスチャは取得でき次第追加する。
"""
from pathlib import Path

import pytest

from src.collect.gamboo_racecard import parse_race_card, is_girls_race, Entry

FX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def entries():
    return parse_race_card((FX / "gamboo_racecard_7car.html").read_text(encoding="utf-8"))


def test_seven_entries_sorted(entries):
    assert len(entries) == 7
    assert [e.car_number for e in entries] == [1, 2, 3, 4, 5, 6, 7]


def test_first_rider_fields(entries):
    e = entries[0]
    assert e.car_number == 1
    assert e.rider_name == "竹内 真一"
    assert e.prefecture == "福岡"          # 全角スペース除去
    assert e.age == 45
    assert e.term == 89
    assert e.class_rank == "A2"
    assert e.gear_ratio == 3.92
    assert e.racing_score == 79.88          # 競走得点


def test_scores_present_for_all(entries):
    assert all(e.racing_score and e.racing_score > 0 for e in entries)
    # 実測: 5番 長谷川 の競走得点
    e5 = next(e for e in entries if e.car_number == 5)
    assert e5.racing_score == 90.33
    assert e5.rider_name == "長谷川 飛向"


def test_is_girls_race_detection():
    assert not is_girls_race(_mk(["A1", "A2", "A1"]))    # 男子戦
    assert is_girls_race(_mk(["L1", "L1", "L2"]))        # ガールズ
    assert not is_girls_race(_mk(["L1", "A1"]))          # 混在は非ガールズ扱い
    assert not is_girls_race([])                          # 空


def test_leg_type_normalized():
    # マ → マーク（他はそのまま）
    e = _mk_one(leg="マ")
    assert e.leg_type == "マーク"


def _mk(ranks):
    return [_mk_one(rank=r) for r in ranks]


def _mk_one(rank="L1", leg="逃"):
    from src.collect.gamboo_racecard import _LEG_NORMALIZE
    return Entry(car_number=1, bracket_number=1, rider_name="x", prefecture="東京",
                 age=25, term=120, class_rank=rank,
                 leg_type=_LEG_NORMALIZE.get(leg, leg),
                 gear_ratio=3.92, racing_score=55.0)
