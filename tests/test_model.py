"""S3 確率モデルのテスト。実出走表フィクスチャ → 強さ → PL210通り → レースタイプまで通す。"""
import math
from pathlib import Path

import pytest

from src.collect.gamboo_racecard import parse_race_card, Entry
from src.model.strength import strengths_from_entries, DEFAULT_TEMP
from src.model.plackett_luce import all_trifecta_probs
from src.model.race_type import classify_race, JIKU, STANDARD, CHAOS

FX = Path(__file__).parent / "fixtures"


def _entries():
    return parse_race_card((FX / "gamboo_racecard_7car.html").read_text(encoding="utf-8"))


def test_strengths_normalized_and_monotonic():
    s = strengths_from_entries(_entries())
    assert len(s) == 7
    assert math.isclose(sum(s.values()), 1.0, rel_tol=1e-9)
    # 競走得点が高い5番(90.33)は低い6番(78.19)より強い
    assert s[5] > s[6]
    # 全て正
    assert all(v > 0 for v in s.values())


def test_end_to_end_racecard_to_trifecta():
    s = strengths_from_entries(_entries())
    probs = all_trifecta_probs(s)
    assert len(probs) == 210                       # 7*6*5
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-9)


def test_temperature_controls_spread():
    entries = _entries()
    tight = strengths_from_entries(entries, temp=2.0)     # 本命偏重
    flat = strengths_from_entries(entries, temp=40.0)     # 横一線
    assert max(tight.values()) > max(flat.values())


def test_race_type_thresholds():
    # 一強 → 軸堅
    solid = {1: 0.7, 2: 0.1, 3: 0.06, 4: 0.05, 5: 0.04, 6: 0.03, 7: 0.02}
    assert classify_race(solid).label == JIKU
    # 完全横一線 → 混戦
    even = {c: 1 / 7 for c in range(1, 8)}
    assert classify_race(even).label == CHAOS
    r = classify_race(even)
    assert math.isclose(r.entropy_norm, 1.0, rel_tol=1e-9)
    assert r.top1_win_prob == pytest.approx(1 / 7)


def test_race_type_on_real_card():
    s = strengths_from_entries(_entries())
    rt = classify_race(s)
    assert rt.label in (JIKU, STANDARD, CHAOS)
    assert 0.0 <= rt.entropy_norm <= 1.0
    assert rt.top2_win_prob >= rt.top1_win_prob


def test_empty_entries():
    assert strengths_from_entries([]) == {}
    assert classify_race({}).label == STANDARD
