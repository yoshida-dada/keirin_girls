"""キャリブレーション検証（課題B）のテスト。合成データで指標の性質を確認する。"""
import random

import pytest

from src.backtest.calibration import (
    brier_score, reliability_curve, expected_calibration_error, calibration_by_bucket,
)


def test_brier_perfect_and_worst():
    assert brier_score([(1.0, 1), (0.0, 0)]) == 0.0        # 完全予測
    assert brier_score([(1.0, 0), (0.0, 1)]) == 1.0        # 完全に外す
    assert brier_score([]) is None


def test_reliability_well_calibrated():
    # 確率 p のイベントを実際に確率 p で発生させれば、各ビンで mean_pred ≈ emp_freq
    random.seed(0)
    pairs = []
    for _ in range(20000):
        p = random.random()
        y = 1 if random.random() < p else 0
        pairs.append((p, y))
    for b in reliability_curve(pairs, n_bins=10):
        if b.count > 50:
            assert abs(b.mean_pred - b.emp_freq) < 0.05      # 対角線近傍
    assert expected_calibration_error(pairs, 10) < 0.02


def test_reliability_miscalibrated_has_high_ece():
    # 常に0.9と予測するが実際は10%しか当たらない＝過信
    pairs = [(0.9, 1 if i < 10 else 0) for i in range(100)]
    ece = expected_calibration_error(pairs, 10)
    assert ece > 0.5


def test_bins_cover_and_count():
    pairs = [(0.05, 0), (0.15, 1), (0.95, 1)]
    curve = reliability_curve(pairs, n_bins=10)
    assert len(curve) == 10
    assert sum(b.count for b in curve) == 3
    assert curve[0].count == 1 and curve[1].count == 1 and curve[9].count == 1


def test_calibration_by_bucket():
    records = [
        ("軸堅×0-10", 0.5, 1), ("軸堅×0-10", 0.5, 0),
        ("混戦×100-300", 0.02, 0), ("混戦×100-300", 0.02, 0), ("混戦×100-300", 0.02, 1),
    ]
    res = calibration_by_bucket(records)
    assert res["軸堅×0-10"]["n"] == 2
    assert res["軸堅×0-10"]["emp_freq"] == pytest.approx(0.5)
    assert res["混戦×100-300"]["n"] == 3
    assert 0.0 <= res["混戦×100-300"]["brier"] <= 1.0
