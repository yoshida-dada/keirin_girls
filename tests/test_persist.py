"""モデル永続化＋推論ヘルパーのテスト。"""
import math
from pathlib import Path

import numpy as np

from src.model.train_pl import PLModel
from src.model.training_data import PL_FEATURES
from src.model.persist import (
    save_model, load_model, strengths_from_model, trifecta_from_model,
)
from src.collect.gamboo_racecard import Entry


def _model():
    return PLModel(weights=np.ones(len(PL_FEATURES)) * 0.1,
                   mean=np.zeros(len(PL_FEATURES)), std=np.ones(len(PL_FEATURES)),
                   feature_names=list(PL_FEATURES))


def _entries():
    return [Entry(car_number=i, bracket_number=i, rider_name=f"r{i}", prefecture="東京",
                  age=None, term=120, class_rank="L1", leg_type="逃", gear_ratio=3.9,
                  racing_score=s) for i, s in zip(range(1, 8), [56, 54, 53, 52, 51, 50, 49])]


def test_save_load_roundtrip(tmp_path):
    p = save_model(_model(), tmp_path / "m.pkl")
    assert p.exists()
    m = load_model(p)
    assert list(m.feature_names) == list(PL_FEATURES)
    assert np.allclose(m.weights, 0.1)


def test_strengths_and_trifecta(tmp_path):
    p = save_model(_model(), tmp_path / "m.pkl")
    m = load_model(p)
    st = strengths_from_model(m, _entries())
    assert math.isclose(sum(st.values()), 1.0, rel_tol=1e-9)
    tri = trifecta_from_model(m, _entries())
    assert len(tri) == 210


def test_missing_features_returns_empty(tmp_path):
    # 競走得点欠損 → 特徴量が揃わず {} を返す（呼び出し側でベースラインへ）
    m = load_model(save_model(_model(), tmp_path / "m.pkl"))
    bad = _entries()
    bad[0].racing_score = None
    assert strengths_from_model(m, bad) == {}
