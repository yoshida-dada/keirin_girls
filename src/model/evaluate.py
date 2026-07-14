"""モデル評価（S3・課題B）。1着予測の的中率・対数損失・キャリブレーションを測る。

強さ関数（特徴量→{車番:P(1着)}）と検証サンプルを受け取り、モデル比較（PL線形 vs ベースライン）
に使う。三連単確率はこの1着強さから plackett_luce で導かれるため、1着較正が土台になる。
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from src.model.training_data import RaceSample
from src.backtest.calibration import brier_score, expected_calibration_error

StrengthsFn = Callable[[np.ndarray, list], dict]


def evaluate(strengths_fn: StrengthsFn, samples: list[RaceSample]) -> dict:
    """1着予測を評価。戻り値: n / top1_acc / logloss / brier / ece。"""
    n = top1 = 0
    logloss = 0.0
    pairs: list[tuple[float, int]] = []
    for s in samples:
        st = strengths_fn(s.X, s.car_numbers)
        if not st:
            continue
        winner = s.order[0]
        pred = max(st, key=st.get)
        top1 += int(pred == winner)
        logloss += -math.log(st.get(winner, 0.0) + 1e-12)
        for car, p in st.items():
            pairs.append((p, 1 if car == winner else 0))
        n += 1
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "top1_acc": round(top1 / n, 4),
        "logloss": round(logloss / n, 4),
        "brier": round(brier_score(pairs), 5),
        "ece": round(expected_calibration_error(pairs), 5),
    }


def baseline_strengths(feature_names: list[str], temp: float = 8.0) -> StrengthsFn:
    """競走得点のみの指数ベースライン強さ関数（strength.py と同型）。比較の基準線。"""
    idx = feature_names.index("racing_score")

    def fn(X: np.ndarray, cars: list) -> dict:
        scores = X[:, idx]
        z = (scores - scores.mean()) / temp
        z -= z.max()
        s = np.exp(z)
        tot = s.sum()
        return {c: float(v / tot) for c, v in zip(cars, s)}
    return fn


def time_split(samples: list[RaceSample], test_frac: float = 0.2
               ) -> tuple[list[RaceSample], list[RaceSample]]:
    """date昇順のサンプルを 前(train)/後(test) に時系列分割（リークなし評価）。"""
    if not samples:
        return [], []
    k = int(len(samples) * (1 - test_frac))
    return samples[:k], samples[k:]
