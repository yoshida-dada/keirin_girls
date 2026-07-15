"""LightGBM 着順多クラス分類 + 順位モンテカルロ（S3比較検証用の新規モデル）。

各車を1サンプルとして「着順クラス」を多クラス分類で学習する:
  class 0 = 1着 / 1 = 2着 / 2 = 3着 / 3 = 4着以下
予測 P(1着)/P(2着)/P(3着)/P(4着以下) を使い、順位を **位置別の逐次サンプリング**
（Plackett-Luce 的）で組み立てる:
  1着 ~ P(1着) を出走車で正規化 → 抽選
  2着 ~ 残り車の P(2着) を正規化 → 抽選
  3着 ~ 残り車の P(3着) を正規化 → 抽選
これを多数回繰り返して各車の 1着率・三連単210通り確率・各着率を得る（monte_carlo）。

MC の期待値は解析的な連鎖確率に一致するため、比較指標の安定化のため
`trifecta_probs`/`strengths` は解析版（=MC期待値）を使い、`monte_carlo` を検証用に提供する。

PLModel / GBDTModel と同じ `.strengths(X, car_numbers) -> {車番: P(1着)}(Σ=1)` を実装するので
evaluate() にそのまま渡せる（出力契約が一致）。学習は渡された samples のみ（リーク防止）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np
import lightgbm as lgb

from src.model.training_data import RaceSample, standardize

_EPS = 1e-12
_NUM_CLASS = 4   # 1着 / 2着 / 3着 / 4着以下


@dataclass
class GBDTMultiClassModel:
    """LightGBM 多クラス着順分類モデル + 位置別逐次サンプリングによる三連単確率。"""
    booster: lgb.Booster
    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    standardize_x: bool = True

    def _proba(self, X: np.ndarray) -> np.ndarray:
        """(n_cars, 4) のクラス確率 [P(1着),P(2着),P(3着),P(4+)] を返す。"""
        z = (X - self.mean) / self.std if self.standardize_x else X
        p = np.asarray(self.booster.predict(z), dtype=float).reshape(len(X), _NUM_CLASS)
        return p

    def strengths(self, X: np.ndarray, car_numbers: list[int]) -> dict[int, float]:
        """{車番: P(1着)}(Σ=1)。多クラスの P(1着) を出走車で正規化（evaluate と同契約）。"""
        p1 = self._proba(X)[:, 0]
        tot = float(p1.sum())
        if tot <= 0:
            n = len(car_numbers)
            return {car: 1.0 / n for car in car_numbers}
        return {car: float(v / tot) for car, v in zip(car_numbers, p1)}

    def trifecta_probs(self, X: np.ndarray, car_numbers: list[int]) -> dict[tuple, float]:
        """三連単210通り確率（位置別逐次サンプリングの解析的期待値）。

        P(a→b→c) = P1_a/Σ_all P1 · P2_b/Σ_{≠a} P2 · P3_c/Σ_{≠a,b} P3
        """
        p = self._proba(X)
        cars = list(car_numbers)
        p1 = {c: p[i, 0] for i, c in enumerate(cars)}
        p2 = {c: p[i, 1] for i, c in enumerate(cars)}
        p3 = {c: p[i, 2] for i, c in enumerate(cars)}
        out: dict[tuple, float] = {}
        s1 = sum(p1.values())
        for a, b, c in permutations(cars, 3):
            if s1 <= 0:
                out[(a, b, c)] = 0.0
                continue
            pa = p1[a] / s1
            s2 = sum(p2[x] for x in cars if x != a)
            pb = (p2[b] / s2) if s2 > 0 else 0.0
            s3 = sum(p3[x] for x in cars if x not in (a, b))
            pc = (p3[c] / s3) if s3 > 0 else 0.0
            out[(a, b, c)] = pa * pb * pc
        # 数値誤差の正規化（Σ=1へ）
        tot = sum(out.values())
        if tot > 0:
            out = {k: v / tot for k, v in out.items()}
        return out

    def monte_carlo(self, X: np.ndarray, car_numbers: list[int], n_sims: int = 4000,
                    seed: int = 0) -> dict:
        """位置別逐次サンプリングを n_sims 回。1着率/各着率/三連単確率(カウント)を返す（検証用）。"""
        rng = np.random.default_rng(seed)
        p = self._proba(X)
        cars = np.array(car_numbers)
        p1, p2, p3 = p[:, 0].copy(), p[:, 1].copy(), p[:, 2].copy()
        win = {c: 0 for c in car_numbers}
        place2 = {c: 0 for c in car_numbers}
        place3 = {c: 0 for c in car_numbers}
        tri: dict[tuple, int] = {}

        def _sample(weights: np.ndarray, mask: np.ndarray) -> int:
            w = weights * mask
            s = w.sum()
            if s <= 0:                      # 全ゼロなら残り一様
                w = mask.astype(float)
                s = w.sum()
            r = rng.random() * s
            return int(np.searchsorted(np.cumsum(w), r))

        idx = np.arange(len(cars))
        for _ in range(n_sims):
            mask = np.ones(len(cars))
            i1 = _sample(p1, mask); mask[i1] = 0
            i2 = _sample(p2, mask); mask[i2] = 0
            i3 = _sample(p3, mask)
            a, b, c = int(cars[i1]), int(cars[i2]), int(cars[i3])
            win[a] += 1; place2[b] += 1; place3[c] += 1
            tri[(a, b, c)] = tri.get((a, b, c), 0) + 1
        return {
            "win_rate": {c: win[c] / n_sims for c in car_numbers},
            "place2_rate": {c: place2[c] / n_sims for c in car_numbers},
            "place3_rate": {c: place3[c] / n_sims for c in car_numbers},
            "trifecta": {k: v / n_sims for k, v in tri.items()},
        }


def _class_label(sample: RaceSample) -> np.ndarray:
    """car_numbers 行順の着順クラス（0=1着/1=2着/2=3着/3=4着以下）。"""
    car_to_cls = {car: i for i, car in enumerate(sample.order[:3])}
    return np.array([car_to_cls.get(c, 3) for c in sample.car_numbers], dtype=int)


def train_gbdt_multiclass(
    samples: list[RaceSample],
    *,
    num_leaves: int = 15,
    learning_rate: float = 0.05,
    n_estimators: int = 200,
    min_data_in_leaf: int = 20,
    l2: float = 1.0,
    standardize_x: bool = True,
    seed: int = 42,
) -> GBDTMultiClassModel:
    """LightGBM 多クラス着順分類を学習する。samples は training_data.load_samples の出力。"""
    if not samples:
        raise ValueError("no training samples")
    names = samples[0].feature_names
    mean, std = standardize(samples)

    Xs, ys = [], []
    for s in samples:
        z = (s.X - mean) / std if standardize_x else s.X
        Xs.append(z)
        ys.append(_class_label(s))
    X = np.vstack(Xs)
    y = np.concatenate(ys)

    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
    params = {
        "objective": "multiclass",
        "num_class": _NUM_CLASS,
        "metric": "multi_logloss",
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_data_in_leaf": min_data_in_leaf,
        "lambda_l2": l2,
        "verbosity": -1,
        "seed": seed,
        "deterministic": True,
        "force_col_wise": True,
    }
    booster = lgb.train(params, dtrain, num_boost_round=n_estimators)
    return GBDTMultiClassModel(booster=booster, mean=mean, std=std,
                               feature_names=list(names), standardize_x=standardize_x)
