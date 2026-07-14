"""Plackett-Luce 線形モデルの学習（S3、仕様書「PL線形」）。

各選手の強さ log s_i = w·z_i（z=標準化特徴量）とし、観測着順（上位3車の順序）の
Plackett-Luce 尤度を最大化して w を推定する。softmaxはレース内シフト不変なので、レース内で
変動する特徴量のみ意味を持つ（training_data.PL_FEATURES）。

学習後は predict_strengths → src.model.plackett_luce.all_trifecta_probs で210通り確率を出す。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.model.training_data import RaceSample, standardize
# scipy は学習(train_pl)でのみ使う。推論（PLModel.strengths）は numpy だけで動くよう
# import は train_pl 関数内に遅延させる（Actionsの軽量な予測更新環境で scipy 不要にするため）。

_EPS = 1e-12


@dataclass
class PLModel:
    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str] = field(default_factory=list)

    def strengths(self, X: np.ndarray, car_numbers: list[int]) -> dict[int, float]:
        """特徴量行列 → {車番: 強さ}(Σ=1)。強さ ∝ exp(w·z)。"""
        z = (X - self.mean) / self.std
        logit = z @ self.weights
        logit -= logit.max()
        s = np.exp(logit)
        tot = s.sum()
        return {car: float(v / tot) for car, v in zip(car_numbers, s)}


def _neg_loglik(w: np.ndarray, Xs: list[np.ndarray], idxs: list[list[int]],
                l2: float) -> float:
    """全レースの上位3着 PL 負対数尤度（+ L2正則化）。idxs=各レースの着順の行インデックス。"""
    nll = 0.0
    for X, order_idx in zip(Xs, idxs):
        logit = X @ w
        logit -= logit.max()
        s = np.exp(logit)
        remaining = s.sum()
        for i in order_idx:
            nll -= np.log(s[i] / (remaining + _EPS) + _EPS)
            remaining -= s[i]
    return nll + l2 * float(w @ w)


def train_pl(samples: list[RaceSample], l2: float = 1.0) -> PLModel:
    """PL線形モデルを学習する。samples は training_data.load_samples の出力。"""
    from scipy.optimize import minimize   # 学習時のみ必要（推論は numpy だけ）
    if not samples:
        raise ValueError("no training samples")
    mean, std = standardize(samples)
    names = samples[0].feature_names
    # 各レースの標準化済み特徴量と、着順の行インデックス列を用意
    Xs, idxs = [], []
    for s in samples:
        z = (s.X - mean) / std
        car_to_row = {c: r for r, c in enumerate(s.car_numbers)}
        order_idx = [car_to_row[c] for c in s.order if c in car_to_row]
        if len(order_idx) < 3:
            continue
        Xs.append(z)
        idxs.append(order_idx)

    w0 = np.zeros(len(names))
    res = minimize(_neg_loglik, w0, args=(Xs, idxs, l2), method="L-BFGS-B")
    return PLModel(weights=res.x, mean=mean, std=std, feature_names=names)
