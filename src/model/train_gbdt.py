"""LightGBM lambdarank による強さモデルの学習（S3、PL線形の対抗馬）。

各レースを1つの group（=車数）とみなし、着順から作った relevance を LambdaRank で
学習する。PLModel と同じ .strengths(X, car_numbers) -> {車番: 確率(Σ=1)} を実装するので、
evaluate() / plackett_luce.all_trifecta_probs にそのまま渡せる（出力契約が一致）。

relevance（上位ほど大）: 1着=3, 2着=2, 3着=1, それ以外=0。RaceSample.order（上位3車番）と
car_numbers を突き合わせて各行に付与する。予測スコアは softmax で Σ=1 に正規化して確率化する。

学習は渡された samples のみで行う（時系列分割は呼び出し側の責務、リーク防止）。
recent_form/age 未取得のため特徴量は限定的（training_data.PL_FEATURES）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import lightgbm as lgb

from src.model.training_data import RaceSample, standardize

# 着順 → relevance（上位ほど高い）。それ以外の車は 0。
_RELEVANCE = {0: 3, 1: 2, 2: 1}   # order のインデックス(0=1着) → relevance


@dataclass
class GBDTModel:
    """LightGBM lambdarank モデル。強さ = softmax(予測スコア)（Σ=1）。"""
    booster: lgb.Booster
    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    standardize_x: bool = True

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std if self.standardize_x else X

    def strengths(self, X: np.ndarray, car_numbers: list[int]) -> dict[int, float]:
        """特徴量行列 → {車番: 強さ}(Σ=1)。予測スコアの softmax（PLModel と同契約）。"""
        z = self._transform(X)
        score = self.booster.predict(z)
        score = np.asarray(score, dtype=float).reshape(-1)
        score -= score.max()
        s = np.exp(score)
        tot = s.sum()
        if tot <= 0:
            n = len(car_numbers)
            return {car: 1.0 / n for car in car_numbers}
        return {car: float(v / tot) for car, v in zip(car_numbers, s)}


def _relevances(sample: RaceSample) -> np.ndarray:
    """RaceSample の car_numbers 行順に対応する relevance ベクトルを作る。"""
    car_to_rel = {car: _RELEVANCE[i] for i, car in enumerate(sample.order[:3])}
    return np.array([car_to_rel.get(c, 0) for c in sample.car_numbers], dtype=int)


def train_gbdt(
    samples: list[RaceSample],
    *,
    num_leaves: int = 15,
    learning_rate: float = 0.05,
    n_estimators: int = 200,
    min_data_in_leaf: int = 20,
    l2: float = 1.0,
    standardize_x: bool = True,
    seed: int = 42,
) -> GBDTModel:
    """LightGBM lambdarank を学習する。samples は training_data.load_samples の出力。

    group は各レース（RaceSample ごとの車数）。ラベルは着順由来の relevance
    （1着=3/2着=2/3着=1/他=0）。標準化は任意（既定 True）だが木モデルなので単調変換に
    ほぼ不変で、PL線形と特徴量スケールを揃える意味で既定 True。
    """
    if not samples:
        raise ValueError("no training samples")
    names = samples[0].feature_names
    mean, std = standardize(samples)

    Xs, ys, groups = [], [], []
    for s in samples:
        z = (s.X - mean) / std if standardize_x else s.X
        Xs.append(z)
        ys.append(_relevances(s))
        groups.append(len(s.car_numbers))

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    group = np.array(groups, dtype=int)

    dtrain = lgb.Dataset(X, label=y, group=group, free_raw_data=False)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3],
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_data_in_leaf": min_data_in_leaf,
        "lambda_l2": l2,
        "max_position": 3,       # 上位3着に評価を集中（三連単の土台）
        "verbosity": -1,
        "seed": seed,
        "deterministic": True,
        "force_col_wise": True,
    }
    booster = lgb.train(params, dtrain, num_boost_round=n_estimators)
    return GBDTModel(booster=booster, mean=mean, std=std,
                     feature_names=list(names), standardize_x=standardize_x)
