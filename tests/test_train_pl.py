"""PL線形学習器のテスト（合成データ）。重みが正しい符号を学習し、較正が改善することを確認。"""
import numpy as np

from src.model.training_data import RaceSample
from src.model.train_pl import train_pl
from src.model.evaluate import evaluate, time_split


def _make_samples(n_races=400, n_riders=7, seed=0):
    """特徴量 x0 が強さを決める合成レース。真の log-strength = 1.5*x0。"""
    rng = np.random.default_rng(seed)
    samples = []
    for r in range(n_races):
        X = rng.normal(size=(n_riders, 2))          # 2特徴（x0が効く, x1はノイズ）
        cars = list(range(1, n_riders + 1))
        logit = 1.5 * X[:, 0]
        s = np.exp(logit - logit.max())
        p = s / s.sum()
        # PL逐次サンプリングで着順を生成
        remaining = cars.copy()
        probs = {c: p[i] for i, c in enumerate(cars)}
        order = []
        for _ in range(3):
            w = np.array([probs[c] for c in remaining])
            w = w / w.sum()
            pick = rng.choice(remaining, p=w)
            order.append(int(pick))
            remaining.remove(pick)
        samples.append(RaceSample(race_id=f"R{r}", date=f"2025-01-{(r % 28) + 1:02d}",
                                  car_numbers=cars, X=X, order=order,
                                  feature_names=["x0", "x1"]))
    return samples


def test_learns_correct_sign():
    samples = _make_samples()
    model = train_pl(samples, l2=0.1)
    w = dict(zip(model.feature_names, model.weights))
    assert w["x0"] > 0.5          # 効く特徴は正で大きい
    assert abs(w["x1"]) < w["x0"]  # ノイズ特徴は小さい


def test_beats_random_and_calibrated():
    samples = _make_samples(500)
    train, test = time_split(samples, 0.25)
    model = train_pl(train, l2=0.1)
    m = evaluate(model.strengths, test)
    assert m["top1_acc"] > 1 / 7    # ランダム(1/7)より良い
    assert m["ece"] < 0.08          # 概ね較正されている


def test_strengths_normalized():
    samples = _make_samples(50)
    model = train_pl(samples, l2=0.1)
    s = model.strengths(samples[0].X, samples[0].car_numbers)
    assert abs(sum(s.values()) - 1.0) < 1e-9
