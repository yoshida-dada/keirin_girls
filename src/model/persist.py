"""学習済みモデルの保存・読込＋出走選手からの推論ヘルパー（S3運用連携）。

API/ダッシュボードが学習済みモデルで確率を出せるよう、モデル成果物の入出力と
「出走選手(Entry) → 1着強さ・三連単210通り確率」の一貫した推論経路を提供する。
特徴量の組み立て（assembler＋PL_FEATURES）をここに閉じ込め、呼び出し側は Entry を渡すだけにする。
"""
from __future__ import annotations

import pickle
from pathlib import Path

from src.collect.gamboo_racecard import Entry
from src.features.assembler import build_features
from src.model.training_data import PL_FEATURES
from src.model.plackett_luce import all_trifecta_probs
from src.model.train_pl import PLModel

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "pl_model.pkl"


def save_model(model: PLModel, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"weights": model.weights, "mean": model.mean, "std": model.std,
                     "feature_names": model.feature_names, "features": PL_FEATURES}, f)
    return path


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> PLModel:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return PLModel(weights=d["weights"], mean=d["mean"], std=d["std"],
                   feature_names=d["feature_names"])


def strengths_from_model(model: PLModel, entries: list[Entry],
                         recent: dict | None = None) -> dict[int, float]:
    """出走選手 → {車番: 1着確率}(Σ=1)。特徴量を組み立てて学習済みモデルで推論する。

    モデルの学習特徴（model.feature_names）に追従。拡張モデルは直近4ヶ月(recent)を必要とするため、
    recent を渡すこと。特徴量が揃わない場合は {} を返す（呼び出し側でベースラインにフォールバック）。
    """
    feats = model.feature_names or PL_FEATURES
    df = build_features(entries, recent or {})
    if df[feats].isna().any().any():
        return {}
    cars = list(df.index)
    X = df.loc[cars, feats].to_numpy(dtype=float)
    return model.strengths(X, cars)


def trifecta_from_model(model: PLModel, entries: list[Entry],
                        recent: dict | None = None) -> dict[tuple, float]:
    """出走選手 → 三連単210通り確率 {(a,b,c): p}。強さが出せなければ {}。"""
    strengths = strengths_from_model(model, entries, recent)
    return all_trifecta_probs(strengths) if strengths else {}
