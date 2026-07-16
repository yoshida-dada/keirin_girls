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
DEFAULT_ELO_STATE_PATH = DEFAULT_MODEL_PATH.parent / "elo_state.json"


def save_elo_state(state: dict, path: str | Path = DEFAULT_ELO_STATE_PATH) -> Path:
    """最終Elo {氏名: Elo} をJSONで保存（ライブ予測で選手の現在Eloを引く）。"""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return path


def load_elo_state(path: str | Path = DEFAULT_ELO_STATE_PATH) -> dict:
    """保存済みElo状態を読む。無ければ {}（全員デフォルトElo扱い）。"""
    import json
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_model(model, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """PL線形 / LightGBM(lambdarank) いずれのモデルも保存する（kindで判別）。

    両者とも .strengths(X, car_numbers) を実装し推論経路(strengths_from_model)は共通。
    LightGBMは booster を文字列化して可搬・pickle安全に保存する。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if type(model).__name__ == "GBDTModel":       # lightgbm を必須化しないため型名で判定
        payload = {"kind": "gbdt", "booster_str": model.booster.model_to_string(),
                   "mean": model.mean, "std": model.std,
                   "feature_names": model.feature_names,
                   "standardize_x": getattr(model, "standardize_x", True)}
    else:
        payload = {"kind": "pl", "weights": model.weights, "mean": model.mean,
                   "std": model.std, "feature_names": model.feature_names, "features": PL_FEATURES}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    """保存済みモデルを読む。kind=="gbdt" は GBDTModel、それ以外は PLModel（後方互換）。"""
    with open(path, "rb") as f:
        d = pickle.load(f)
    if d.get("kind") == "gbdt":
        import lightgbm as lgb
        from src.model.train_gbdt import GBDTModel
        booster = lgb.Booster(model_str=d["booster_str"])
        return GBDTModel(booster=booster, mean=d["mean"], std=d["std"],
                         feature_names=d["feature_names"],
                         standardize_x=d.get("standardize_x", True))
    return PLModel(weights=d["weights"], mean=d["mean"], std=d["std"],
                   feature_names=d["feature_names"])


def strengths_from_model(model: PLModel, entries: list[Entry],
                         recent: dict | None = None,
                         elo_state: dict | None = None,
                         tactics_ctx: dict | None = None,
                         narabi_ctx: dict | None = None) -> dict[int, float]:
    """出走選手 → {車番: 1着確率}(Σ=1)。特徴量を組み立てて学習済みモデルで推論する。

    モデルの学習特徴（model.feature_names）に追従。拡張モデルは直近4ヶ月(recent)を、
    Elo付きモデルは elo_state({氏名: Elo}) を、展開特徴付きモデルは tactics_ctx（current_tactics
    の氏名別 as-of history）を必要とする。特徴量が揃わなければ {} を返す。
    """
    import pandas as pd
    feats = model.feature_names or PL_FEATURES
    df = build_features(entries, recent or {})
    if "rel_elo" in feats:                      # Eloモデル: レース内相対Eloを列追加
        from src.model.elo import DEFAULT_ELO
        state = elo_state or {}
        elos = pd.Series({e.car_number: state.get(e.rider_name, DEFAULT_ELO) for e in entries})
        df["rel_elo"] = elos - elos.mean()
    from src.features.tactics_features import TACTIC_NAMES
    if any(n in feats for n in TACTIC_NAMES):   # 展開特徴付きモデル: 10列を推論と同一関数で付与
        from src.features.rider_tactics import tactics_for_entries
        from src.features.tactics_features import tactic_columns
        tac = tactics_for_entries(entries, recent or {}, tactics_ctx or {})
        cols = tactic_columns(list(df.index), tac)          # {car: [A(6)...B(4)]}
        for i, name in enumerate(TACTIC_NAMES):
            df[name] = [cols[c][i] for c in df.index]
    from src.features.rider_narabi import NARABI_KEYS
    if any(n in feats for n in NARABI_KEYS):    # 並び予想付きモデル: 3列を推論と同一関数で付与
        from src.features.rider_narabi import narabi_from_order, narabi_columns
        nb = narabi_ctx or {}
        per_car = narabi_from_order(nb.get("order") or [], nb.get("legs") or {})
        ncols = narabi_columns(list(df.index), per_car)
        for i, name in enumerate(NARABI_KEYS):
            df[name] = [ncols[c][i] for c in df.index]
    if df[feats].isna().any().any():
        return {}
    cars = list(df.index)
    X = df.loc[cars, feats].to_numpy(dtype=float)
    return model.strengths(X, cars)


def trifecta_from_model(model: PLModel, entries: list[Entry],
                        recent: dict | None = None, elo_state: dict | None = None) -> dict[tuple, float]:
    """出走選手 → 三連単210通り確率 {(a,b,c): p}。強さが出せなければ {}。"""
    strengths = strengths_from_model(model, entries, recent, elo_state)
    return all_trifecta_probs(strengths) if strengths else {}
