"""学習済み Plackett-Luce モデルのロードと推論（API用ラッパ）。

起動時に `src.model.persist.load_model()` を一度だけ試み、成功すればプロセス内に保持する。
`/predict` は本モジュールの `predict_strengths` を経由して強さ（各車の1着確率 Σ=1）を得る:
  学習済みモデルで特徴量が揃えば実確率、揃わない（gear_ratio欠損等）／モデル未ロードなら
  競走得点ベースのベースライン（src.model.strength）にグレースフルにフォールバックする。

モデル成果物が読めない環境でも import・起動は必ず成功する（例外は握って model=None）。
"""
from __future__ import annotations

from typing import Optional

from api import _bootstrap  # noqa: F401  sys.path にプロジェクトルートを載せる

from src.collect.gamboo_racecard import Entry
from src.model.persist import DEFAULT_MODEL_PATH, load_model, strengths_from_model
from src.model.strength import strengths_from_entries
from src.model.train_pl import PLModel

# ソース識別子（レスポンスの model_source / /model/info で使う）
SOURCE_TRAINED = "trained_model"
SOURCE_BASELINE = "baseline"

# /predict は出走選手の生入力（直近4ヶ月データ無し）を受けるため、基本8特徴量の
# モデルを使う（拡張20特徴モデルは recent_form を要し生入力では駆動できない）。
# 高精度な拡張モデルによる予測は出走表を取得する scripts/predict_race・/predictions で使う。
_BASE_MODEL_PATH = DEFAULT_MODEL_PATH.parent / "pl_model_base.pkl"


def _try_load_model() -> tuple[Optional[PLModel], Optional[str]]:
    """/predict 用の基本特徴モデルを1度だけロードする。失敗なら (None, 理由)。"""
    for path in (_BASE_MODEL_PATH, DEFAULT_MODEL_PATH):   # 基本→無ければ既定
        try:
            return load_model(path), None
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
    return None, last


# 起動時に一度だけロードを試みてモジュール変数に保持する。
_MODEL, _LOAD_ERROR = _try_load_model()


def model_loaded() -> bool:
    """学習済みモデルがプロセス内に保持されているか。"""
    return _MODEL is not None


def _to_entries(riders) -> list[Entry]:
    """API入力 riders（RiderInput 相当）を既存 Entry へ変換する。無い項目は None。"""
    return [
        Entry(
            car_number=r.car_number,
            bracket_number=r.bracket_number,
            rider_name=r.rider_name or "",
            prefecture=r.prefecture or "",
            age=r.age,
            term=r.term,
            class_rank=r.class_rank or "",
            leg_type=r.leg_type or "",
            gear_ratio=r.gear_ratio,
            racing_score=r.racing_score,
        )
        for r in riders
    ]


def predict_strengths(riders) -> tuple[dict[int, float], str]:
    """riders → ({車番: 1着確率 Σ=1}, source)。

    学習済みモデルがあり特徴量が揃えば `strengths_from_model`（source="trained_model"）。
    空辞書（特徴量欠損）またはモデル未ロードなら `strengths_from_entries`（source="baseline"）。
    どちらのソースでも算出できなければ {} を返す（呼び出し側で 422）。
    """
    entries = _to_entries(riders)

    if _MODEL is not None:
        strengths = strengths_from_model(_MODEL, entries)
        if strengths:
            return strengths, SOURCE_TRAINED

    # フォールバック（モデル未ロード or 特徴量欠損）
    return strengths_from_entries(entries), SOURCE_BASELINE


def model_info() -> dict:
    """/model/info 用のモデルメタ情報。モデルが無ければ loaded=False。"""
    info: dict = {
        "loaded": model_loaded(),
        "model_path": str(DEFAULT_MODEL_PATH),
        "default_source": SOURCE_TRAINED if model_loaded() else SOURCE_BASELINE,
    }
    if _MODEL is not None:
        feature_names = list(getattr(_MODEL, "feature_names", []) or [])
        weights = getattr(_MODEL, "weights", None)
        info["feature_names"] = feature_names
        info["weights"] = (
            {name: float(w) for name, w in zip(feature_names, list(weights))}
            if weights is not None and feature_names
            else None
        )
    else:
        info["feature_names"] = None
        info["weights"] = None
        if _LOAD_ERROR:
            info["load_error"] = _LOAD_ERROR
    return info
