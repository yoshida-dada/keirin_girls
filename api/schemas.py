"""Pydantic モデル（Phase5 API のリクエスト/レスポンス）。

/predict の入出力を型で固める。モデル未学習の項目は status="pending" で返す方針のため、
EV セクションはオッズ未指定時に pending を返せるよう Optional 構造にする。
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ----------------------------- /predict 入力 -----------------------------
class RiderInput(BaseModel):
    """出走選手1名の入力。ベースライン強さに必須なのは car_number と racing_score のみ。

    学習済み Plackett-Luce モデルは racing_score に加えて gear_ratio・leg_type 等の特徴量を使う。
    それらが揃えば実確率（model_source="trained_model"）、欠ければ競走得点ベースの
    ベースライン（model_source="baseline", strength.strengths_from_entries）に自動フォールバックする。
    """
    car_number: int = Field(..., ge=1, le=9, description="車番")
    racing_score: Optional[float] = Field(None, description="競走得点（強さの土台）")
    leg_type: Optional[str] = Field(None, description="脚質 逃/捲/差/マーク")
    bracket_number: Optional[int] = Field(None, description="枠番")
    age: Optional[int] = Field(None, description="年齢")
    term: Optional[int] = Field(None, description="期別")
    class_rank: Optional[str] = Field(None, description="級班 例 L1")
    prefecture: Optional[str] = Field(None, description="府県")
    gear_ratio: Optional[float] = Field(None, description="ギヤ倍数")
    rider_name: Optional[str] = Field(None, description="選手名")


class PredictRequest(BaseModel):
    riders: list[RiderInput] = Field(..., min_length=2, description="出走選手のリスト")
    odds: Optional[dict[str, float]] = Field(
        None,
        description='三連単オッズ（任意）。キーは "a-b-c" 形式の着順文字列、値は表示オッズ。',
    )
    ev_threshold: float = Field(1.10, gt=0, description="購入対象とするEV(gross)の下限")
    top_n: int = Field(20, ge=1, le=210, description="返す確率上位の件数")


# ----------------------------- /predict 出力 -----------------------------
class RaceTypeOut(BaseModel):
    label: str
    top1_win_prob: float
    top2_win_prob: float
    entropy_norm: float


class WinProbOut(BaseModel):
    car_number: int
    win_prob: float


class TrifectaProbOut(BaseModel):
    combo: str                 # "a-b-c"
    cars: list[int]            # [a, b, c]（着順どおり）
    prob: float                # 三連単的中確率
    necessary_odds: float      # 損益分岐の必要オッズ = 1/prob


class EVRowOut(BaseModel):
    combo: str
    cars: list[int]
    prob: float
    model_prob: float
    market_prob: float
    odds: float
    odds_bucket: str
    ev_gross: float
    ev_net: float


class EVSection(BaseModel):
    status: str                        # "ok" | "pending"
    note: Optional[str] = None
    ev_threshold: Optional[float] = None
    n_candidates: Optional[int] = None
    n_buy: Optional[int] = None
    buy: list[EVRowOut] = []
    top_by_ev: list[EVRowOut] = []
    by_bucket: Optional[dict] = None


class PredictResponse(BaseModel):
    field_size: int
    model_source: str                  # "trained_model" | "baseline"（強さの供給元）
    race_type: RaceTypeOut
    win_probs: list[WinProbOut]        # 各車の1着確率（Σ=1）
    n_combos: int                      # 三連単の総通り数（7車なら210）
    top_probs: list[TrifectaProbOut]   # 確率上位（top_n件）
    ev: EVSection


# ----------------------------- 共通 -----------------------------
class HealthOut(BaseModel):
    status: str
    service: str
    model_ready: bool


class ModelInfoOut(BaseModel):
    """GET /model/info の応答。学習済みモデルの有無・特徴量・重み・既定ソース。"""
    loaded: bool                                   # 学習済みモデルを保持しているか
    model_path: str                                # 参照した pkl のパス
    default_source: str                            # "trained_model" | "baseline"
    feature_names: Optional[list[str]] = None      # 学習に使った特徴量名
    weights: Optional[dict[str, float]] = None     # 特徴量名→重み（あれば）
    load_error: Optional[str] = None               # ロード失敗時の理由
