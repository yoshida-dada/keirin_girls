"""FastAPI アプリ（Phase5 運用API）。

当日レースの「レースタイプ判定 ＋ 対象バケット ＋ 買い目候補 ＋ 必要オッズ」を返す。
確率・EV・レースタイプのロジックは既存 src/ を import して再利用する（重複実装しない）:
  api.model.predict_strengths（学習済みPLモデル or ベースライン）
    → plackett_luce.all_trifecta_probs → race_type.classify_race
  （オッズ指定時のみ）ev.ev_engine.build_trifecta_ev_table

起動:  uvicorn api.main:app --reload
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import _bootstrap  # noqa: F401  sys.path にプロジェクトルートを載せる
from api import db
from api import model as model_service
from api.schemas import (
    EVRowOut,
    EVSection,
    HealthOut,
    ModelInfoOut,
    PredictRequest,
    PredictResponse,
    RaceTypeOut,
    TrifectaProbOut,
    WinProbOut,
)

# --- 既存ロジック（変更せず import して使う） ---
from src.ev.ev_engine import build_trifecta_ev_table
from src.model.plackett_luce import all_trifecta_probs
from src.model.race_type import classify_race

app = FastAPI(
    title="ガールズケイリン 期待値予測API",
    description="当日レースのレースタイプ判定・210通り確率・買い目候補・必要オッズを返す（Phase5）。"
                " モデル未学習の項目は status='pending' で返す。",
    version="0.1.0",
)

# ダッシュボード（GitHub Pages 等）から叩けるよう CORS を有効化
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _combo_str(combo: tuple[int, int, int]) -> str:
    return "-".join(str(x) for x in combo)


def _parse_combo_key(key: str) -> tuple[int, int, int]:
    parts = key.replace("→", "-").split("-")
    if len(parts) != 3:
        raise ValueError(f"不正なcomboキー: {key!r}（'a-b-c' 形式で指定）")
    a, b, c = (int(x) for x in parts)
    return (a, b, c)


# ------------------------------- エンドポイント -------------------------------
@app.get("/health", response_model=HealthOut, summary="稼働確認")
def health() -> HealthOut:
    # model_ready は学習済みモデルが実際にロードできているかを反映する。
    return HealthOut(
        status="ok", service="keirin-ev-api", model_ready=model_service.model_loaded()
    )


@app.get("/model/info", response_model=ModelInfoOut, summary="学習済みモデルの情報")
def model_info() -> ModelInfoOut:
    """学習済み Plackett-Luce モデルの有無・特徴量名・重み・既定ソースを返す。

    モデルが読めない環境でも loaded=False で 200 を返す（API はベースラインで稼働）。
    """
    return ModelInfoOut(**model_service.model_info())


@app.get("/dataset/status", summary="収集状況（data.json の data_status 相当）")
def dataset_status() -> dict:
    """SQLite を読み取り専用で短時間開いて収集状況を集計して返す。"""
    try:
        return db.get_data_status()
    except Exception as e:  # DB 未配置・ロック等
        raise HTTPException(status_code=503, detail=f"データセットを読めません: {e}")


@app.get("/buckets", summary="バケット別ROI/キャリブレーション（現状 pending）")
def buckets() -> dict:
    """（レースタイプ×オッズ帯）バケット。モデル/バックテスト未完成のため pending を返す。"""
    return db.get_pending_sections()["buckets"]


@app.get("/race-types", summary="レースタイプ分布（現状 pending）")
def race_types() -> dict:
    """レースタイプ分布。分類の全レース集計は未算出のため pending を返す。"""
    return db.get_pending_sections()["race_type_dist"]


def _dashboard_data_path() -> Path:
    """ダッシュボードの data.json パス（環境変数 KEIRIN_DASHBOARD_DATA で上書き可）。"""
    import os

    env = os.environ.get("KEIRIN_DASHBOARD_DATA")
    if env:
        return Path(env)
    return _bootstrap.PROJECT_ROOT / "dashboard" / "data.json"


def _read_dashboard_section(key: str, pending: dict) -> dict:
    """dashboard/data.json の指定セクションを読み取り専用で返す共通ヘルパー。

    data.json が無い / JSON 破損 / 当該セクションが無い / status!="ok" の場合は
    渡された pending プレースホルダ（status="pending"）を 200 で返す。
    """
    path = _dashboard_data_path()
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return dict(pending)

    section = data.get(key)
    if not isinstance(section, dict) or section.get("status") != "ok":
        return dict(pending)
    return section


_PREDICTIONS_PENDING = {
    "status": "pending",
    "date": None,
    "model": None,
    "note": "本日のレース予測はまだ生成されていません（data.json 未生成 or predictions 未収載）。",
    "races": [],
}

_ACCURACY_PENDING = {
    "status": "pending",
    "note": "予測精度はまだ算出されていません（data.json 未生成 or prediction_accuracy 未収載）。",
    "period": None,
    "n_races": None,
    "top1_rate": None,
    "top3_rate": None,
    "trifecta": [],
    "brier": None,
}

_ACCURACY_HISTORY_PENDING = {
    "status": "pending",
    "note": "予測精度の週次推移はまだ算出されていません（data.json 未生成 or accuracy_history 未収載）。",
    "period": None,
    "n_weeks": None,
    "n_races": None,
    "weeks": [],
}


@app.get("/predictions/today", summary="本日のレース予測（着順確率・買い目推奨ではない）")
def predictions_today() -> dict:
    """dashboard/data.json の predictions セクションを読み取り専用で返す。

    これは各車の着順（1着）確率と三連単モデル確率であって買い目推奨ではない。
    data.json が無い / predictions セクションが無い / status!="ok" の場合は
    status="pending" のプレースホルダを返す（ダッシュボードと同じ契約）。
    """
    return _read_dashboard_section("predictions", _PREDICTIONS_PENDING)


@app.get("/accuracy", summary="予測精度（過去レースでのモデル的中率）")
def accuracy() -> dict:
    """dashboard/data.json の prediction_accuracy セクションを読み取り専用で返す。

    検証期間でのモデル的中率（top1/top3・三連単 topn・Brier）。data.json が無い /
    prediction_accuracy セクションが無い / status!="ok" の場合は status="pending" を返す。
    """
    return _read_dashboard_section("prediction_accuracy", _ACCURACY_PENDING)


@app.get("/predictions/history", summary="予測精度の週次推移")
def predictions_history() -> dict:
    """dashboard/data.json の accuracy_history セクションを読み取り専用で返す。

    週次の的中率推移。data.json が無い / accuracy_history セクションが無い /
    status!="ok" の場合は status="pending" のプレースホルダを返す。
    """
    return _read_dashboard_section("accuracy_history", _ACCURACY_HISTORY_PENDING)


@app.post("/predict", response_model=PredictResponse, summary="当日レースの確率・買い目・必要オッズ")
def predict(req: PredictRequest) -> PredictResponse:
    """出走選手リスト（＋任意でオッズ）から、レースタイプ・210通り確率上位・必要オッズを返す。

    強さ（各車の1着確率 Σ=1）は学習済み Plackett-Luce モデルで算出する。特徴量（gear_ratio 等）が
    揃わない／モデル未ロードなら競走得点ベースのベースラインに自動フォールバックする
    （model_source で供給元を返す）。三連単確率・レースタイプ・EV は供給された強さで従来どおり算出。
    """
    # 強さ（Σ=1）と供給元。P(1着)=強さ に一致する（plackett_luce のドキュメント参照）。
    strengths, model_source = model_service.predict_strengths(req.riders)
    if not strengths:
        raise HTTPException(
            status_code=422,
            detail="競走得点(racing_score>0)を持つ選手が居ないため確率を算出できません。",
        )

    # レースタイプ（1着確率分布の形状から 軸堅/標準/混戦）
    rt = classify_race(strengths)

    # 三連単 全通り確率（7車なら 210 通り）
    probs = all_trifecta_probs(strengths)

    # 各車の1着確率
    win_probs = [
        WinProbOut(car_number=car, win_prob=round(p, 6))
        for car, p in sorted(strengths.items(), key=lambda kv: kv[1], reverse=True)
    ]

    # 確率上位 top_n ＋ 必要オッズ(1/p)
    top_sorted = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[: req.top_n]
    top_probs = [
        TrifectaProbOut(
            combo=_combo_str(combo),
            cars=list(combo),
            prob=round(p, 8),
            necessary_odds=round(1.0 / p, 2) if p > 0 else 0.0,
        )
        for combo, p in top_sorted
    ]

    ev_section = _build_ev_section(probs, req.odds, req.ev_threshold)

    return PredictResponse(
        field_size=len(strengths),
        model_source=model_source,
        race_type=RaceTypeOut(
            label=rt.label,
            top1_win_prob=rt.top1_win_prob,
            top2_win_prob=rt.top2_win_prob,
            entropy_norm=rt.entropy_norm,
        ),
        win_probs=win_probs,
        n_combos=len(probs),
        top_probs=top_probs,
        ev=ev_section,
    )


def _row_to_out(r) -> EVRowOut:
    return EVRowOut(
        combo=_combo_str(r.combo),
        cars=list(r.combo),
        prob=r.prob,
        model_prob=r.model_prob,
        market_prob=r.market_prob,
        odds=r.odds,
        odds_bucket=r.odds_bucket,
        ev_gross=r.ev_gross,
        ev_net=r.ev_net,
    )


def _build_ev_section(
    probs: dict[tuple, float], odds_in: Optional[dict[str, float]], ev_threshold: float
) -> EVSection:
    """オッズ指定時のみ EV テーブルを構築。未指定なら pending を返す。"""
    if not odds_in:
        return EVSection(
            status="pending",
            note="オッズ未指定のためEVは未算出です（riders と共に odds を渡すと算出します）。",
            ev_threshold=ev_threshold,
        )
    try:
        odds = {_parse_combo_key(k): float(v) for k, v in odds_in.items()}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    table = build_trifecta_ev_table(probs, odds, ev_threshold=ev_threshold)
    buy = [_row_to_out(r) for r in table["buy"]]
    top_by_ev = [_row_to_out(r) for r in table["all"][:20]]
    return EVSection(
        status="ok",
        ev_threshold=ev_threshold,
        n_candidates=len(table["all"]),
        n_buy=len(buy),
        buy=buy,
        top_by_ev=top_by_ev,
        by_bucket=table["by_bucket"],
    )
