"""Phase5 API のエンドポイント検証（FastAPI TestClient）。

/predict はダミー7車入力で 210通り確率とレースタイプが返ることを確認する。
DB 依存の /dataset/status は DB が無い環境でも落ちないよう分岐して検証する。
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api import model as model_service
from api.db import default_db_path
from api.main import app

client = TestClient(app)

# ダミー7車（競走得点にばらつき）。学習済みモデルの特徴量（gear_ratio/leg_type）を付与。
SEVEN_RIDERS = [
    {"car_number": 1, "racing_score": 55.0, "leg_type": "逃", "bracket_number": 1, "age": 24, "gear_ratio": 3.85},
    {"car_number": 2, "racing_score": 52.0, "leg_type": "追", "bracket_number": 2, "age": 27, "gear_ratio": 3.85},
    {"car_number": 3, "racing_score": 58.0, "leg_type": "捲", "bracket_number": 3, "age": 22, "gear_ratio": 3.92},
    {"car_number": 4, "racing_score": 50.0, "leg_type": "差", "bracket_number": 4, "age": 30, "gear_ratio": 3.85},
    {"car_number": 5, "racing_score": 61.0, "leg_type": "逃", "bracket_number": 5, "age": 25, "gear_ratio": 3.92},
    {"car_number": 6, "racing_score": 49.0, "leg_type": "マーク", "bracket_number": 6, "age": 33, "gear_ratio": 3.85},
    {"car_number": 7, "racing_score": 54.0, "leg_type": "差", "bracket_number": 7, "age": 28, "gear_ratio": 3.85},
]

# 競走得点だけの7車（gear_ratio 欠損）。学習済みモデルの特徴量が揃わずベースラインになる。
SEVEN_RIDERS_SCORES_ONLY = [
    {"car_number": i, "racing_score": s}
    for i, s in enumerate([55.0, 52.0, 58.0, 50.0, 61.0, 49.0, 54.0], start=1)
]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # model_ready は学習済みモデルの実ロード状況を反映する。
    assert body["model_ready"] is model_service.model_loaded()


def test_buckets_pending():
    r = client.get("/buckets")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending"
    # data.json 契約: 3タイプ × 5オッズ帯 = 15 セル
    assert len(body["cells"]) == 15
    assert body["race_types"] == ["軸堅", "標準", "混戦"]


def test_race_types_pending():
    r = client.get("/race-types")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending"
    assert {c["type"] for c in body["counts"]} == {"軸堅", "標準", "混戦"}


def test_dataset_status():
    r = client.get("/dataset/status")
    # DB があれば 200 / 集計、無ければ 503。どちらも許容し形を検証。
    if default_db_path().exists():
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ok"
        assert body["races_total"] >= 0
        assert "sample_counts" in body
    else:
        assert r.status_code == 503


def test_predict_seven_riders_returns_210_and_racetype():
    r = client.post("/predict", json={"riders": SEVEN_RIDERS, "top_n": 30})
    assert r.status_code == 200, r.text
    body = r.json()
    # 7車 → 三連単 210 通り
    assert body["field_size"] == 7
    assert body["n_combos"] == 210
    # レースタイプが 3種のいずれか
    assert body["race_type"]["label"] in ("軸堅", "標準", "混戦")
    # 1着確率は Σ≈1
    total = sum(w["win_prob"] for w in body["win_probs"])
    assert abs(total - 1.0) < 1e-3
    # 確率上位は top_n 件、必要オッズ = 1/prob
    assert len(body["top_probs"]) == 30
    top = body["top_probs"][0]
    assert abs(top["necessary_odds"] - 1.0 / top["prob"]) < 0.5
    # オッズ未指定なので EV は pending
    assert body["ev"]["status"] == "pending"


def test_predict_with_odds_builds_ev():
    # 210点の擬似オッズ = 必要オッズの1.2倍（全点で EV>1 になる作為的な入力）
    probe = client.post("/predict", json={"riders": SEVEN_RIDERS, "top_n": 210})
    combos = probe.json()["top_probs"]
    odds = {c["combo"]: round(1.2 * c["necessary_odds"], 1) for c in combos}

    r = client.post(
        "/predict",
        json={"riders": SEVEN_RIDERS, "odds": odds, "ev_threshold": 1.10},
    )
    assert r.status_code == 200, r.text
    ev = r.json()["ev"]
    assert ev["status"] == "ok"
    assert ev["n_candidates"] == 210
    assert ev["n_buy"] >= 1
    # 買い目行の combo は "a-b-c" 形式
    assert all("-" in b["combo"] for b in ev["buy"])


def test_predict_rejects_no_scores():
    riders = [{"car_number": i} for i in range(1, 8)]  # racing_score 無し
    r = client.post("/predict", json={"riders": riders})
    assert r.status_code == 422


def test_predict_uses_trained_model():
    """gear_ratio/leg_type 付き7車 → 学習済みモデルで実確率が返る。"""
    assert model_service.model_loaded(), "学習済みモデル(pl_model.pkl)が必要"
    r = client.post("/predict", json={"riders": SEVEN_RIDERS, "top_n": 30})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_source"] == "trained_model"
    assert body["field_size"] == 7
    assert body["n_combos"] == 210
    total = sum(w["win_prob"] for w in body["win_probs"])
    assert abs(total - 1.0) < 1e-3


def test_predict_falls_back_to_baseline():
    """競走得点のみ（特徴量欠損）→ ベースラインにフォールバックする。"""
    r = client.post("/predict", json={"riders": SEVEN_RIDERS_SCORES_ONLY, "top_n": 30})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_source"] == "baseline"
    assert body["n_combos"] == 210
    total = sum(w["win_prob"] for w in body["win_probs"])
    assert abs(total - 1.0) < 1e-3


def test_model_info():
    r = client.get("/model/info")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["loaded"] is model_service.model_loaded()
    assert body["default_source"] in ("trained_model", "baseline")


def test_predictions_today_structure():
    """本日のレース予測: 200 と JSON 構造（status/note と races の各キー）を検証。"""
    r = client.get("/predictions/today")
    assert r.status_code == 200, r.text
    body = r.json()
    # 契約: status は "ok" | "pending"、常に note と races(list) を持つ
    assert body["status"] in ("ok", "pending")
    assert "note" in body
    assert isinstance(body["races"], list)

    if body["status"] == "ok":
        assert isinstance(body.get("model"), str) and body["model"]
        assert isinstance(body.get("date"), str) and body["date"]
        assert len(body["races"]) >= 1
        race = body["races"][0]
        for key in ("venue", "race_no", "deadline", "race_type", "riders", "top_trifecta"):
            assert key in race, f"race に {key} がありません"
        assert race["race_type"] in ("軸堅", "標準", "混戦")
        # riders は win_prob 降順（data.json 契約）
        wins = [rd["win_prob"] for rd in race["riders"]]
        assert wins == sorted(wins, reverse=True)
        rider = race["riders"][0]
        for key in ("car", "name", "score", "leg", "win_rate", "win_prob"):
            assert key in rider, f"rider に {key} がありません"
        if race["top_trifecta"]:
            tf = race["top_trifecta"][0]
            for key in ("combo", "prob", "odds", "need_odds"):
                assert key in tf, f"top_trifecta に {key} がありません"


def test_predictions_today_pending_when_missing(monkeypatch, tmp_path):
    """data.json が無い場合は status='pending' のプレースホルダを 200 で返す。"""
    from api import main as api_main

    monkeypatch.setenv("KEIRIN_DASHBOARD_DATA", str(tmp_path / "does_not_exist.json"))
    # 環境変数を参照するのは _dashboard_data_path なので再読込は不要
    r = client.get("/predictions/today")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "pending"
    assert body["races"] == []
    assert api_main  # import 参照を保持（未使用警告回避）
