# KEIRIN 期待値予測API（Phase5）

ガールズケイリン期待値予測AIの運用API。当日レースの **レースタイプ判定・三連単210通り確率・買い目候補・必要オッズ** を返す。
確率/EV/レースタイプのロジックは既存 `src/` を import して再利用しており（重複実装なし）、
レスポンスは `dashboard/data.json` のデータ契約に整合させている。

> モデルは未学習のため、強さは **競走得点ベースのベースライン**（`src/model/strength.py`）。
> バケットROI・レースタイプ分布・キャリブレーション等の集計項目は `status="pending"` で形だけ返す。

## 起動方法

```bash
# 依存（requirements.txt に既に fastapi / uvicorn[standard] あり）
pip install -r requirements.txt
pip install httpx        # テスト（FastAPI TestClient）に必要

# プロジェクトルート（KEIRIN/）で起動
uvicorn api.main:app --reload
# → http://127.0.0.1:8000  /  対話ドキュメント: http://127.0.0.1:8000/docs
```

- SQLite パスは既定で `data/keirin.sqlite`。環境変数 `KEIRIN_DB_SQLITE` で上書き可。
- DBは **読み取り専用（mode=ro）** で短時間だけ開いてすぐ閉じる（収集プロセスのロックを避ける）。
- CORS は全許可（ダッシュボードから直接叩ける）。

## エンドポイント

| メソッド | パス | 概要 |
|---|---|---|
| GET | `/health` | 稼働確認。`{status, service, model_ready}` |
| GET | `/dataset/status` | 収集状況。`data.json` の `data_status` 相当をSQLiteから実データ集計 |
| GET | `/buckets` | バケット別ROI/キャリブレーション。現状 `status="pending"`（3タイプ×5オッズ帯=15セル） |
| GET | `/race-types` | レースタイプ分布。現状 `status="pending"` |
| POST | `/predict` | 出走選手リスト（＋任意オッズ）→ レースタイプ・210通り確率上位・必要オッズ・EV |

### POST /predict

リクエスト:

```json
{
  "riders": [
    {"car_number": 1, "racing_score": 55.0, "leg_type": "逃", "bracket_number": 1, "age": 24},
    {"car_number": 2, "racing_score": 52.0},
    "... 7車分 ..."
  ],
  "odds": {"5-3-1": 50.0, "5-1-3": 62.0, "... 210点（任意） ...": 0.0},
  "ev_threshold": 1.10,
  "top_n": 20
}
```

- `riders[]`: 強さ計算に必須なのは `car_number` と `racing_score` のみ。
  脚質・枠番・年齢等は将来の特徴量拡張用に受け取るが、現ベースラインでは競走得点だけを使う。
- `odds`（任意）: キーは `"a-b-c"` 形式の着順文字列、値は表示オッズ。EV算出には全210点を渡すのが前提
  （`src/ev/market.py` で控除率を除去するため）。

レスポンス（抜粋）:

```json
{
  "field_size": 7,
  "race_type": {"label": "標準", "top1_win_prob": 0.28, "top2_win_prob": 0.49, "entropy_norm": 0.9},
  "win_probs": [{"car_number": 5, "win_prob": 0.28}, "..."],
  "n_combos": 210,
  "top_probs": [{"combo": "5-3-1", "cars": [5,3,1], "prob": 0.0239, "necessary_odds": 41.79}, "..."],
  "ev": {"status": "pending | ok", "n_buy": 0, "buy": [], "top_by_ev": [], "by_bucket": {}}
}
```

- `necessary_odds = 1 / prob`（損益分岐の必要オッズ）。
- `ev` は `odds` 未指定なら `status="pending"`。指定時は `src/ev/ev_engine.build_trifecta_ev_table`
  でEV表を構築し、`ev_threshold` 超の `buy`、`by_bucket`（オッズ帯別集計）を返す。

## 処理パイプライン（/predict）

```
riders → src.model.strength.strengths_from_entries   （競走得点→強さ Σ=1、= 各車の1着確率）
       → src.model.plackett_luce.all_trifecta_probs   （三連単210通りの確率）
       → src.model.race_type.classify_race            （軸堅/標準/混戦）
       → （odds指定時）src.ev.ev_engine.build_trifecta_ev_table  （EV表・買い目）
```

## ダッシュボードとの関係

- `/dataset/status` は `scripts/build_dashboard_data.py` の集計関数（`collect_data_status`）をそのまま
  import して返すため、`dashboard/data.json` の `data_status` と同一スキーマ。
- `/buckets`・`/race-types` は同スクリプトの `pending_model_sections()` を流用し、`data.json` の
  `buckets` / `race_type_dist` と同形の pending を返す。
- ダッシュボードは静的 `data.json`（バッチ生成）を、当日レースの動的予測はこのAPIを叩く、という役割分担。

## ファイル構成

```
api/
├── __init__.py       # パッケージ
├── _bootstrap.py     # sys.path にプロジェクトルートを追加（src/config/scripts を import 可能に）
├── main.py           # FastAPI アプリ・エンドポイント
├── schemas.py        # Pydantic モデル（リクエスト/レスポンス）
├── db.py             # SQLite 読み取り専用アクセス（mode=ro・短時間）
├── README.md
└── tests/
    └── test_api.py   # TestClient による各エンドポイント検証
```

## テスト

```bash
pip install httpx
python -m pytest api/tests/ -q
```
