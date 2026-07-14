# ガールズケイリン 期待値予測AI - 開発ガイド

詳細仕様は [girls_keirin_ai_spec.md](girls_keirin_ai_spec.md)。本ファイルは開発の約束事と地図。

## コンセプト（1段落）
全レース・全210通りの3連単確率を Plackett-Luce で推定し、「レースタイプ(軸堅/標準/混戦) × オッズ帯」の
バケットごとに実現ROIを検証。**ROIが100%を超えるゾーンだけ**を実運用対象にする。核心はモデル精度より
Phase4のバケット分析。狙い目マップ完成までは実弾投入しない。

## ディレクトリ構成
```
KEIRIN/
├── girls_keirin_ai_spec.md   # 仕様書（v3）
├── config/settings.py        # 定数・DB接続・パス（絶対パス禁止, 環境変数優先）
├── db/schema.sql             # PostgreSQL スキーマ
├── src/
│   ├── collect/   # S1 スクレイパー群（出走表/結果/選手成績/確定オッズ/オッズ時系列）
│   ├── features/  # S2 特徴量（仕様書2章。Tier1→レース全体→派生）
│   ├── model/     # S3 Plackett-Luce + レースタイプ分類
│   ├── ev/        # S4 EVエンジン（競馬予想から移植）
│   └── backtest/  # S5 バケット分析・バックテスト（戦略の核）
├── scripts/       # 実行スクリプト（収集・学習・バックテスト）
└── tests/
```

## 流用元（重要）
成熟した賭式エンジンが姉妹プロジェクト `../競馬予想/` にある。S4は原則ここから移植する:
| 用途 | 移植元 |
|---|---|
| オッズ逆算確率 | `analyzer/market.py: implied_win_probs` |
| logit blend | `analyzer/market.py: blend_loglinear` |
| EV表・買い目選定 | `betting/ev_table.py: build_ev_table` |
| quarter Kelly | `betting/bankroll.py: kelly_stake_yen` |
| 資金配分・上限 | `betting/staking.py`, `apply_hard_limits` |
| オッズ時系列収集 | `collect_odds_snapshot.py` |
| バックテスト骨格 | `backtest.py` |
競馬予想は SQLite だが本プロジェクトは PostgreSQL。ロジックは流用しDB層は置換する。

## データ取得元（S1の前提, 2026-07-14調査）
| 種別 | 主系 | 代替 |
|---|---|---|
| 出走表 / 結果 / 3連単確定オッズ全210点 | **GambooBET(keirin.kdreams.jp)** HTML静的・ログイン不要・EUC-JP | Winticket 非公開JSON API（グレー） |
| 選手成績詳細 | 競輪ステーション + オッズパーク（相互補完） | 結果の決まり手から率を自前再計算 |
| オッズ時系列（締切前5〜10分刻み） | **配布元なし → 自前ポーリングで自作**（GambooBET/Winticketを定期取得） | keirinodds.com は検証参照のみ |
- **netkeirin(keirin.netkeiba.com)は使わない**（規約でスクレイピング禁止＋自動IP-BAN）。
- robots.txtと利用規約は別物。投票サイトは実装前に規約本文の逐条確認が必須（未実施）。

## 開発ルール
1. **不足情報・仕様不明は実装前に必ず確認する。** 「たぶんこう」で進めない。
2. **懸念は代替案とセットで報告する。**
3. パスは `pathlib` + 相対。絶対パス埋め込み禁止。秘匿値は環境変数（`.env`, `config/settings.py`）。
4. スクレイピングは規約遵守・1秒以上の間隔（課題G）。
5. **リーク防止**: 選手成績は as_of（`snapshot_date` ≤ レース日で最新）で結合。確定直前オッズを学習特徴量に使わない（2.7）。
6. Anthropic SDK は公式SDKを使う（raw requests 不可）。最新モデルを使う。

## 作業単位（手順）と skill/subagent 方針
- **S0** 基盤+スキーマ（済）/ **S1** データ基盤 / **S2** 特徴量 / **S3** 確率モデル / **S4** EVエンジン / **S5** バケット分析 / **S6** 運用UI / **S7** 拡張
- **skill候補**（着手時に順次作成）: `keirin-scraper`（S1）, `keirin-feature`（S2）, `bucket-backtest`（S5）
- **subagent候補**（ユーザー指示時のみ起動）: S1のソース別スクレイパー、S2の特徴量カテゴリは相互独立で並行可

## コマンド
```bash
pip install -r requirements.txt
psql -f db/schema.sql              # スキーマ適用（DB作成後）
pytest tests/
```
