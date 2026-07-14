# ガールズケイリン 期待値予測AI（keirin_girls）

女子競輪（L級・7車立て）の3連単を対象にした**着順予測AI**。出走表・レース結果・確定オッズを収集し、
Plackett-Luce / LightGBM で各選手の1着確率と三連単210通りの確率、レースタイプ（軸堅／標準／混戦）を推定する。

- **ダッシュボード（GitHub Pages）**: 本日のレース予測・レースタイプ分布・キャリブレーションを表示。
- **予測の性質**: これは**着順予測の確率**を出すツールであり、**馬券の推奨・黒字保証ではない**。
  検証（walk-forward）の結果、現在の特徴量では控除率25%を超える収益エッジは確認できていないため、
  買い目・EVは提示していない（研究段階）。

## 構成
```
src/collect/   データ収集（GambooBET等のスクレイパー）
src/features/  特徴量（as-of厳守・リーク防止）
src/model/     Plackett-Luce線形 / LightGBM / レースタイプ分類
src/ev/        EVエンジン（控除率除去・logit blend・Kelly。研究用）
src/backtest/  バケット分析・キャリブレーション（Phase4）
api/           FastAPI（予測配信）
dashboard/     PWAダッシュボード（GitHub Pagesで公開）
scripts/       収集・学習・予測・data.json生成の実行スクリプト
```

## モデルの精度（時系列検証, out-of-sample）
- 1着的中率 約67%（7車） / Brier 0.068 / ECE 0.030（較正良好）
- 特徴量: 競走得点・相対得点・直近4ヶ月成績（勝率・脚質率・B率）・ギヤ 等

## ダッシュボードの更新
```bash
python scripts/build_predictions.py --predict   # dashboard/data.json を再生成
git commit -am "update predictions" && git push  # push で Pages 自動デプロイ
```

## 免責
個人利用・研究目的。予測精度は保証しない。馬券購入は自己責任で。
