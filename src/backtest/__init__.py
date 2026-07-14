"""S5 バケット分析・バックテスト (Phase4, 本戦略の核)。

(レースタイプ × オッズ帯)バケット定義 → 実現ROI集計(haircut後オッズ) →
バケット別キャリブレーション(reliability/Brier, 課題B) → favorite-longshot bias実証 →
EV閾値チューニング → 狙い目マップ確定。狙い目マップができるまで実弾投入しない。
bucket-backtest skill に手順を切り出す予定。
"""
