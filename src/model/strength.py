"""強さスコア → Plackett-Luce の入力（S3ベースライン、仕様書「最初の一歩#2」）。

出走表（gamboo_racecard.Entry）から各選手の「強さ」 s_i を作り、正規化して
plackett_luce.all_trifecta_probs に渡すと三連単210通りの確率が出る。

ベースラインは競走得点の指数スケール:  s_i = exp((score_i − mean) / TEMP)
  競走得点は選手の地力を最もよく表すTier1土台（2.1）。差を softmax 温度 TEMP でスケールする。
  TEMP を大きくすると横一線（混戦）、小さくすると本命偏重になる。データが貯まったら
  MLE/勾配ブースティングで係数を学習して置き換える（現状は competitive な既定値）。

ラインが無いガールズでは相対関係が効く（2.10）。ここでは競走得点の相対差のみを使い、脚質構成等の
レース全体特徴量は S2 で追加してこの関数を拡張する。
"""
from __future__ import annotations

import math
from statistics import mean

from src.collect.gamboo_racecard import Entry

DEFAULT_TEMP = 8.0   # 競走得点差の softmax 温度（点）。ガールズの得点レンジに合わせた既定値。


def strengths_from_entries(entries: list[Entry], temp: float = DEFAULT_TEMP) -> dict[int, float]:
    """出走表から {車番: 強さ}（Σ=1 正規化済み）を返す。競走得点が無い選手は除外。

    強さ s_i ∝ exp((score_i − 平均) / temp)。差だけが効くので平均は任意（数値安定のため引く）。
    """
    scored = [(e.car_number, e.racing_score) for e in entries
              if e.racing_score and e.racing_score > 0]
    if not scored:
        return {}
    avg = mean(s for _, s in scored)
    raw = {car: math.exp((s - avg) / temp) for car, s in scored}
    tot = sum(raw.values())
    return {car: v / tot for car, v in raw.items()}
