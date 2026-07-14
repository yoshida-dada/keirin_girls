"""市場情報の処理（S4。移植元: ../../競馬予想/analyzer/market.py）。

三連単オッズから控除率を除いた市場implied確率を出し、モデル確率とlogit(ログ線形)ブレンドする。
  q_combo   = (1/odds_combo) / Σ(1/odds)          # 控除率25%を除いた市場確率（Σ=1）
  p_blend  ∝ p_model^α · q_market^(1−α)            # ログ線形プール、その後Σ=1に正規化

α はモデル重み（0=市場そのもの / 1=モデルそのもの）。ログ線形プールは
logit を線形結合して softmax する操作の多クラス一般化・正準形。
オッズを持たない買い目はブレンド対象から除外（q が定義できないため）。
"""
from __future__ import annotations

import math

_EPS = 1e-9


def implied_trifecta_probs(odds: dict) -> dict:
    """三連単オッズ {(a,b,c): odds} から控除率を除いた市場implied確率 {(a,b,c): q}(Σ=1) を返す。

    これが仕様書S4の「オッズ逆算確率（控除率25%を除去）」。全210点のオッズを渡すことで
    Σ(1/odds)≈1/(1−控除率) となり、正規化で控除率が除去される（課題A）。
    """
    inv = {k: (1.0 / o) for k, o in odds.items() if o and o > 0}
    tot = sum(inv.values())
    if tot <= 0:
        return {}
    return {k: v / tot for k, v in inv.items()}


def blend_loglinear(model: dict, market: dict, alpha: float) -> dict:
    """ログ線形プール。model・market ともに正の確率を持つ買い目のみでブレンドし Σ=1 に正規化。

    alpha はモデル重み [0,1]。market が空（オッズ未取得）なら model をそのまま正規化して返す。
    """
    if not market:
        tot = sum(v for v in model.values() if v and v > 0)
        return {k: v / tot for k, v in model.items() if v and v > 0} if tot > 0 else {}
    a = max(0.0, min(1.0, alpha))
    raw = {}
    for k, q in market.items():
        p = model.get(k, 0.0)
        if p is None or p <= 0 or q <= 0:
            continue
        raw[k] = math.exp(a * math.log(max(p, _EPS)) + (1.0 - a) * math.log(max(q, _EPS)))
    tot = sum(raw.values())
    if tot <= 0:
        return {}
    return {k: v / tot for k, v in raw.items()}
