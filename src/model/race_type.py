"""レースタイプ分類（S3、課題C）。確率分布の形状から 軸堅／標準／混戦 を定量判定する。

仕様書の候補指標を実装:
  - top1_win_prob : トップ選手の1着確率（大きいほど軸堅）
  - top2_gap      : 上位2名の1着確率合計 −（残りの平均×2）的な優位。ここでは上位2名合計を使用
  - entropy_norm  : 1着確率分布のエントロピー / log(N)（0=一強, 1=完全横一線）

分類はエントロピー正規化値のしきい値による（既定値。EVゾーンを分離するしきい値は Phase4 で検証）。
7車立て前提だが頭数可変でも動く（log(N)で正規化するため）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# レースタイプのラベル
JIKU = "軸堅"      # favorite-solid
STANDARD = "標準"
CHAOS = "混戦"      # muddled

# entropy_norm のしきい値。tune_race_type.py の分析(2026-07-15, 3年5713レース)で調整。
# 旧(0.80,0.93)は軸堅81%/混戦1.9%と偏り3タイプが分離できず。新値で軸堅24%/標準54%/混戦22%と
# バランスし、的中率も 軸堅88%>標準65%>混戦44% と明確に分離する。
DEFAULT_ENTROPY_EDGES = (0.50, 0.75)   # <0.50=軸堅 / 0.50–0.75=標準 / >0.75=混戦


@dataclass
class RaceType:
    label: str
    top1_win_prob: float
    top2_win_prob: float      # 上位2名の1着確率合計
    entropy_norm: float       # 0..1


def _entropy_norm(win_probs: dict) -> float:
    ps = [p for p in win_probs.values() if p and p > 0]
    n = len(ps)
    if n <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in ps)
    return h / math.log(n)


def classify_race(win_probs: dict, edges: tuple[float, float] = DEFAULT_ENTROPY_EDGES) -> RaceType:
    """1着確率分布（Σ=1想定）から レースタイプを判定する。

    win_probs が空なら STANDARD を返す（判定不能）。
    """
    if not win_probs:
        return RaceType(STANDARD, 0.0, 0.0, 0.0)
    ranked = sorted((p for p in win_probs.values() if p and p > 0), reverse=True)
    top1 = ranked[0] if ranked else 0.0
    top2 = sum(ranked[:2])
    ent = _entropy_norm(win_probs)
    lo, hi = edges
    if ent < lo:
        label = JIKU
    elif ent > hi:
        label = CHAOS
    else:
        label = STANDARD
    return RaceType(label=label, top1_win_prob=round(top1, 6),
                    top2_win_prob=round(top2, 6), entropy_norm=round(ent, 6))
