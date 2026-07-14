"""Plackett-Luce による三連単210通りの確率導出（S3。移植元: ../../競馬予想/analyzer/exotic.py）。

各選手の「強さ」 s_i（strength.py が出力）から逐次選択で着順確率を計算:
  P(i が1着)            = s_i / Σs
  P(i→j→k が1-2-3着)   = (s_i/S)·(s_j/(S-s_i))·(s_k/(S-s_i-s_j))

7車立て固定なので三連単は 7*6*5 = 210 通り。ここは純粋なモデル側（オッズは扱わない）。
出力（全210通りの確率）は S4 の EVエンジン（src/ev/ev_engine）へ model_probs として渡す。
"""
from __future__ import annotations

from itertools import permutations


def sequential_prob(strengths: dict, order: tuple) -> float:
    """order の並びで上位着（1着,2着,...）を占める確率（PL逐次選択）。"""
    remaining = sum(strengths.values())
    p = 1.0
    for r in order:
        s = strengths.get(r, 0.0)
        if remaining <= 0 or s <= 0:
            return 0.0
        p *= s / remaining
        remaining -= s
    return p


def trifecta_prob(strengths: dict, a, b, c) -> float:
    """三連単 a→b→c の的中確率。"""
    return sequential_prob(strengths, (a, b, c))


def all_trifecta_probs(strengths: dict) -> dict[tuple, float]:
    """出走選手の強さから三連単全210通りの確率 {(a,b,c): p} を返す（順序あり）。"""
    riders = list(strengths.keys())
    return {(a, b, c): sequential_prob(strengths, (a, b, c))
            for a, b, c in permutations(riders, 3)}


def normalize_strengths(win_probs: dict) -> dict:
    """正の値のみ取り出し Σ=1 に正規化。P(1着)=s_i を予測1着確率に一致させる。"""
    tot = sum(v for v in win_probs.values() if v and v > 0)
    if tot <= 0:
        return {}
    return {r: (p / tot) for r, p in win_probs.items() if p and p > 0}
