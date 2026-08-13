"""三連単の条件付き紐補正（PLの後段）。analyze_himo_conditional の知見を確率分布に反映する。

PL(独立)は「◎が勝つ時の2着＝○」を約8pt過大評価し、2着はもっと割れる（実測）。また◎の
並び直後のマーク(番手)は2着に来やすい。そこで PL 逐次選択の 2着/3着 の重みだけを補正する:
  ・1着 P(a)=s_a/Σs           … 素のPL（本命確率は較正済みなので触らない）
  ・2着 重み  w_b = s_b^(1/t2) … t2>1で平坦化（○への過集中を緩和）。a==◎ かつ b==◎の並び番手なら ×(1+mark)
  ・3着 重み  w_c = s_c^(1/t3) … t3>1で平坦化
各段で残り選手内に正規化するため、全210通りの和は必ず1（正しい確率分布）。

  from src.model.himo_adjust import corrected_trifecta_probs, combo_logprob, PL_PARAMS
"""
from __future__ import annotations

import math
from itertools import permutations

# PL と等価（補正なし）。検証のベースライン。
PL_PARAMS = {"t2": 1.0, "t3": 1.0, "mark": 0.0}
# 既定の補正。validate_himo_adjust の hold-out で確定（2026-07-17, 三連単log-loss −0.115・2着top3 +1.7pt）。
DEFAULT_PARAMS = {"t2": 1.5, "t3": 1.4, "mark": 0.25}


def marker_of(fav, narabi_pos, lines=None):
    """◎のマーク(番手)＝直後で追走する選手。無ければ None。

    lines（記者の並び予想のライン構成 [[車番...], ...]）があれば**それを正とする**。
    男子は複数ラインが1本の隊列として連結されるため、「隊列の直後」を番手とみなすと
    ラインの最後尾では**次ラインの先頭（＝敵）**を番手として拾ってしまう。

    実測（男子25,155R）:
      ・真の番手（同一ライン・直後）が勝者の2着に来る率 41.0%（n=20,006）
      ・勝者がライン最後尾/単騎のとき「隊列の直後」が2着に来る率 13.3%（n=3,948）
        ← ランダムの16.5%を**下回る**。ここへ加点すると補正が逆方向に働く
      ・隊列直後が真の番手と食い違うのは全選手の32.2%
    → ラインの最後尾・単騎には番手がいない。誰も代わりに立てず None を返す。

    lines が無い場合（ガールズ＝ラインの概念が無く line_id もほぼ空）は従来どおり
    隊列の直後を返す。ガールズ本番モデルはこの定義で学習済みで、推論だけ変えると
    train/inference skew になる。
    """
    if fav is None:
        return None
    if lines:
        for mem in lines:
            for i, car in enumerate(mem):
                if car == fav:
                    return mem[i + 1] if i + 1 < len(mem) else None
        return None                      # ライン構成に◎がいない（欠場等）
    if not narabi_pos:
        return None
    fp = narabi_pos.get(fav)
    if fp is None:
        return None
    for car, pos in narabi_pos.items():
        if car != fav and pos == fp + 1:
            return car
    return None


# 旧名。既存の呼び出しを壊さないために残す。
_marker_of = marker_of


def _pow(s, t):
    return s ** (1.0 / t) if s > 0 else 0.0


def combo_logprob(strengths: dict, narabi_pos: dict | None, order3, params: dict,
                  lines: list | None = None) -> float:
    """観測三連単 order3=(a,b,c) の補正後 log 確率（チューニング用・単一combo・軽量）。

    lines を渡すと番手をライン基準で判定する（男子）。marker_of の docstring を参照。
    """
    riders = [r for r in strengths if strengths[r] > 0]
    a, b, c = order3
    if a not in strengths or b not in strengths or c not in strengths:
        return -50.0
    fav = max(strengths, key=strengths.get)
    marker = marker_of(fav, narabi_pos, lines)
    t2, t3, mk = params["t2"], params["t3"], params["mark"]
    S = sum(strengths[r] for r in riders)
    if S <= 0 or strengths.get(a, 0) <= 0:
        return -50.0
    p1 = strengths[a] / S
    rem2 = [r for r in riders if r != a]
    w2 = {r: _pow(strengths[r], t2) for r in rem2}
    if a == fav and marker in w2:
        w2[marker] *= (1.0 + mk)
    Z2 = sum(w2.values())
    p2 = w2.get(b, 0.0) / Z2 if Z2 > 0 else 0.0
    rem3 = [r for r in rem2 if r != b]
    w3 = {r: _pow(strengths[r], t3) for r in rem3}
    Z3 = sum(w3.values())
    p3 = w3.get(c, 0.0) / Z3 if Z3 > 0 else 0.0
    p = p1 * p2 * p3
    return math.log(p) if p > 0 else -50.0


def corrected_trifecta_probs(strengths: dict, narabi_pos: dict | None = None,
                             params: dict | None = None,
                             lines: list | None = None) -> dict[tuple, float]:
    """補正後の三連単全210通り {(a,b,c): p}。params 既定は DEFAULT_PARAMS。

    lines（記者の並び予想のライン構成）を渡すと番手をライン基準で判定する（男子）。
    """
    p = params or DEFAULT_PARAMS
    t2, t3, mk = p["t2"], p["t3"], p["mark"]
    riders = [r for r in strengths if strengths[r] > 0]
    if len(riders) < 3:
        return {}
    fav = max(strengths, key=strengths.get)
    marker = marker_of(fav, narabi_pos, lines)
    S = sum(strengths[r] for r in riders)
    pow2 = {r: _pow(strengths[r], t2) for r in riders}
    pow3 = {r: _pow(strengths[r], t3) for r in riders}
    out = {}
    for a in riders:
        p1 = strengths[a] / S
        rem2 = [r for r in riders if r != a]
        w2 = {r: pow2[r] for r in rem2}
        if a == fav and marker in w2:
            w2[marker] *= (1.0 + mk)
        Z2 = sum(w2.values())
        if Z2 <= 0:
            continue
        for b in rem2:
            p2 = w2[b] / Z2
            rem3 = [r for r in rem2 if r != b]
            Z3 = sum(pow3[r] for r in rem3)
            if Z3 <= 0:
                continue
            base = p1 * p2 / Z3
            for c in rem3:
                out[(a, b, c)] = base * pow3[c]
    return out
