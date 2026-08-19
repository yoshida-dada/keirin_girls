"""二車単(exacta)の推奨買い目選定（発走5分前オッズ運用）。

方針は `scripts/validate_exacta_selection.py` の as-of walk-forward 比較で選定した **P8**:
  「◎（勝率最大）を1着に固定し、相手（2着）はモデル上位の中から市場人気薄（オッズ高）優先で
   2〜3点」。8ポリシー中で回収率が最高(76.0%)・fold安定(SD±4.6)。純EVで両脚穴を追うと壊れる
  （48%）ため、必ず◎でアンカーする。三連単の展開買い目で的中の土台を作り、二車単で妙味を取る。

依然として回収率<100%（控除の壁）＝黒字ではない。的中頻度と妙味のバランスを取る参考買い目。
"""
from __future__ import annotations

from collections import defaultdict


def marginal_exacta_probs(trifecta_probs: dict) -> dict:
    """本番の三連単確率 {(a,b,c): p} を二車単 {(a,b): p}(1-2着へ周辺化) に落とす。"""
    ex: dict[tuple[int, int], float] = defaultdict(float)
    for (a, b, c), p in trifecta_probs.items():
        ex[(a, b)] += p
    return dict(ex)


def select_exacta(
    strengths: dict,
    trifecta_probs: dict,
    exacta_odds: dict,
    *,
    n_points: int = 3,
    pool: int = 4,
) -> dict | None:
    """P8選定。戻り値 dict（買い目・合成オッズ・合成的中率・EV）。オッズ不足時は None。

    strengths     : {車番: 強さ}（◎=argmax）
    trifecta_probs: 本番の三連単確率 {(a,b,c): p}（mix/himo）
    exacta_odds   : {(1着,2着): オッズ}（発走前の実オッズ、GambooBET 2shatan）
    n_points      : 買い目点数（2 or 3）。pool から人気薄優先で採る母数。
    """
    if not strengths or not exacta_odds:
        return None
    anchor = max(strengths, key=strengths.get)
    ex_prob = marginal_exacta_probs(trifecta_probs)
    # ◎を1着とする相手候補を、モデル確率の高い順に pool 件
    cand = sorted((b for (a, b) in ex_prob if a == anchor and (anchor, b) in exacta_odds),
                  key=lambda b: -ex_prob.get((anchor, b), 0))[:pool]
    if len(cand) < 2:
        return None
    # 母数の中から市場人気薄（オッズ高）優先で n_points 点
    picks = sorted(cand, key=lambda b: -exacta_odds[(anchor, b)])[:n_points]

    rows = []
    inv = 0.0
    hit = 0.0
    for b in picks:
        o = exacta_odds[(anchor, b)]
        p = ex_prob.get((anchor, b), 0.0)
        rows.append({
            "combo": f"{anchor}-{b}",
            "first": anchor, "second": b,
            "odds": round(o, 1),
            "prob": round(p, 5),
            "ev": round(p * o, 2),          # >1 = 市場が過小評価（1点あたり）
        })
        inv += 1.0 / o if o > 0 else 0.0
        hit += p
    rows.sort(key=lambda r: -r["ev"])       # 表示はEV降順
    return {
        "policy": "P8 ◎軸→相手人気薄",
        "anchor": anchor,
        "points": len(rows),
        "buys": rows,
        "synth_odds": round(1.0 / inv, 1) if inv > 0 else None,   # 合成オッズ=1/Σ(1/オッズ)
        "hit_prob": round(hit, 4),                                # 合成的中率=Σモデル確率
        "sum_ev": round(sum(r["prob"] * r["odds"] for r in rows), 2),
        "note": "◎で的中の土台、相手を人気薄に振る参考買い目。回収率<100%（黒字ではない）。",
    }
