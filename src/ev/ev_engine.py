"""三連単EVエンジン（S4。移植元: ../../競馬予想/betting/ev_table.py）。

モデル確率（S3のPlackett-Luce出力）と締切前実オッズ（全210点）から各買い目のEVを算出し、
「対象バケット内 × EV閾値 × 点数上限 × 確率下限」で購入対象を選ぶ（仕様書S4）。

  EV(gross) = 的中確率 × 実オッズ × haircut係数   （1.0が損益分岐。haircut=課題D）
  EV(net)   = EV(gross) − 1                        （期待収益率）

★前提: EVが正になるのは「モデル確率が実オッズより正確なとき」だけ。黒字は保証しない。
Phase4のバケットROI検証で100%超を確認するまで実弾投入しない（狙い目マップ）。
"""
from __future__ import annotations

from dataclasses import dataclass

from config.settings import EV_GUARD, ODDS_BUCKET_EDGES


@dataclass
class EVRow:
    combo: tuple            # 三連単の車番タプル (a,b,c)（着順どおり）
    prob: float             # EV算出に使った確率（shrink後）
    model_prob: float       # モデルの生確率（shrink前）
    market_prob: float      # 市場implied確率（控除率除去済み）
    odds: float             # 実オッズ（表示オッズ。EVにのみhaircutを掛ける）
    odds_bucket: str        # オッズ帯ラベル（課題A/Phase4）
    ev_gross: float         # 確率×オッズ×haircut（1.0が分岐）
    ev_net: float           # ev_gross − 1
    guarded_out: bool = False  # ガード/対象外バケットで購入対象から外れたか


def odds_bucket_label(odds: float, edges: list[float] = ODDS_BUCKET_EDGES) -> str:
    """オッズ値をバケットラベルへ。例 edges=[10,30] → "0-10" / "10-30" / "30+"。"""
    prev = 0
    for e in edges:
        if odds < e:
            return f"{prev}-{e}"
        prev = e
    return f"{prev}+"


def _haircut_for(bucket: str, haircut) -> float:
    """haircut がスカラーならそのまま、dict ならバケット別係数（無ければ1.0）を返す。"""
    if isinstance(haircut, dict):
        return haircut.get(bucket, 1.0)
    return haircut if haircut else 1.0


def build_trifecta_ev_table(
    model_probs: dict[tuple, float],
    odds: dict[tuple, float],
    ev_threshold: float = 1.10,
    guards: dict | None = EV_GUARD,
    odds_haircut=1.0,                       # float または {bucket_label: 係数}
    allowed_odds_buckets: set[str] | None = None,  # 対象オッズ帯（None=全帯）
    market_probs: dict[tuple, float] | None = None,  # 控除率除去済み（None=1/oddsから算出）
) -> dict:
    """三連単全点のEVを算出し購入対象を選ぶ。

    guards（None で無効）:
      - shrink_to_market: モデル確率を市場implied確率へ縮小し尾部の過信を抑える
      - min_prob: これ未満の確率は「ノイズ穴」として購入対象外
      - max_odds: これ超の高配当は購入対象外
    allowed_odds_buckets: 対象オッズ帯の集合（Phase4の狙い目マップで指定）。
    戻り値: {"all": [EVRow...降順], "buy": [EVRow...降順], "by_bucket": {bucket: {...}}}
    """
    g = guards or {}
    alpha = g.get("shrink_to_market", 0.0)
    min_prob = g.get("min_prob")
    max_odds = g.get("max_odds")

    if market_probs is None:
        # 控除率除去済み市場確率を 1/odds から算出（全210点を渡す前提）
        inv = {k: 1.0 / o for k, o in odds.items() if o and o > 0}
        tot = sum(inv.values())
        market_probs = {k: v / tot for k, v in inv.items()} if tot > 0 else {}

    all_rows: list[EVRow] = []
    by_bucket: dict[str, dict] = {}

    for combo, o in odds.items():
        if not o or o <= 0:
            continue
        p_model = model_probs.get(combo)
        if p_model is None or p_model <= 0:
            continue
        q = market_probs.get(combo, 0.0)
        # 尾部ガード: モデル確率を市場へ縮小（alpha=0で無効）
        p = (1.0 - alpha) * p_model + alpha * q
        bucket = odds_bucket_label(o)
        ev = p * o * _haircut_for(bucket, odds_haircut)

        guarded = False
        if min_prob is not None and p < min_prob:
            guarded = True
        if max_odds is not None and o > max_odds:
            guarded = True
        if allowed_odds_buckets is not None and bucket not in allowed_odds_buckets:
            guarded = True

        row = EVRow(
            combo=combo, prob=round(p, 8), model_prob=round(p_model, 8),
            market_prob=round(q, 8), odds=o, odds_bucket=bucket,
            ev_gross=round(ev, 4), ev_net=round(ev - 1.0, 4), guarded_out=guarded,
        )
        all_rows.append(row)
        b = by_bucket.setdefault(bucket, {"n_candidates": 0, "n_buy": 0, "max_ev": None})
        b["n_candidates"] += 1
        if b["max_ev"] is None or ev > b["max_ev"]:
            b["max_ev"] = round(ev, 3)

    all_rows.sort(key=lambda r: r.ev_gross, reverse=True)
    buy = [r for r in all_rows if (not r.guarded_out) and r.ev_gross >= ev_threshold]
    for r in buy:
        by_bucket[r.odds_bucket]["n_buy"] += 1

    return {"all": all_rows, "buy": buy, "by_bucket": by_bucket}


def format_combo(combo: tuple) -> str:
    """三連単タプルを表示用に（着順は→区切り）。例 (3,1,5) → "3→1→5"。"""
    return "→".join(str(x) for x in combo)
