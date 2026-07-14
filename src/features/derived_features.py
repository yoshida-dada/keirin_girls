"""派生特徴量（S2、仕様書2.8「効果大」）。

出走表同梱の直近4ヶ月成績（RecentForm）＋レース全体特徴量（RaceContext）から、人間が暗黙に
考える要素を数値化する。全て発走前確定値のみ（as-os・リークなし）。

いま算出するもの:
  逃げ期待値      = 逃げ率 × レース内逃げ型人数          （2.8）
  展開有利指数    = 差し率 × レース内逃げ型人数          （2.8）
  安定指数        = 着順分布の標準偏差（小＝安定）        （2.8）
  好調度(recent)  = 直近4ヶ月の平均着順                   （好調指数の直近側。長期差分は後述）
  能力差          = 本人得点 − レース平均得点             （2.8）

保留（自前resultsのas-osローリング蓄積後 = docs/design_s2_features.md）:
  バンク適性指数（当該会場勝率−全体勝率）／成長率（競走得点の推移）／
  好調指数の長期差分（直近平均着順 − 年間平均着順）。
"""
from __future__ import annotations

from dataclasses import dataclass

from src.collect.gamboo_racecard import Entry, RecentForm
from src.features.race_features import race_context, rider_relative


@dataclass
class DerivedFeatures:
    car_number: int
    escape_ev: float | None = None          # 逃げ期待値
    tenkai_advantage: float | None = None   # 展開有利指数
    stability: float | None = None          # 安定指数（着順std, 小=安定）
    recent_avg_finish: float | None = None  # 直近4ヶ月の平均着順
    ability_gap: float | None = None        # 本人得点 − レース平均


def _finish_distribution(f: RecentForm, field_size: int) -> list[tuple[float, int]]:
    """着順分布 [(着順代表値, 回数)]。着外は 4..N の中央値で代表する。"""
    out_pos = (4 + field_size) / 2.0
    return [(1.0, f.first or 0), (2.0, f.second or 0), (3.0, f.third or 0), (out_pos, f.out or 0)]


def _weighted_mean_std(dist: list[tuple[float, int]]) -> tuple[float | None, float | None]:
    n = sum(w for _, w in dist)
    if n == 0:
        return None, None
    mean = sum(p * w for p, w in dist) / n
    var = sum(w * (p - mean) ** 2 for p, w in dist) / n
    return mean, var ** 0.5


def derived_features(entries: list[Entry], recent: dict[int, RecentForm]
                     ) -> dict[int, DerivedFeatures]:
    """各選手の派生特徴量を {車番: DerivedFeatures} で返す。"""
    ctx = race_context(entries)
    rel = rider_relative(entries)
    if ctx is None:
        return {}
    escape_count = ctx.escape_count
    field_size = ctx.field_size

    out: dict[int, DerivedFeatures] = {}
    for e in entries:
        car = e.car_number
        d = DerivedFeatures(car_number=car)
        if car in rel:
            d.ability_gap = rel[car].rel_score_mean
        f = recent.get(car)
        if f and f.starts:
            starts = f.starts
            escape_rate = (f.escape or 0) / starts
            closing_rate = (f.closing or 0) / starts
            d.escape_ev = round(escape_rate * escape_count, 4)
            d.tenkai_advantage = round(closing_rate * escape_count, 4)
            mean, std = _weighted_mean_std(_finish_distribution(f, field_size))
            d.recent_avg_finish = round(mean, 3) if mean is not None else None
            d.stability = round(std, 3) if std is not None else None
        out[car] = d
    return out
