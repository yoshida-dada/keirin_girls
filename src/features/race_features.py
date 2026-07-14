"""レース全体特徴量＋相対特徴量（S2、仕様書2.5/2.6/2.10）。

出走表(entries)だけで完結し、全てレース発走前に確定する値（as-os・リークなし）。
「選手Aがこの6人と走るときどれだけ有利か」を表すレース単位設計（2.10）の中核。
混戦度（race_type）とも材料を共有する。
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from src.collect.gamboo_racecard import Entry

# 脚質の型分類（逃げ型 = 先行して隊列を作る側）。マーク/差しは追走型。
ESCAPE_LEGS = {"逃"}
DASH_LEGS = {"捲"}


@dataclass
class RaceContext:
    """レース全体を表すスカラー特徴量（得点分散＝混戦度の土台, 2.5）。"""
    field_size: int
    mean_score: float
    std_score: float               # 得点標準偏差（大＝実力差あり/小＝混戦）
    top_score: float
    top_gap: float                 # トップ − 2位
    top2_minus_rest: float         # 上位2名平均 − 残りの平均（軸の堅さ, 2.10）
    escape_count: int              # 逃げ型人数（逃げ期待値/展開有利指数の材料 2.8）
    dash_count: int                # 捲り型人数
    age_mean: float | None = None
    age_std: float | None = None
    leg_composition: dict = field(default_factory=dict)  # {脚質: 人数}


@dataclass
class RiderRelative:
    """各選手のレース内相対特徴量（2.6・効果大）。"""
    car_number: int
    rel_score_mean: float          # 本人得点 − レース平均
    rel_score_max: float           # 本人得点 − レース最高（0が最強、負値）
    score_rank: int                # レース内得点順位（1=最上位）


def _scores(entries: list[Entry]) -> list[tuple[int, float]]:
    return [(e.car_number, e.racing_score) for e in entries
            if e.racing_score is not None and e.racing_score > 0]


def race_context(entries: list[Entry]) -> RaceContext | None:
    """レース全体特徴量を算出。得点を持つ選手が2名未満なら None。"""
    scored = _scores(entries)
    if len(scored) < 2:
        return None
    scores = sorted((s for _, s in scored), reverse=True)
    mean = statistics.mean(scores)
    std = statistics.pstdev(scores)
    top = scores[0]
    top_gap = scores[0] - scores[1]
    if len(scores) > 2:
        top2_minus_rest = statistics.mean(scores[:2]) - statistics.mean(scores[2:])
    else:
        top2_minus_rest = statistics.mean(scores[:2])

    legs = [e.leg_type for e in entries if e.leg_type]
    leg_comp: dict[str, int] = {}
    for lt in legs:
        leg_comp[lt] = leg_comp.get(lt, 0) + 1
    escape_count = sum(1 for lt in legs if lt in ESCAPE_LEGS)
    dash_count = sum(1 for lt in legs if lt in DASH_LEGS)

    ages = [e.age for e in entries if e.age]
    age_mean = statistics.mean(ages) if ages else None
    age_std = statistics.pstdev(ages) if len(ages) > 1 else None

    return RaceContext(
        field_size=len(entries), mean_score=round(mean, 3), std_score=round(std, 3),
        top_score=top, top_gap=round(top_gap, 3), top2_minus_rest=round(top2_minus_rest, 3),
        escape_count=escape_count, dash_count=dash_count,
        age_mean=age_mean, age_std=(round(age_std, 3) if age_std else None),
        leg_composition=leg_comp,
    )


def rider_relative(entries: list[Entry]) -> dict[int, RiderRelative]:
    """各選手のレース内相対得点・得点順位を返す（得点欠損の選手は除外）。"""
    scored = _scores(entries)
    if not scored:
        return {}
    mean = statistics.mean(s for _, s in scored)
    top = max(s for _, s in scored)
    ranked = sorted(scored, key=lambda x: -x[1])
    rank_of = {car: i + 1 for i, (car, _) in enumerate(ranked)}
    return {
        car: RiderRelative(
            car_number=car,
            rel_score_mean=round(s - mean, 3),
            rel_score_max=round(s - top, 3),
            score_rank=rank_of[car],
        )
        for car, s in scored
    }
