"""特徴量アセンブラ（S2の統合点）。1レース → 特徴量行列（車番 × 特徴量）を組む。

全て発走前確定値（as-os）のみ。出所は①出走表(entries)②出走表同梱の直近4ヶ月(RecentForm)
③それらから算出したレース全体/相対/派生特徴量。resultsのas-osローリング（バンク別・直近N走・
成長率）はデータ蓄積後に本アセンブラへ追加する（build に as_of_date 引数を後付け予定）。

出力は pandas DataFrame（index=車番, columns=特徴量）。PL線形／LightGBM 双方の入力に使う。
"""
from __future__ import annotations

import pandas as pd

from src.collect.gamboo_racecard import Entry, RecentForm
from src.features.race_features import race_context, rider_relative
from src.features.derived_features import derived_features

# 脚質フラグの対象（ガールズ）
_LEG_FLAGS = {"is_escape": "逃", "is_dash": "捲", "is_closing": "差", "is_mark": "マーク"}


def _rate(count, starts):
    if not starts or count is None:
        return None
    return count / starts


def build_features(entries: list[Entry], recent: dict[int, RecentForm] | None = None
                   ) -> pd.DataFrame:
    """1レースの特徴量行列を返す（index=車番）。recent 未指定なら直近由来はNaN。"""
    recent = recent or {}
    ctx = race_context(entries)
    rel = rider_relative(entries)
    der = derived_features(entries, recent)

    rows = []
    for e in entries:
        car = e.car_number
        f = recent.get(car)
        r = rel.get(car)
        d = der.get(car)
        row = {
            "car_number": car,
            # 出走表（土台 Tier1）
            "racing_score": e.racing_score,
            "gear_ratio": e.gear_ratio,
            "bracket_number": e.bracket_number,
            "age": e.age,
            # 相対（2.6）
            "rel_score_mean": r.rel_score_mean if r else None,
            "rel_score_max": r.rel_score_max if r else None,
            "score_rank": r.score_rank if r else None,
            # 直近4ヶ月（as-os）
            "win_rate": f.win_rate if f else None,
            "top2_rate": f.top2_rate if f else None,
            "top3_rate": f.top3_rate if f else None,
            "escape_rate": _rate(f.escape, f.starts) if f else None,
            "dash_rate": _rate(f.dash, f.starts) if f else None,
            "closing_rate": _rate(f.closing, f.starts) if f else None,
            "mark_rate": _rate(f.mark, f.starts) if f else None,
            "b_rate": _rate(f.b_count, f.starts) if f else None,
            "s_rate": _rate(f.s_count, f.starts) if f else None,
            "recent_starts": f.starts if f else None,
            # 派生（2.8）
            "escape_ev": d.escape_ev if d else None,
            "tenkai_advantage": d.tenkai_advantage if d else None,
            "stability": d.stability if d else None,
            "recent_avg_finish": d.recent_avg_finish if d else None,
            "ability_gap": d.ability_gap if d else None,
            # レース全体（2.5、全選手に同値をブロードキャスト）
            "race_mean_score": ctx.mean_score if ctx else None,
            "race_std_score": ctx.std_score if ctx else None,
            "race_top_gap": ctx.top_gap if ctx else None,
            "race_top2_minus_rest": ctx.top2_minus_rest if ctx else None,
            "race_escape_count": ctx.escape_count if ctx else None,
            "field_size": ctx.field_size if ctx else len(entries),
        }
        # 脚質フラグ
        for flag, leg in _LEG_FLAGS.items():
            row[flag] = int(e.leg_type == leg) if e.leg_type else 0
        rows.append(row)

    df = pd.DataFrame(rows).set_index("car_number").sort_index()
    return df


# 学習器へ渡す特徴量列（識別列・ラベルは含めない）。学習/推論で共有する。
FEATURE_COLUMNS = [
    "racing_score", "gear_ratio", "bracket_number", "age",
    "rel_score_mean", "rel_score_max", "score_rank",
    "win_rate", "top2_rate", "top3_rate",
    "escape_rate", "dash_rate", "closing_rate", "mark_rate", "b_rate", "s_rate",
    "escape_ev", "tenkai_advantage", "stability", "recent_avg_finish", "ability_gap",
    "race_mean_score", "race_std_score", "race_top_gap", "race_top2_minus_rest",
    "race_escape_count", "field_size",
    "is_escape", "is_dash", "is_closing", "is_mark",
]
