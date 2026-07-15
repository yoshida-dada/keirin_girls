"""展開特徴の「モデル入力10列」を学習・推論で**共通生成**する（train/inference skew防止）。

compare_tactics.py の検証と同一の構成:
  A(6列) = rider_tactics の絶対値をレース内相対化（value − レース内present平均, 欠損=0）
  B(4列) = race_dynamics のレース内変動列（既に相対量。欠損=0）
順序は TACTIC_NAMES（= A_NAMES + B_NAMES）で固定。学習(deploy)も推論(persist)も本関数を通す。
"""
from __future__ import annotations

from src.features.race_dynamics import dynamics_for_cars

# A: rider_tactics の絶対値キー → 相対化後の列名
A_KEYS = ["lead_index", "lead_index_sb", "sikake", "avg_last_lap",
          "escape_survival", "leg_change_rate"]
A_NAMES = ["t_lead_rel", "t_leadsb_rel", "t_sikake_rel", "t_lastlap_rel",
           "t_escape_rel", "t_legchg_rel"]
# B: race_dynamics のレース内変動列（レース定数 pace_*/lead_contest は投入しない）
B_NAMES = ["lead_margin", "sikake_rel", "escape_success", "last_lap_rel"]
# モデルに入れる10列（この順で連結する）
TACTIC_NAMES = A_NAMES + B_NAMES


def _rel(vals: list) -> list:
    """レース内相対化: present の平均を引く。欠損は0（=レース平均扱い）。compare_tactics と同型。"""
    present = [v for v in vals if v is not None]
    mean = sum(present) / len(present) if present else 0.0
    return [(v - mean) if v is not None else 0.0 for v in vals]


def tactic_columns(cars: list[int], tac_by_car: dict[int, dict]) -> dict[int, list]:
    """出走車 cars と各車の raw 展開特徴(tac_by_car) から、モデル入力10列を車番キーで返す。

    A = 各 A_KEY のレース内相対化、B = dynamics_for_cars（同一純関数）のレース内変動列。
    返り値: {car: [A(6) ... B(4)]}（TACTIC_NAMES 順）。学習・推論で同一。
    """
    # A: 相対化列（キーごとに全車ぶん相対化し、車番へ配る）
    a_cols = {c: [] for c in cars}
    for key in A_KEYS:
        rel = _rel([tac_by_car.get(c, {}).get(key) for c in cars])
        for c, v in zip(cars, rel):
            a_cols[c].append(v)
    # B: レース展開（dynamics_for_cars は学習バッチと同一関数）
    dyn = dynamics_for_cars(list(cars), [tac_by_car.get(c, {}) for c in cars])
    out: dict[int, list] = {}
    for c in cars:
        b = [(dyn[c].get(k) if dyn[c].get(k) is not None else 0.0) for k in B_NAMES]
        out[c] = a_cols[c] + b
    return out
