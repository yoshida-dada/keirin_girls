"""学習サンプル(RaceSample)を本番モデルの feature_names に合わせて拡張する共通関数。

本番モデルは 拡張20 + rel_elo(+ 展開10) の構成。rel_elo や展開特徴を各所（build_predictions の
calibration/race_type_dist、accuracy_history、deploy スクリプト）でバラバラに付与すると
train/inference skew や shape 不整合の温床になるため、ここに一本化する。付与列は
`src/features/tactics_features.tactic_columns`（推論と同一関数）を通す＝skew防止。

  augment_samples(samples, db_path, feature_names) -> 拡張済み samples（Xとfeature_namesを更新）
順序: [元特徴 ... , rel_elo(あれば), 展開10列(あれば)]。model.feature_names の並びと一致させる。
"""
from __future__ import annotations

import copy

import numpy as np

from src.model.elo import compute_pre_race_elo, DEFAULT_ELO
from src.features.tactics_features import TACTIC_NAMES, tactic_columns


def augment_samples(samples: list, db_path, feature_names: list | None) -> list:
    """feature_names に応じて rel_elo / 展開10列 を as-of 付与した samples を返す。"""
    names = feature_names or []
    need_elo = "rel_elo" in names
    need_tac = any(n in names for n in TACTIC_NAMES)
    if not (need_elo or need_tac):
        return samples

    pre_elo = compute_pre_race_elo(db_path) if need_elo else None
    tactics = None
    if need_tac:
        from src.features.rider_tactics import compute_pre_race_tactics
        tactics = compute_pre_race_tactics(db_path)   # 各(race_id,car)の as-of raw 展開特徴

    out = []
    for s in samples:
        s2 = copy.copy(s)
        X = s.X
        fn = list(s.feature_names)
        if need_elo:
            elos = np.array([pre_elo.get((s.race_id, c), DEFAULT_ELO) for c in s.car_numbers])
            X = np.hstack([X, (elos - elos.mean()).reshape(-1, 1)])
            fn = fn + ["rel_elo"]
        if need_tac:
            tac_by_car = {c: tactics.get((s.race_id, c), {}) for c in s.car_numbers}
            cols = tactic_columns(list(s.car_numbers), tac_by_car)   # 推論と同一関数
            mat = np.array([cols[c] for c in s.car_numbers], dtype=float)
            X = np.hstack([X, mat])
            fn = fn + list(TACTIC_NAMES)
        s2.X = X
        s2.feature_names = fn
        out.append(s2)
    return out
