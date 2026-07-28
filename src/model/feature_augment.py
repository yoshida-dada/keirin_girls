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
from src.features.rider_narabi import NARABI_KEYS, narabi_columns
from src.features.bank_features import BANK_KEYS, bank_columns


def _venue_map(db_path) -> dict[str, str]:
    """race_id → venue_code。バンク交互作用はレースの開催場に依存するため引く。"""
    import sqlite3
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    out = {rid: v for rid, v in c.execute("SELECT race_id,venue_code FROM races")}
    c.close()
    return out


def augment_samples(samples: list, db_path, feature_names: list | None) -> list:
    """feature_names に応じて rel_elo / 展開列 / 並び予想列 / バンク交互作用列 を as-of 付与する。"""
    names = feature_names or []
    need_elo = "rel_elo" in names
    need_bank = any(n in names for n in BANK_KEYS)
    # バンク交互作用は展開特徴（逃げ残率・主導権指数）から作るので、展開の計算が要る
    need_tac = any(n in names for n in TACTIC_NAMES) or need_bank
    need_nb = any(n in names for n in NARABI_KEYS)
    if not (need_elo or need_tac or need_nb):
        return samples

    pre_elo = compute_pre_race_elo(db_path) if need_elo else None
    venue = _venue_map(db_path) if need_bank else {}
    tactics = None
    if need_tac:
        from src.features.rider_tactics import compute_pre_race_tactics
        tactics = compute_pre_race_tactics(db_path)   # 各(race_id,car)の as-of raw 展開特徴
    narabi = None
    if need_nb:
        from src.features.rider_narabi import compute_narabi_features
        narabi = compute_narabi_features(db_path)      # 各(race_id,car)の並び予想 生特徴

    out = []
    for s in samples:
        s2 = copy.copy(s)
        X = s.X
        fn = list(s.feature_names)
        if need_elo:
            elos = np.array([pre_elo.get((s.race_id, c), DEFAULT_ELO) for c in s.car_numbers])
            X = np.hstack([X, (elos - elos.mean()).reshape(-1, 1)])
            fn = fn + ["rel_elo"]
        tac_by_car = None
        if need_tac:
            tac_by_car = {c: tactics.get((s.race_id, c), {}) for c in s.car_numbers}
            cols = tactic_columns(list(s.car_numbers), tac_by_car)   # 推論と同一関数
            # モデルが持つ展開列だけを追加（10列↔11列の移行でも shape 不整合を出さない）
            tkeep = [(i, name) for i, name in enumerate(TACTIC_NAMES) if name in names]
            if tkeep:
                mat = np.array([[cols[c][i] for i, _ in tkeep] for c in s.car_numbers], dtype=float)
                X = np.hstack([X, mat])
                fn = fn + [name for _, name in tkeep]
        if need_nb:
            nb_by_car = {c: narabi.get((s.race_id, c), {}) for c in s.car_numbers}
            ncols = narabi_columns(list(s.car_numbers), nb_by_car)   # 推論と同一関数(NARABI_KEYS順)
            # モデルが持つ並び列だけを追加（34特徴↔36特徴の移行でも不整合を出さない）
            keep = [(i, name) for i, name in enumerate(NARABI_KEYS) if name in names]
            nmat = np.array([[ncols[c][i] for i, _ in keep] for c in s.car_numbers], dtype=float)
            X = np.hstack([X, nmat])
            fn = fn + [name for _, name in keep]
        if need_bank:
            # 展開の相対量 × バンクの逃げ有利度。推論と同一関数（skew防止）
            bcols = bank_columns(list(s.car_numbers), tac_by_car or {}, venue.get(s.race_id))
            bkeep = [(i, name) for i, name in enumerate(BANK_KEYS) if name in names]
            bmat = np.array([[bcols[c][i] for i, _ in bkeep] for c in s.car_numbers], dtype=float)
            X = np.hstack([X, bmat])
            fn = fn + [name for _, name in bkeep]
        s2.X = X
        s2.feature_names = fn
        out.append(s2)
    return out
