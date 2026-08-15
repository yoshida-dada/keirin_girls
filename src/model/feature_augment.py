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
from src.features.line_features import (LINE_KEYS, line_columns, class_level,
                                        LEGOH_KEYS, legoh_columns,
                                        SERI_KEYS, seri_columns, positions_with_seri)


def _line_ctx(db_path):
    """race_id ごとの {車番:(line_id,pos)} / {車番:得点} / 級班リスト を引く（男子ライン特徴用）。"""
    import sqlite3
    from collections import defaultdict
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    line_of = defaultdict(dict)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        line_of[rid][car] = (li, pi)
    scores, cls = defaultdict(dict), defaultdict(list)
    for rid, car, sc, cr in c.execute(
            "SELECT race_id,car_number,racing_score,class_rank FROM entries"):
        if sc:
            scores[rid][car] = sc
        if cr:
            cls[rid].append(cr)
    legs = defaultdict(dict)
    for rid, car, lg in c.execute("SELECT race_id,car_number,leg FROM narabi"):
        legs[rid][car] = lg
    # 競り（同じ位置を争うグループ）。DBの pos_in_line は生の連番のままなので、
    # ここで拾って positions_with_seri に渡し、**特徴を作る時点で**位置を補正する。
    seri = defaultdict(lambda: defaultdict(list))
    try:
        for rid, car, g in c.execute(
                "SELECT race_id,car_number,seri_group FROM narabi"
                " WHERE seri_group IS NOT NULL"):
            seri[rid][g].append(car)
    except Exception:
        pass                      # seri_group 列が無い旧DB
    c.close()
    return line_of, scores, cls, legs, {r: list(v.values()) for r, v in seri.items()}


def _venue_map(db_path) -> dict[str, str]:
    """race_id → venue_code。バンク交互作用はレースの開催場に依存するため引く。"""
    import sqlite3
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    out = {rid: v for rid, v in c.execute("SELECT race_id,venue_code FROM races")}
    c.close()
    return out


def _apply_seri(line_of: dict, seri: list | None) -> dict:
    """{車番:(line_id,pos)} に競り補正を掛ける。競りの選手は同じ pos を共有する。"""
    if not seri or not line_of:
        return line_of
    by_line: dict[int, list] = {}
    for car, (li, pi) in line_of.items():
        by_line.setdefault(li, []).append((pi, car))
    out = {}
    for li, v in by_line.items():
        order = [c for _, c in sorted(v)]
        gs = [g for g in seri if all(c in order for c in g)]
        for car, pos in positions_with_seri(order, gs).items():
            out[car] = (li, pos)
    return out


def augment_samples(samples: list, db_path, feature_names: list | None) -> list:
    """feature_names に応じて rel_elo / 展開列 / 並び予想列 / バンク交互作用列 を as-of 付与する。

    競り補正（ライン内位置の共有）と ln_seri 列は**必ずセット**で、モデルの feature_names に
    ln_seri があるかだけで決まる。片方だけ入る組み合わせを作らないための措置。
    位置補正だけを推論側に入れて列を持たないモデルに食わせると skew になる（2026-08-15 に
    実際にその状態を作ってしまい、検証で不採用になった時に気づいた）。
    """
    names = feature_names or []
    need_elo = "rel_elo" in names
    need_bank = any(n in names for n in BANK_KEYS)
    # バンク交互作用は展開特徴（逃げ残率・主導権指数）から作るので、展開の計算が要る
    need_tac = any(n in names for n in TACTIC_NAMES) or need_bank
    need_nb = any(n in names for n in NARABI_KEYS)
    need_line = any(n in names for n in LINE_KEYS)
    need_legoh = any(n in names for n in LEGOH_KEYS)
    need_seri = any(n in names for n in SERI_KEYS)
    if not (need_elo or need_tac or need_nb or need_line or need_legoh or need_seri):
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
    line_of = scores = cls = legs = seri = None
    if need_line or need_legoh or need_seri:
        line_of, scores, cls, legs, seri = _line_ctx(db_path)

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
        if need_line:
            cars = list(s.car_numbers)
            lo = line_of.get(s.race_id, {})
            if need_seri:            # 補正と ln_seri 列は必ずセット（上の docstring 参照）
                lo = _apply_seri(lo, (seri or {}).get(s.race_id))
            lcols = line_columns(cars, lo, scores.get(s.race_id, {}),
                                 class_level(cls.get(s.race_id, [])))   # 推論と同一関数
            lkeep = [(i, name) for i, name in enumerate(LINE_KEYS) if name in names]
            lmat = np.array([[lcols[c][i] for i, _ in lkeep] for c in cars], dtype=float)
            X = np.hstack([X, lmat])
            fn = fn + [name for _, name in lkeep]
        if need_legoh:                          # 脚質one-hot（推論と同一関数）
            cars = list(s.car_numbers)
            gcols = legoh_columns(cars, (legs or {}).get(s.race_id, {}))
            gkeep = [(i, name) for i, name in enumerate(LEGOH_KEYS) if name in names]
            gmat = np.array([[gcols[c][i] for i, _ in gkeep] for c in cars], dtype=float)
            X = np.hstack([X, gmat])
            fn = fn + [name for _, name in gkeep]
        if need_seri:                           # 競り参加フラグ（推論と同一関数）
            cars = list(s.car_numbers)
            scols = seri_columns(cars, (seri or {}).get(s.race_id))
            skeep = [(i, name) for i, name in enumerate(SERI_KEYS) if name in names]
            smat = np.array([[scols[c][i] for i, _ in skeep] for c in cars], dtype=float)
            X = np.hstack([X, smat])
            fn = fn + [name for _, name in skeep]
        # 最後に model.feature_names の並びへ揃える。ここが無いと「どの順で hstack したか」に
        # 暗黙依存し、呼び出し側が列を挟み直す羽目になる（実際 deploy_men.py がそうしていて、
        # analyze_dev_patterns から男子モデルを使った時に 31列 vs 39列 で落ちた）。
        if names and set(names).issubset(set(fn)):
            idx = [fn.index(n) for n in names]
            X = X[:, idx]
            fn = list(names)
        s2.X = X
        s2.feature_names = fn
        out.append(s2)
    return out
