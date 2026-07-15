"""バケット分析・バックテスト（Phase4・本戦略の核）。

学習済みモデルの確率と確定オッズ・実払戻から、「レースタイプ(軸堅/標準/混戦) × オッズ帯」の
バケットごとに、EV閾値超えを機械的に全買いしたときの**実現ROI**と**キャリブレーション**を集計する。

  ROI = 回収額 / 投票額（100%超なら実運用候補）。回収は的中買い目の実払戻(payouts_trifecta)。
  キャリブレーションは各買い目の (モデル確率, 的中0/1) から Brier/ECE（課題B）。

★リーク防止: モデルは学習期間で訓練し、**検証期間（未来）のレースだけ**を集計する（out-of-sample）。
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.features.assembler import build_features
from src.model.training_data import PL_FEATURES, _entries_of, _recent_of
from src.model.plackett_luce import all_trifecta_probs
from src.model.race_type import classify_race
from src.ev.ev_engine import odds_bucket_label
from src.backtest.calibration import brier_score, expected_calibration_error
from src.features.tactics_features import TACTIC_NAMES, tactic_columns

_STAKE = 100  # 1点あたり投票額（円）


@dataclass
class ComboRecord:
    race_id: str
    race_type: str
    odds_bucket: str
    combo: tuple
    model_prob: float
    odds: float
    ev: float
    is_win: bool
    payout: int          # 的中時の払戻(100円あたり)。非的中は0


def _combo(s: str) -> tuple:
    return tuple(int(x) for x in s.split("-"))


def build_records(db_path: str | Path, model, race_ids: list[str],
                  haircut: float = 1.0) -> list[ComboRecord]:
    """検証期間の各レース×各買い目について ComboRecord を作る（モデルは学習済み）。

    モデルの feature_names に応じて rel_elo / 展開10列 を **as-of（当該レースを含まない履歴）**で
    付与する（compute_pre_race_elo / compute_pre_race_tactics のバッチ値。学習と同一の tactic_columns
    を通すので skew 無し）。検証レースは out-of-sample、特徴はas-of＝リーク無し。
    """
    feats = model.feature_names or PL_FEATURES       # モデルの学習特徴に追従
    need_elo = "rel_elo" in feats
    need_tac = any(n in feats for n in TACTIC_NAMES)
    pre_elo = tactics = None
    if need_elo:
        from src.model.elo import compute_pre_race_elo, DEFAULT_ELO
        pre_elo = compute_pre_race_elo(db_path)
    else:
        DEFAULT_ELO = 1500.0
    if need_tac:
        from src.features.rider_tactics import compute_pre_race_tactics
        tactics = compute_pre_race_tactics(db_path)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    records: list[ComboRecord] = []
    try:
        for race_id in race_ids:
            odds = {_combo(c): o for c, o in conn.execute(
                "SELECT combo, odds FROM odds_final_trifecta WHERE race_id=?", (race_id,))}
            win = conn.execute(
                "SELECT combo, payout FROM payouts_trifecta WHERE race_id=?", (race_id,)).fetchone()
            if not odds or not win:
                continue
            win_combo, win_payout = _combo(win[0]), int(win[1])
            entries = _entries_of(conn, race_id)
            recent = _recent_of(conn, race_id)
            df = build_features(entries, recent)
            if need_elo:                              # レース内相対Elo（as-of）
                elos = np.array([pre_elo.get((race_id, c), DEFAULT_ELO) for c in df.index])
                df["rel_elo"] = elos - elos.mean()
            if need_tac:                              # 展開10列（as-of・学習と同一関数）
                tac_by_car = {c: tactics.get((race_id, c), {}) for c in df.index}
                cols = tactic_columns(list(df.index), tac_by_car)
                for i, name in enumerate(TACTIC_NAMES):
                    df[name] = [cols[c][i] for c in df.index]
            if df[feats].isna().any().any():
                continue
            cars = list(df.index)
            X = df.loc[cars, feats].to_numpy(dtype=float)
            strengths = model.strengths(X, cars)                # {車番: P(1着)}
            probs = all_trifecta_probs(strengths)               # {combo: p}
            rtype = classify_race(strengths).label
            for combo, o in odds.items():
                p = probs.get(combo)
                if p is None or o <= 0:
                    continue
                is_win = combo == win_combo
                records.append(ComboRecord(
                    race_id=race_id, race_type=rtype,
                    odds_bucket=odds_bucket_label(o), combo=combo,
                    model_prob=p, odds=o, ev=p * o * haircut, is_win=is_win,
                    payout=win_payout if is_win else 0,
                ))
    finally:
        conn.close()
    return records


def bucket_roi(records: list[ComboRecord], ev_threshold: float,
               min_prob: float = 0.0, max_odds: float | None = None) -> dict[tuple, dict]:
    """(レースタイプ, オッズ帯) バケット別に、EV閾値超えを全買いした実現ROIと較正を返す。

    戻り値: {(race_type, odds_bucket): {n_bets, n_hits, stake, ret, roi, brier, ece, n_all}}。
    """
    # まず全買い目でバケット別の較正材料を貯め、購入対象は別に集計
    agg: dict[tuple, dict] = {}
    for r in records:
        key = (r.race_type, r.odds_bucket)
        b = agg.setdefault(key, {"pairs": [], "n_bets": 0, "n_hits": 0,
                                 "stake": 0, "ret": 0})
        b["pairs"].append((r.model_prob, 1 if r.is_win else 0))
        bought = r.ev >= ev_threshold and r.model_prob >= min_prob and \
            (max_odds is None or r.odds <= max_odds)
        if bought:
            b["n_bets"] += 1
            b["stake"] += _STAKE
            if r.is_win:
                b["n_hits"] += 1
                b["ret"] += r.payout            # 100円あたり払戻
    out: dict[tuple, dict] = {}
    for key, b in agg.items():
        roi = (b["ret"] / b["stake"]) if b["stake"] else None
        out[key] = {
            "n_all": len(b["pairs"]), "n_bets": b["n_bets"], "n_hits": b["n_hits"],
            "stake": b["stake"], "ret": b["ret"],
            "roi": round(roi, 4) if roi is not None else None,
            "brier": brier_score(b["pairs"]),
            "ece": expected_calibration_error(b["pairs"]),
        }
    return out


def odds_bucket_roi(records: list[ComboRecord], ev_threshold: float = 0.0) -> dict[str, dict]:
    """オッズ帯だけで集計（favorite-longshot bias の確認用）。ev_threshold=0で全買い目。"""
    agg: dict[str, dict] = {}
    for r in records:
        b = agg.setdefault(r.odds_bucket, {"n": 0, "hits": 0, "stake": 0, "ret": 0})
        if r.ev >= ev_threshold:
            b["n"] += 1
            b["stake"] += _STAKE
            if r.is_win:
                b["hits"] += 1
                b["ret"] += r.payout
    return {k: {**v, "roi": round(v["ret"] / v["stake"], 4) if v["stake"] else None}
            for k, v in sorted(agg.items())}
