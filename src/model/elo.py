"""Eloレーティング特徴量（②・ばんえいAIに倣った能力値）。

レース結果を時系列に処理し、各選手のEloを更新する。各レースの特徴量には**発走前(as-of)のElo**
を使う（リーク防止）。競走得点は更新が遅く直近の調子を反映しにくいため、Eloで相対力の変化を補う。

多人数レースはペアワイズで更新: レース内の全ペア(i,j)について、i が j より上位なら i の勝ち。
  期待勝率 E_ij = 1 / (1 + 10^((R_j - R_i)/400))
  R_i += (K / (N-1)) * Σ_j (S_ij - E_ij)      # S_ij=1(iが上位)/0
選手は登録番号が無いため氏名で同定する（ガールズは母数が小さく実用上十分）。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_ELO = 1500.0
DEFAULT_K = 24.0


def compute_pre_race_elo(db_path: str | Path, k: float = DEFAULT_K
                         ) -> dict[tuple[str, int], float]:
    """各エントリ(race_id, car_number)の**発走前**Eloを返す。レースを日付順に処理して更新する。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        races = conn.execute(
            "SELECT race_id FROM races ORDER BY race_date, race_id").fetchall()
        elo: dict[str, float] = {}
        pre: dict[tuple[str, int], float] = {}
        for (race_id,) in races:
            ents = conn.execute(
                "SELECT car_number, rider_name FROM entries WHERE race_id=?",
                (race_id,)).fetchall()
            car_name = {c: n for c, n in ents}
            positions = dict(conn.execute(
                "SELECT car_number, position FROM results"
                " WHERE race_id=? AND position IS NOT NULL", (race_id,)).fetchall())
            # 発走前Eloを記録（結果の有無に関わらず特徴量として使える）
            for c, name in car_name.items():
                pre[(race_id, c)] = elo.get(name, DEFAULT_ELO)
            if len(positions) < 2:
                continue
            cars = [c for c in positions if c in car_name]
            n = len(cars)
            if n < 2:
                continue
            delta = {c: 0.0 for c in cars}
            for i in cars:
                ri = elo.get(car_name[i], DEFAULT_ELO)
                for j in cars:
                    if i == j:
                        continue
                    rj = elo.get(car_name[j], DEFAULT_ELO)
                    e = 1.0 / (1.0 + 10 ** ((rj - ri) / 400.0))
                    s = 1.0 if positions[i] < positions[j] else 0.0
                    delta[i] += (k / (n - 1)) * (s - e)
            for c in cars:
                name = car_name[c]
                elo[name] = elo.get(name, DEFAULT_ELO) + delta[c]
        return pre
    finally:
        conn.close()


def final_elo_state(db_path: str | Path, k: float = DEFAULT_K) -> dict[str, float]:
    """全履歴処理後の最終Elo {氏名: Elo}（ライブ予測で選手の現在Eloを引くのに使う）。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        races = conn.execute("SELECT race_id FROM races ORDER BY race_date, race_id").fetchall()
        elo: dict[str, float] = {}
        for (race_id,) in races:
            car_name = dict(conn.execute(
                "SELECT car_number, rider_name FROM entries WHERE race_id=?", (race_id,)).fetchall())
            positions = dict(conn.execute(
                "SELECT car_number, position FROM results"
                " WHERE race_id=? AND position IS NOT NULL", (race_id,)).fetchall())
            cars = [c for c in positions if c in car_name]
            if len(cars) < 2:
                continue
            delta = {c: 0.0 for c in cars}
            for i in cars:
                ri = elo.get(car_name[i], DEFAULT_ELO)
                for j in cars:
                    if i == j:
                        continue
                    rj = elo.get(car_name[j], DEFAULT_ELO)
                    e = 1.0 / (1.0 + 10 ** ((rj - ri) / 400.0))
                    s = 1.0 if positions[i] < positions[j] else 0.0
                    delta[i] += (k / (len(cars) - 1)) * (s - e)
            for c in cars:
                elo[car_name[c]] = elo.get(car_name[c], DEFAULT_ELO) + delta[c]
        return elo
    finally:
        conn.close()
