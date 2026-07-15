"""選手×バンク長(周長)の適性特徴（as-of・リーク防止）。

同じ選手でも 333m(短走路) / 400m / 500m でコーナーの角度・直線長・required脚質が変わり、
「この選手が当該バンクを自分の平均より得意/苦手か」を能力値とは別に持てると仮説する。

compute_pre_race_bank(db_path, k_shrink):
    レースを **実施日(_race_date_from_id)→race_id 順** に処理し、選手(氏名)ごとに
    **バンク長別**の出走数・1着数・3着内数を累積する。各エントリの発走前(as-of)時点で
    当該バンクの適性を記録してから、そのレース結果で履歴を更新する（当該レースは混ざらない）。

shrinkage（経験ベイズ）:
    バンク別勝率を、その選手の**全バンク通算勝率**へ縮約する。
        bank_win_shrunk = (bank_wins + k*overall_win_rate) / (bank_starts + k)
    サンプルが薄い区分（500m≈446R / 333m≈916R と 400m≈4620R に偏在）でも過学習しないよう、
    経験の薄い選手×バンクは通算平均へ強く引き戻す。top3内率も同型。通算実績が皆無の選手は
    全体事前分布（7車立ての 1/7, 3/7）へ縮約する。

返り値: {(race_id, car_number): {
    "bank_win_shrunk", "bank_top3_shrunk", "bank_starts"(当該バンクのサポート数), "bank"}}
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

from src.features.rider_history import _race_date_from_id
from src.features.venue_meta import bank_length

# 通算実績ゼロの選手を縮約する全体事前分布（7車立ての一様prior）。
WIN_PRIOR = 1.0 / 7.0
TOP3_PRIOR = 3.0 / 7.0


def compute_pre_race_bank(db_path: str | Path, k_shrink: int = 20
                          ) -> dict[tuple[str, int], dict]:
    """各エントリ(race_id, car_number)の**発走前**バンク適性を返す。

    k_shrink … 経験ベイズの縮約強度（バンク別サンプルがこの本数のとき、実測と通算平均を
    半々で混ぜる）。500m/333mが薄いため既定20と強めに引き戻す。
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        venue_of = dict(conn.execute("SELECT race_id, venue_code FROM races").fetchall())
        ent_rows = conn.execute(
            "SELECT race_id, car_number, rider_name FROM entries").fetchall()
        res_rows = conn.execute(
            "SELECT race_id, car_number, position FROM results"
            " WHERE position IS NOT NULL").fetchall()
    finally:
        conn.close()

    # レース単位に entries / results をまとめる
    entries: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for rid, car, name in ent_rows:
        entries[rid].append((car, name))
    positions: dict[str, dict[int, int]] = defaultdict(dict)
    for rid, car, pos in res_rows:
        positions[rid][car] = pos

    # 実施日→race_id 順（race_date は初日固定バグのため _race_date_from_id を使う）
    order = sorted(venue_of.keys(),
                   key=lambda r: (_race_date_from_id(r) or "", r))

    # 通算（全バンク）とバンク別の累積カウンタ
    ov_starts: dict[str, int] = defaultdict(int)
    ov_wins: dict[str, int] = defaultdict(int)
    ov_top3: dict[str, int] = defaultdict(int)
    bk_starts: dict[tuple[str, int], int] = defaultdict(int)
    bk_wins: dict[tuple[str, int], int] = defaultdict(int)
    bk_top3: dict[tuple[str, int], int] = defaultdict(int)

    pre: dict[tuple[str, int], dict] = {}
    for rid in order:
        bank = bank_length(venue_of.get(rid, ""))
        ents = entries.get(rid, [])
        # --- 発走前(as-of) 値を記録 ---
        for car, name in ents:
            os_ = ov_starts[name]
            ow_rate = ov_wins[name] / os_ if os_ else WIN_PRIOR
            ot_rate = ov_top3[name] / os_ if os_ else TOP3_PRIOR
            if bank is None:                      # 未知バンク（実運用DBには無い想定）
                win_s, top3_s, support = ow_rate, ot_rate, 0
            else:
                bs = bk_starts[(name, bank)]
                win_s = (bk_wins[(name, bank)] + k_shrink * ow_rate) / (bs + k_shrink)
                top3_s = (bk_top3[(name, bank)] + k_shrink * ot_rate) / (bs + k_shrink)
                support = bs
            pre[(rid, car)] = {
                "bank_win_shrunk": win_s,
                "bank_top3_shrunk": top3_s,
                "bank_starts": support,
                "bank": bank if bank is not None else 0,
            }
        # --- 当該レースの結果で履歴更新（記録の後 = リーク無し） ---
        pos_map = positions.get(rid)
        if not pos_map:
            continue
        for car, name in ents:
            pos = pos_map.get(car)
            if pos is None:
                continue
            ov_starts[name] += 1
            if pos == 1:
                ov_wins[name] += 1
            if pos <= 3:
                ov_top3[name] += 1
            if bank is not None:
                bk_starts[(name, bank)] += 1
                if pos == 1:
                    bk_wins[(name, bank)] += 1
                if pos <= 3:
                    bk_top3[(name, bank)] += 1
    return pre
