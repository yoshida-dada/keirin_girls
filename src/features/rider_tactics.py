"""展開シミュレーション特徴量（S2拡張・as-of＝リーク無し）。

「誰が主導権を握り、どこで仕掛け、逃げ切れるか」という人間の展開読みを数値化する。
2系統のソースから作る:

  A) recent_form（出走表同梱の直近4ヶ月成績 = 発走前に確定済み・元々as-of）
     → 主導権指数(lead_index / lead_index_sb) と 仕掛け距離指数(sikake)。
        純関数 `tactics_from_recent_form()` で 1行から算出。推論時は
        parse_recent_form() の RecentForm からも同じ関数で作れる。

  B) results履歴（自前で実施日順にas-of集計。当該レースは含めない＝「記録→更新」）
     → as-of平均上がり(avg_last_lap) / 逃げ残率(escape_survival) / 脚質変化率(leg_change_rate)。

リーク防止:
  * recent_form は元々「発走前に確定していた集計値」なので当該レースを含まない（安全）。
  * results ベースは elo.compute_pre_race_elo と同じ作法で、各レースの特徴を**発走前に記録**して
    から、そのレースの結果で履歴を更新する（当該レース自身は集計に混ざらない）。
  * 実施日は races.race_date（開催初日固定バグ）ではなく rider_history._race_date_from_id で復元。

欠損の扱い:
  * recent_form 無し、または決まり手総数 n=0 → lead_index/sikake は None。
  * lead_index_sb は starts=0（着順総数0）で None。
  * results履歴が無い新人 → avg_last_lap は None。escape_survival は事前分布(縮約先)へ寄る値を返す。
  * kimarite は 1・2着のみ記録される疎データ（全44k走中~13k）。直近の非空 kimarite が2本未満なら
    leg_change_rate は None（後述の算出例で欠損率を報告）。
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

from src.features.rider_history import _race_date_from_id

# 逃げ残率の縮約(shrinkage)パラメータ。B走(バック先頭≒逃げ)のサンプルが薄いので、その選手の
# 通算top3率（それも無ければ全体事前）へ k サンプル分だけ縮約する。
ESCAPE_SURVIVAL_K = 10.0
# 7車立てで着順がランダムなら top3 に入る確率 = 3/7。履歴ゼロ選手の最終フォールバック。
GLOBAL_TOP3_PRIOR = 3.0 / 7.0
# 脚質変化率で見る直近走数（非空 kimarite のみ）。
LEG_CHANGE_WINDOW = 3


# ---------------------------------------------------------------------------
# A) recent_form 由来（純関数）
# ---------------------------------------------------------------------------
def _pick(rf, *keys):
    """recent_form(テーブル列名: escape_cnt…) と RecentForm(属性名: escape…) の双方を許容して引く。"""
    for k in keys:
        v = rf.get(k) if isinstance(rf, dict) else getattr(rf, k, None)
        if v is not None:
            return v
    return None


def tactics_from_recent_form(rf) -> dict:
    """recent_form 1行（dict もしくは RecentForm）から主導権/仕掛け指数を返す純関数。

    返すキー:
      lead_index      : 逃げ率 + 0.3*捲り率            （決まり手ベース。n=0 は None）
      lead_index_sb   : 0.6*(B率) + 0.4*(S率)          （先頭通過ベース。starts=0 は None）
      sikake          : (逃*700 + 捲*450 + 差*150 + マ*100)/n  仕掛け距離の重み和（n=0 は None）
      kimarite_n      : 決まり手総数 n（透明性のため付随。0=展開系指数が全て None）
    """
    escape = _pick(rf, "escape_cnt", "escape") or 0
    dash = _pick(rf, "dash_cnt", "dash") or 0
    closing = _pick(rf, "closing_cnt", "closing") or 0
    mark = _pick(rf, "mark_cnt", "mark") or 0
    s_count = _pick(rf, "s_count") or 0
    b_count = _pick(rf, "b_count") or 0
    first = _pick(rf, "first_cnt", "first") or 0
    second = _pick(rf, "second_cnt", "second") or 0
    third = _pick(rf, "third_cnt", "third") or 0
    out = _pick(rf, "out_cnt", "out") or 0

    n = escape + dash + closing + mark          # 決まり手（勝ち脚）総数
    starts = first + second + third + out       # 着順が付いた走数

    lead_index = (escape / n + 0.3 * (dash / n)) if n > 0 else None
    sikake = ((escape * 700 + dash * 450 + closing * 150 + mark * 100) / n) if n > 0 else None
    lead_index_sb = (0.6 * (b_count / starts) + 0.4 * (s_count / starts)) if starts > 0 else None

    return {
        "lead_index": lead_index,
        "lead_index_sb": lead_index_sb,
        "sikake": sikake,
        "kimarite_n": n,
    }


# ---------------------------------------------------------------------------
# B) results 履歴由来（as-of ローリング）＋統合バッチ
# ---------------------------------------------------------------------------
def _recent_form_map(conn) -> dict[tuple[str, int], dict]:
    cols = ["race_id", "car_number", "s_count", "b_count", "escape_cnt", "dash_cnt",
            "closing_cnt", "mark_cnt", "first_cnt", "second_cnt", "third_cnt", "out_cnt",
            "win_rate", "top2_rate", "top3_rate"]
    rows = conn.execute(f"SELECT {','.join(cols)} FROM recent_form").fetchall()
    out: dict[tuple[str, int], dict] = {}
    for r in rows:
        d = dict(zip(cols, r))
        out[(d["race_id"], d["car_number"])] = d
    return out


def _history_feats(name, starts, top3, lap_sum, lap_cnt, b_runs, b_top3, kim_hist) -> dict:
    """results履歴アキュムレータ（as-of時点）から avg_last_lap/escape_survival/leg_change_rate を作る。

    学習ループ（compute_pre_race_tactics）と推論（current_tactics 最終状態）で**同一式**を共有し
    train/inference skew を防ぐ。引数は各 defaultdict と kim_hist[name] の deque。
    """
    avg_last_lap = (lap_sum[name] / lap_cnt[name]) if lap_cnt[name] else None
    prior = (top3[name] / starts[name]) if starts[name] else GLOBAL_TOP3_PRIOR
    escape_survival = (b_top3[name] + ESCAPE_SURVIVAL_K * prior) / (b_runs[name] + ESCAPE_SURVIVAL_K)
    hist = list(kim_hist[name])
    if len(hist) >= 2:
        changes = sum(1 for a, b in zip(hist, hist[1:]) if a != b)
        leg_change_rate = changes / (len(hist) - 1)
    else:
        leg_change_rate = None
    return {"avg_last_lap": avg_last_lap, "escape_survival": escape_survival,
            "leg_change_rate": leg_change_rate}


def _run_history(db_path, record_pre: bool):
    """results履歴をas-of集計する共通ループ。

    record_pre=True: 各エントリの発走前 history 特徴を out[(rid,car)] に記録して返す（学習用）。
    record_pre=False: 記録せず全レース処理後の最終アキュムレータと氏名集合を返す（推論=current用）。
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        rf_map = _recent_form_map(conn) if record_pre else {}
        ent_by_race: dict[str, dict[int, str]] = defaultdict(dict)
        for race_id, car, name in conn.execute(
                "SELECT race_id, car_number, rider_name FROM entries"):
            ent_by_race[race_id][car] = name
        res_by_race: dict[str, dict[int, tuple]] = defaultdict(dict)
        for race_id, car, pos, lap, sb, kim in conn.execute(
                "SELECT race_id, car_number, position, last_lap, sb, kimarite FROM results"):
            res_by_race[race_id][car] = (pos, lap, sb, kim)
    finally:
        conn.close()

    race_ids = sorted(ent_by_race, key=lambda rid: (_race_date_from_id(rid) or "", rid))
    acc = dict(starts=defaultdict(int), top3=defaultdict(int), lap_sum=defaultdict(float),
               lap_cnt=defaultdict(int), b_runs=defaultdict(int), b_top3=defaultdict(int),
               kim_hist=defaultdict(lambda: deque(maxlen=LEG_CHANGE_WINDOW)))
    names: set[str] = set()
    out: dict[tuple[str, int], dict] = {}

    for rid in race_ids:
        car_name = ent_by_race[rid]
        for name in car_name.values():
            names.add(name)
        if record_pre:
            for car, name in car_name.items():
                feat = dict(tactics_from_recent_form(rf_map.get((rid, car), {})))
                feat.update(_history_feats(name, acc["starts"], acc["top3"], acc["lap_sum"],
                                           acc["lap_cnt"], acc["b_runs"], acc["b_top3"], acc["kim_hist"]))
                out[(rid, car)] = feat
        for car, (pos, lap, sb, kim) in res_by_race.get(rid, {}).items():
            name = car_name.get(car)
            if name is None:
                continue
            if pos is not None:
                acc["starts"][name] += 1
                if pos <= 3:
                    acc["top3"][name] += 1
            if lap is not None:
                acc["lap_sum"][name] += lap
                acc["lap_cnt"][name] += 1
            if sb and "B" in sb:
                acc["b_runs"][name] += 1
                if pos is not None and pos <= 3:
                    acc["b_top3"][name] += 1
            if kim:
                acc["kim_hist"][name].append(kim)
    return out, acc, names


@lru_cache(maxsize=4)
def current_tactics(db_path: str | Path) -> dict[str, dict]:
    """全history処理後の**現時点**の history 特徴を氏名ごとに返す（推論用・final_elo_state 相当）。

    本日のレース（DB未収録）を予測する際、各出走選手の avg_last_lap/escape_survival/leg_change_rate を
    引くのに使う。compute_pre_race_tactics と同一の集計・式（_history_feats）なので学習と整合する。
    履歴ゼロの選手も含む（escape_survival は事前3/7へ縮約された値）。
    """
    _, acc, names = _run_history(db_path, record_pre=False)
    return {name: _history_feats(name, acc["starts"], acc["top3"], acc["lap_sum"],
                                 acc["lap_cnt"], acc["b_runs"], acc["b_top3"], acc["kim_hist"])
            for name in names}


def tactics_for_entries(entries, recent: dict, current_tac: dict) -> dict[int, dict]:
    """推論用: 出走選手ごとの raw 展開特徴（recent_form由来 + current_tac由来）を車番キーで返す。

    compute_pre_race_tactics が返すのと同じキー構成（lead_index/lead_index_sb/sikake/kimarite_n
    + avg_last_lap/escape_survival/leg_change_rate）。履歴が無い選手は事前分布側の値。
    """
    zero_hist = {"avg_last_lap": None, "escape_survival": GLOBAL_TOP3_PRIOR, "leg_change_rate": None}
    out: dict[int, dict] = {}
    for e in entries:
        rf = recent.get(e.car_number)
        feat = dict(tactics_from_recent_form(rf)) if rf is not None else {
            "lead_index": None, "lead_index_sb": None, "sikake": None, "kimarite_n": 0}
        feat.update(current_tac.get(e.rider_name) or zero_hist)
        out[e.car_number] = feat
    return out


def compute_pre_race_tactics(db_path: str | Path) -> dict[tuple[str, int], dict]:
    """各エントリ (race_id, car_number) の発走前(as-of)展開特徴を統合して返す。

    返す各値のキー:
      lead_index       : 主導権指数（決まり手ベース）          … recent_form / None
      lead_index_sb    : 主導権指数（先頭通過ベース）          … recent_form / None
      sikake           : 仕掛け距離指数                        … recent_form / None
      avg_last_lap     : as-of 平均上がりタイム(秒, 小=速い)   … results履歴 / None
      escape_survival  : 逃げ残率(捲り耐性, shrinkage済 0-1)   … results履歴（常に値・薄い時は事前へ縮約）
      leg_change_rate  : 脚質変化率（直近≤3走の決まり手の変化割合 0-1）… results履歴 / None(kimarite<2本)
      kimarite_n       : recent_form の決まり手総数（診断用）
    """
    out, _, _ = _run_history(db_path, record_pre=True)
    return out


if __name__ == "__main__":  # 簡易サニティ実行: PYTHONIOENCODING=utf-8 python -m src.features.rider_tactics
    import statistics
    from config.settings import DATA_DIR

    feats = compute_pre_race_tactics(DATA_DIR / "keirin.sqlite")
    print("entries:", len(feats))
    for key in ["lead_index", "lead_index_sb", "sikake", "avg_last_lap",
                "escape_survival", "leg_change_rate"]:
        vals = [v[key] for v in feats.values() if v.get(key) is not None]
        miss = 1 - len(vals) / len(feats)
        if vals:
            print(f"{key:16s} n={len(vals):6d} miss={miss:5.1%} "
                  f"min={min(vals):.3f} med={statistics.median(vals):.3f} max={max(vals):.3f}")
        else:
            print(f"{key:16s} all-missing")
