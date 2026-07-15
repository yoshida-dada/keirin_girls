"""選手の現在(as-of最新)成績と対戦成績（ダッシュボード表示用）。

rider_rolling.py は各過去レースの発走前(as-of)値を返す学習用モジュールだが、本モジュールは
**本日のレース**を出走表から予測する場面で使う。収集済み全履歴(results/races)を集計した
「現時点での」各選手の通算成績・直近成績・対戦成績を、選手氏名で引けるように返す。

  current_stats(db)  … {氏名: {career_win_rate, career_starts, recent5_avg_finish,
                              last_date, venue: {場コード: 勝率}, venue_starts: {場コード: 出走数}}}
  head_to_head(db, {車番: 氏名})
                     … その1レースの出走者同士の過去対戦成績（同一レースでの着順比較）を
                       車番キーのマトリクスで返す。

いずれも results.rider_name（全行populated）で同定する。本日のレースは収集済みDBに含まれない
ため、集計に当該レースは混ざらず（＝リークにならず）「発走前の実力指標」として使える。
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict, deque
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def current_stats(db_path: str) -> dict[str, dict]:
    """収集済み全履歴を集計した、各選手(氏名)の現時点の成績を返す。

    lru_cache で同一プロセス内の複数レース予測をまたいで再利用する（全走行を1度だけ走査）。
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        rows = conn.execute(
            "SELECT ra.race_date, res.rider_name, res.position, ra.venue_code"
            " FROM results res JOIN races ra ON res.race_id = ra.race_id"
            " WHERE res.position IS NOT NULL AND res.rider_name IS NOT NULL"
            " ORDER BY ra.race_date, res.race_id"
        ).fetchall()
    finally:
        conn.close()

    starts: dict[str, int] = defaultdict(int)
    wins: dict[str, int] = defaultdict(int)
    venue_starts: dict[tuple[str, str], int] = defaultdict(int)
    venue_wins: dict[tuple[str, str], int] = defaultdict(int)
    recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
    last_date: dict[str, str] = {}
    names: set[str] = set()

    for rdate, name, pos, venue in rows:
        names.add(name)
        starts[name] += 1
        venue_starts[(name, venue)] += 1
        if pos == 1:
            wins[name] += 1
            venue_wins[(name, venue)] += 1
        recent[name].append(pos)
        if rdate:
            last_date[name] = rdate

    out: dict[str, dict] = {}
    for name in names:
        s = starts[name]
        rc = recent[name]
        vcodes = {v for (n, v) in venue_starts if n == name}
        out[name] = {
            "career_starts": s,
            "career_win_rate": (wins[name] / s) if s else None,
            "recent5_avg_finish": (sum(rc) / len(rc)) if rc else None,
            "last_date": last_date.get(name),
            "venue": {v: (venue_wins[(name, v)] / venue_starts[(name, v)])
                      for v in vcodes if venue_starts[(name, v)]},
            "venue_starts": {v: venue_starts[(name, v)] for v in vcodes},
        }
    return out


@lru_cache(maxsize=8)
def style_counts(db_path: str, window_days: int = 365) -> dict[str, dict]:
    """各選手(氏名)の脚質プロファイルを直近 window_days 日で集計して返す。

    基準日 = 収集済みデータの最新レース日（安定・キャッシュ可能）。返り値は各氏名につき:
      s      : S(スタート先頭)回数     b : B(バック先頭)回数
      nige   : 1着かつ決まり手=逃 の回数（逃げ型の指標）
      makuri : 1着かつ決まり手=捲 の回数（捲り型）
      sashi  : 1着かつ決まり手=差 の回数（差し型）
      races  : 集計対象の出走数（母数）
    ガールズは連携が薄くマーク先着(1着マ)がほぼ無いため、逃/捲/差 を主指標とする。
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        latest = conn.execute("SELECT MAX(race_date) FROM races").fetchone()[0]
        if not latest:
            return {}
        cutoff = (date.fromisoformat(latest) - timedelta(days=window_days)).isoformat()
        rows = conn.execute(
            "SELECT res.rider_name, res.position, res.sb, res.kimarite"
            " FROM results res JOIN races ra ON res.race_id = ra.race_id"
            " WHERE ra.race_date >= ? AND res.rider_name IS NOT NULL",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()

    agg: dict[str, dict] = {}
    for name, pos, sb, kim in rows:
        a = agg.setdefault(name, {"s": 0, "b": 0, "nige": 0, "makuri": 0, "sashi": 0, "races": 0})
        a["races"] += 1
        if sb:
            if "S" in sb:
                a["s"] += 1
            if "B" in sb:
                a["b"] += 1
        if pos == 1 and kim:
            if kim == "逃":
                a["nige"] += 1
            elif kim == "捲":
                a["makuri"] += 1
            elif kim == "差":
                a["sashi"] += 1
    return agg


def _race_date_from_id(race_id: str) -> str | None:
    """race_id から実施日を復元する。

    races.race_date は開催全日で「開催初日」が入っており実際の実施日ではない（収集時の既知の癖）。
    race_id の先頭14桁 = 開催日コード {場2}{初日YYYYMMDD}{開催日目NN}{連番2} から
    実施日 = 初日 + (NN-1)日 を計算する（gamboo_schedule.kaisai_race_date と同じ）。
    """
    try:
        start = date(int(race_id[2:6]), int(race_id[6:8]), int(race_id[8:10]))
        nn = int(race_id[10:12])
        return (start + timedelta(days=nn - 1)).isoformat()
    except (ValueError, IndexError):
        return None


def meet_results(db_path: str, names: tuple[str, ...], kaisai_code: str) -> dict[str, list]:
    """出走者の「今場所成績」（当該開催の各走の 日付/レース番号/着順/上り）を返す。

    開催は race_id の先頭10桁(=場コード+開催初日)で同定する（kaisai_code と一致）。本日のレースは
    DB未収録なので、返るのは2日目以降に見える前日までの走のみ（＝初日は空）。日付は races.race_date
    ではなく race_id から復元する（race_date は初日固定のため、日別に区別できないため）。
    返り値: {氏名: [{"date","race_no","position","last_lap"} ...]}（実施日・R昇順）。
    """
    if not names:
        return {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        ph = ",".join("?" * len(names))
        rows = conn.execute(
            f"SELECT res.rider_name, res.race_id, ra.race_no, res.position, res.last_lap"
            f" FROM results res JOIN races ra ON res.race_id = ra.race_id"
            f" WHERE res.rider_name IN ({ph}) AND res.race_id LIKE ?",
            (*names, kaisai_code + "%"),
        ).fetchall()
    finally:
        conn.close()

    out: dict[str, list] = {}
    for name, race_id, rno, pos, lap in rows:
        out.setdefault(name, []).append(
            {"date": _race_date_from_id(race_id) or "", "race_no": rno,
             "position": pos, "last_lap": lap})
    for lst in out.values():                      # 実施日・R昇順に整列
        lst.sort(key=lambda m: (m["date"], m["race_no"] if m["race_no"] is not None else 0))
    return out


def head_to_head(db_path: str, car_name: dict[int, str]) -> dict:
    """その1レースの出走者同士の過去対戦成績を車番キーのマトリクスで返す。

    同一レースで両者が着順を持つ過去走を「対戦」とみなし、各ペアで先着回数を数える。
    返り値: {"cars": [車番...], "names": {車番: 氏名}, "cell": {a: {b: {"w","l","n"}}}}
      w = a が b に先着した回数 / l = b が a に先着した回数 / n = 対戦数(w+l)。
    """
    cars = sorted(car_name)
    names = [car_name[c] for c in cars]
    name_to_car = {}
    for c in cars:                       # 同名衝突時は最小車番を採用（表示用途のため許容）
        name_to_car.setdefault(car_name[c], c)

    if len(set(names)) < 2:
        return {"cars": cars, "names": {c: car_name[c] for c in cars}, "cell": {}}

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        ph = ",".join("?" * len(names))
        rows = conn.execute(
            f"SELECT race_id, rider_name, position FROM results"
            f" WHERE position IS NOT NULL AND rider_name IN ({ph})",
            names,
        ).fetchall()
    finally:
        conn.close()

    # レースごとに {氏名: 着順} を集め、当該レースに居る出走者ペアの先着関係を数える
    by_race: dict[str, dict[str, int]] = defaultdict(dict)
    for race_id, name, pos in rows:
        if name in name_to_car:
            by_race[race_id][name] = pos

    cell: dict[int, dict[int, dict]] = defaultdict(dict)
    for positions in by_race.values():
        present = list(positions.items())
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                na, pa = present[i]
                nb, pb = present[j]
                ca, cb = name_to_car[na], name_to_car[nb]
                if ca == cb:
                    continue
                ra = cell[ca].setdefault(cb, {"w": 0, "l": 0, "n": 0})
                rb = cell[cb].setdefault(ca, {"w": 0, "l": 0, "n": 0})
                if pa < pb:
                    ra["w"] += 1; rb["l"] += 1
                else:
                    ra["l"] += 1; rb["w"] += 1
                ra["n"] += 1; rb["n"] += 1

    return {
        "cars": cars,
        "names": {c: car_name[c] for c in cars},
        "cell": {a: dict(b) for a, b in cell.items()},
    }
