"""選手間の関係性（同県・同地区・同期）特徴。entries のみから発走前に確定する（リーク無し）。

ガールズはラインが無いが、**同県選手は自力ある選手の後ろに入って「ライン的なもの」を作る**ことが
あるとされる。同地区・同期でも連携の可能性。本モジュールは出走表(entries)だけを使って、各出走者が
「どんな仲間と一緒に走っているか」を数値化する（着順=results は一切参照しない＝学習特徴としてリーク無し）。

per (race_id, car_number):
  n_same_pref       : 同県の他選手数（0..6）
  n_same_district   : 同地区の他選手数（同県を含む。0..6）
  n_same_term       : 同期（term一致）の他選手数（term不明=999/Nullは非該当）
  ally_of_top       : この車が有力者本人でなく、有力者と同県なら1（＝番手候補・恩恵を受けうる側）
  ally_of_top_dist  : 有力者本人でなく、同県ではないが同地区なら1（地区連携候補）
  top_is_allied     : この車が有力者本人で、かつ同県の仲間が居るなら1（＝強化されたライン先頭）

有力者(top)は **racing_score 最大**の車で定義（発走前確定・タイは最小車番）。model 1着確率で定義した
版は診断側(analyze_relations)で別途扱う。返り値: {(race_id, car_number): {上記キー}}。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

# 標準的な競輪地区（府県→地区）。新潟は競輪では関東地区所属のため関東に含める（ユーザーマップ補完）。
DISTRICT_OF: dict[str, str] = {}
for _dist, _prefs in {
    "北日本": ["北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島"],
    "関東": ["茨城", "栃木", "群馬", "埼玉", "新潟"],
    "南関東": ["千葉", "東京", "神奈川", "山梨", "静岡"],
    "中部": ["愛知", "岐阜", "三重", "富山", "石川", "福井", "長野"],
    "近畿": ["滋賀", "京都", "大阪", "兵庫", "奈良", "和歌山"],
    "中国": ["岡山", "広島", "山口", "鳥取", "島根"],
    "四国": ["香川", "徳島", "愛媛", "高知"],
    "九州": ["福岡", "佐賀", "長崎", "熊本", "大分", "宮崎", "鹿児島", "沖縄"],
}.items():
    for _p in _prefs:
        DISTRICT_OF[_p] = _dist


def normalize_pref(pref) -> str | None:
    """府県表記を正規化（空白除去）。'外国'/不明は None を返す（連携対象外）。"""
    if pref is None:
        return None
    p = str(pref).replace(" ", "").replace("　", "").strip()
    if not p or p == "外国":
        return None
    return p


def district_of(pref) -> str | None:
    return DISTRICT_OF.get(normalize_pref(pref) or "")


RELATION_KEYS = [
    "n_same_pref", "n_same_district", "n_same_term",
    "ally_of_top", "ally_of_top_dist", "top_is_allied",
]


def compute_relation_features(db_path: str | Path) -> dict[tuple[str, int], dict]:
    """entries から全レースの関係性特徴を {(race_id, car): {...}} で返す（field_size問わず）。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        rows = conn.execute(
            "SELECT race_id, car_number, prefecture, term, racing_score FROM entries"
        ).fetchall()
    finally:
        conn.close()

    # レース単位に束ねる
    by_race: dict[str, list] = {}
    for rid, car, pref, term, score in rows:
        by_race.setdefault(rid, []).append((car, pref, term, score))

    out: dict[tuple[str, int], dict] = {}
    for rid, members in by_race.items():
        # 有力者 = racing_score 最大（None は最下位扱い、タイは最小車番）
        def _key(m):
            return (m[3] if m[3] is not None else -1e9, -m[0])
        top = max(members, key=_key)
        top_car = top[0]
        top_pref = normalize_pref(top[1])
        top_dist = district_of(top[1])

        for car, pref, term, _score in members:
            npref = normalize_pref(pref)
            dist = district_of(pref)
            valid_term = term if (term is not None and term != 999) else None

            n_same_pref = sum(
                1 for (c2, p2, _t2, _s2) in members
                if c2 != car and npref is not None and normalize_pref(p2) == npref)
            n_same_dist = sum(
                1 for (c2, p2, _t2, _s2) in members
                if c2 != car and dist is not None and district_of(p2) == dist)
            n_same_term = sum(
                1 for (c2, _p2, t2, _s2) in members
                if c2 != car and valid_term is not None
                and (t2 if (t2 is not None and t2 != 999) else None) == valid_term)

            is_top = (car == top_car)
            same_pref_top = (not is_top and npref is not None and npref == top_pref)
            same_dist_top = (not is_top and dist is not None and dist == top_dist)

            out[(rid, car)] = {
                "n_same_pref": float(n_same_pref),
                "n_same_district": float(n_same_dist),
                "n_same_term": float(n_same_term),
                "ally_of_top": 1.0 if same_pref_top else 0.0,
                "ally_of_top_dist": 1.0 if (same_dist_top and not same_pref_top) else 0.0,
                "top_is_allied": 1.0 if (is_top and n_same_pref > 0) else 0.0,
            }
    return out
