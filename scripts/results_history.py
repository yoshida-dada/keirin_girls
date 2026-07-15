"""過去約1ヶ月の開催結果を「モデルの印(◎○▲△×)・決着印順・決着三連単のモデル順位・払戻・
レースタイプ」で集計する（ダッシュボード成績セクション用）。

印 = モデル1着確率の降順に ◎(1)→○(2)→▲(3)→△(4)→×(5)、6位以降は「−」。
決着印順 = 実際の1着/2着/3着 の印を連結（例「◎△○」）。
予測順位 = 決着した三連単(1-2-3着)がモデルの三連単確率で何位か（1〜210）。
特徴は本番31特徴を as-of 付与（build と同一の feature_augment.augment_samples）。build_predictions から
呼ばれ data.json の results_history に載り、スケジューラの日次buildで自動更新される。

  python scripts/results_history.py --db data/keirin.sqlite --days 30
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.persist import load_model
from src.model.feature_augment import augment_samples
from src.model.race_type import classify_race
from src.model.plackett_luce import all_trifecta_probs
from src.features.rider_history import _race_date_from_id
from src.features.venue_meta import venue_name

MARKS = ["◎", "○", "▲", "△", "×"]


def _mark(rank: int) -> str:
    return MARKS[rank] if rank < len(MARKS) else "−"


def build_results_history(db_path, days: int = 30) -> dict:
    """直近 days 日の開催結果を集計して dict を返す（dashboard用）。"""
    model = load_model()
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    all_samples = load_samples(db_path, features=PL_FEATURES_FULL)
    recent = [s for s in all_samples
              if (_race_date_from_id(s.race_id) or "") >= cutoff and len(s.order) >= 3]
    if not recent:
        return {"status": "pending", "note": "直近の確定結果がありません。"}
    recent = augment_samples(recent, db_path, model.feature_names)   # as-of 特徴（本番31特徴）

    ids = [s.race_id for s in recent]
    ph = ",".join("?" * len(ids))
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        names = {(rid, c): nm for rid, c, nm in conn.execute(
            f"SELECT race_id, car_number, rider_name FROM entries WHERE race_id IN ({ph})", ids)}
        meta = {rid: (vc, rno) for rid, vc, rno in conn.execute(
            f"SELECT race_id, venue_code, race_no FROM races WHERE race_id IN ({ph})", ids)}
        payouts = {rid: (combo, pay, pop) for rid, combo, pay, pop in conn.execute(
            f"SELECT race_id, combo, payout, popularity FROM payouts_trifecta WHERE race_id IN ({ph})", ids)}
    finally:
        conn.close()

    races = []
    for s in recent:
        st = model.strengths(s.X, s.car_numbers)                    # {車番: 1着確率}
        if not st:
            continue
        ranked = sorted(st.items(), key=lambda kv: -kv[1])
        mark_of = {car: _mark(i) for i, (car, _) in enumerate(ranked)}
        rid = s.race_id
        vc, rno = meta.get(rid, ("", None))
        order = list(s.order[:3])                                    # 1,2,3着の車番
        probs = all_trifecta_probs(st)
        tri = sorted(probs, key=lambda k: -probs[k])
        actual = tuple(order)
        tri_rank = (tri.index(actual) + 1) if actual in tri else None
        pay = payouts.get(rid)
        races.append({
            "date": _race_date_from_id(rid),
            "venue": venue_name(vc) or vc,
            "race_no": rno,
            "race_type": classify_race(st).label,                   # 軸堅/標準/混戦
            "marks": [{"car": c, "name": names.get((rid, c), ""),
                       "mark": mark_of[c], "win_prob": round(p, 4)} for c, p in ranked],
            "finish": [{"pos": i + 1, "car": c, "name": names.get((rid, c), ""),
                        "mark": mark_of.get(c, "−")} for i, c in enumerate(order)],
            "finish_marks": "".join(mark_of.get(c, "−") for c in order),
            "win_combo": pay[0] if pay else "-".join(map(str, order)),
            "tri_rank": tri_rank,
            "payout": pay[1] if pay else None,
            "popularity": pay[2] if pay else None,
        })
    races.sort(key=lambda r: (r["date"], r["race_no"] or 0), reverse=True)   # 新しい順
    return {
        "status": "ok",
        "note": "過去約1ヶ月の開催結果。印=モデル1着確率順(◎○▲△×)、予測順位=決着三連単のモデル確率順位。",
        "period": f"{races[-1]['date']}〜{races[0]['date']}",
        "n_races": len(races),
        "races": races,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="過去1ヶ月の成績集計")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    h = build_results_history(args.db, days=args.days)
    if h["status"] != "ok":
        print(h); return
    print(f"{h['period']} / {h['n_races']}レース")
    for r in h["races"][:8]:
        print(f"  {r['date']} {r['venue']}R{r['race_no']} [{r['race_type']}] "
              f"決着{r['finish_marks']} 三連単{r['win_combo']} モデル{r['tri_rank']}位 "
              f"払戻{r['payout']}円({r['popularity']}人気)")


if __name__ == "__main__":
    main()
