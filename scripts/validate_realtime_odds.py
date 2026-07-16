"""リアルタイムオッズ選定 vs 最終オッズ選定の実現ROI比較（going-forward検証）。

パリミチュアルは確定配当で支払われるため、回収は常に最終配当(payouts_trifecta)。本スクリプトは
「その時刻に見えていたオッズ(odds_snapshots)でEV判定して買い、最終配当で決済」した場合のROIを、
締切までの残り時間(バケット)別に集計し、「最終オッズで判定」した場合と比較する。
→ 早い(ソフトな)オッズで選ぶと選定が良くなる/悪くなるかを実測する（最終オッズ選定は未来情報の
カンニングなので、リアルタイム選定が現実の上限に近い）。

前提: 対象レースは (a) odds_snapshots に時系列があり、(b) 本DBに entries/recent_form/results/
odds_final/payouts が揃っている（＝full収集済みで確定）こと。スナップショット蓄積は今日以降なので、
数週間データが貯まるまで対象は少ない。

  PYTHONIOENCODING=utf-8 python scripts/validate_realtime_odds.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.persist import load_model
from src.backtest.bucket_analysis import build_records
from src.backtest.selection import group_by_race, settle_full

ROOT = Path(__file__).resolve().parent.parent
SNAP_DB = ROOT / "data" / "odds_snapshots.sqlite"
# 締切までの残り分 → バケット
BUCKETS = [(0, 5, "≤5分"), (5, 15, "5-15分"), (15, 30, "15-30分"),
           (30, 60, "30-60分"), (60, 999, "60分+")]


def _mins(deadline: str, taken_at: str) -> float | None:
    """deadline(HH:MM) と taken_at(ISO) から締切までの残り分。異日/欠損は None。"""
    from datetime import datetime
    if not deadline or ":" not in deadline:
        return None
    try:
        ta = datetime.fromisoformat(taken_at)
    except ValueError:
        return None
    h, m = (int(x) for x in deadline.split(":"))
    dl = ta.replace(hour=h, minute=m, second=0, microsecond=0)
    return (dl - ta).total_seconds() / 60.0


def main() -> None:
    ap = argparse.ArgumentParser(description="リアルタイムvs最終オッズ選定のROI比較")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--ev", type=float, default=1.10, help="EV閾値")
    args = ap.parse_args()

    if not SNAP_DB.exists():
        print("odds_snapshots.sqlite が無い。スケジューラでリアルタイム取得を蓄積してから実行。"); return
    snap = sqlite3.connect(f"file:{SNAP_DB}?mode=ro", uri=True)
    snap_ids = [r[0] for r in snap.execute("SELECT DISTINCT race_id FROM odds_snapshots_trifecta")]

    main_c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    paid = {r[0] for r in main_c.execute("SELECT DISTINCT race_id FROM payouts_trifecta")}
    usable = [rid for rid in snap_ids if rid in paid]
    print(f"スナップショット {len(snap_ids)}レース / うち確定済み(本DBに払戻あり) {len(usable)}レース")
    if not usable:
        print("検証可能レースがまだありません（今日以降の蓄積を待つ）。"); return

    # モデル確率・最終オッズ・的中・確定配当（build_records は full収集済みのみ通す）
    model = load_model()
    recs = build_records(args.db, model, usable)
    by_race = group_by_race(recs)
    dls = dict(main_c.execute(
        f"SELECT race_id, deadline FROM races WHERE race_id IN ({','.join('?'*len(usable))})", usable))
    print(f"モデル確率を計算できたレース: {len(by_race)}\n")

    # 参考: 最終オッズでEV選定（未来情報カンニング＝上限の目安）
    def sel_final(rrecs):
        return [r for r in rrecs if r.odds and r.odds < 9999 and r.ev >= args.ev]
    final_by = {rid: sel_final(rs) for rid, rs in by_race.items()}
    sf = settle_full({k: v for k, v in final_by.items() if v})

    # スナップショット時刻のオッズでEV選定 → バケット別に決済
    agg = {b[2]: {} for b in BUCKETS}
    for rid, rrecs in by_race.items():
        dl = dls.get(rid)
        model_p = {r.combo: r.model_prob for r in rrecs}
        rec_of = {r.combo: r for r in rrecs}
        taken = [t[0] for t in snap.execute(
            "SELECT DISTINCT taken_at FROM odds_snapshots_trifecta WHERE race_id=?", (rid,))]
        # バケットごとに「そのバケットで最も締切に近いスナップショット」を選ぶ
        best = {}
        for ta in taken:
            mm = _mins(dl, ta)
            if mm is None or mm < 0:
                continue
            for lo, hi, name in BUCKETS:
                if lo <= mm < hi:
                    if name not in best or mm < best[name][0]:
                        best[name] = (mm, ta)
        for name, (_, ta) in best.items():
            snap_odds = {tuple(int(x) for x in c.split("-")): o for c, o in snap.execute(
                "SELECT combo, odds FROM odds_snapshots_trifecta WHERE race_id=? AND taken_at=?", (rid, ta))}
            picks = [rec_of[combo] for combo, o in snap_odds.items()
                     if combo in rec_of and o and o < 9999 and model_p[combo] * o >= args.ev]
            if picks:
                agg[name][rid] = picks

    print(f"{'選定タイミング':<12}{'R数':>6}{'点/R':>7}{'的中':>6}{'的中率':>8}{'ROI均等':>8}{'ROIドッチ':>9}")
    if sf:
        print(f"{'最終オッズ(参考)':<12}{sf['n_races']:>6}{sf['pts']:>7.1f}{sf['n_hits']:>6}"
              f"{sf['hit_rate']*100:>7.1f}%{sf['roi_eq']*100:>8.1f}%{sf['roi_du']*100:>9.1f}%")
    for _, _, name in BUCKETS:
        s = settle_full(agg[name])
        if s:
            print(f"{name:<12}{s['n_races']:>6}{s['pts']:>7.1f}{s['n_hits']:>6}"
                  f"{s['hit_rate']*100:>7.1f}%{s['roi_eq']*100:>8.1f}%{s['roi_du']*100:>9.1f}%")
        else:
            print(f"{name:<12}{'(データ不足)':>18}")
    print("\n※ 回収は常に最終配当(パリミチュアル)。ここは『どの時刻のオッズで買い目を選ぶと"
          "最終配当ベースのROIが良いか』の比較。的中が十分貯まるまで数値は参考値。")


if __name__ == "__main__":
    main()
