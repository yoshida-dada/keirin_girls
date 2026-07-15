"""予測実績の推移（D13の時系列拡張）。検証期間の的中率を週次で集計する。

verify_predictions.build_accuracy_section は検証期間全体の的中率を1点で出すが、本モジュールは
同じ検証期間（time_split の後ろ 25%）のレースを ISO 週（週の月曜始まり）でバケットに区切り、
各週の 1着的中率・上位3内率・三連単Top10率・レース数 を集計する。モデル精度が時間的に安定
しているか（学習に使っていない直近区間での週次実績）を可視化するためのデータを返す。

  python scripts/accuracy_history.py --db data/keirin.sqlite

build_accuracy_history(db) は dashboard/data.json 用の dict を返す（build_predictions から利用）。
買い目推奨ではなく、検証期間でモデル予測を実結果と照合した実績の推移である。
"""
from __future__ import annotations

import argparse
import copy
import sys
from datetime import date as _date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.persist import load_model
from src.model.evaluate import time_split
from src.model.elo import compute_pre_race_elo, DEFAULT_ELO
from src.model.plackett_luce import all_trifecta_probs


def _augment_if_elo(db_path, model, samples):
    """Eloモデル(rel_elo付き)のとき各サンプルのXにレース内相対Eloを足す（verify_predictions と同型）。"""
    if "rel_elo" not in (model.feature_names or []):
        return samples
    pre = compute_pre_race_elo(db_path)
    out = []
    for s in samples:
        s2 = copy.copy(s)
        elos = np.array([pre.get((s.race_id, c), DEFAULT_ELO) for c in s.car_numbers])
        s2.X = np.hstack([s.X, (elos - elos.mean()).reshape(-1, 1)])
        out.append(s2)
    return out


def _week_start(d: str) -> str:
    """日付文字列 YYYY-MM-DD → その週の月曜(ISO週始まり)の YYYY-MM-DD。"""
    y, m, dd = (int(x) for x in d.split("-"))
    dt = _date(y, m, dd)
    return (dt - timedelta(days=dt.weekday())).isoformat()


def build_accuracy_history(db_path, test_frac: float = 0.25, weeks: int | None = None) -> dict:
    """検証期間の的中率を週次で集計して dict を返す（dashboard用）。

    weeks を与えると直近 weeks 週分のみを返す（None=全週）。
    """
    model = load_model()
    samples = load_samples(db_path, features=PL_FEATURES_FULL)
    samples = _augment_if_elo(db_path, model, samples)
    _, test = time_split(samples, test_frac)

    # week_start ごとに集計（samples は date 昇順なので週も昇順に並ぶ）
    order: list[str] = []
    agg: dict[str, dict] = {}
    for s in test:
        st = model.strengths(s.X, s.car_numbers)
        if not st or len(s.order) < 3:
            continue
        wk = _week_start(s.date)
        if wk not in agg:
            agg[wk] = {"n": 0, "top1": 0, "top3": 0, "tri10": 0, "first": s.date, "last": s.date}
            order.append(wk)
        b = agg[wk]
        winner = s.order[0]
        ranked = sorted(st, key=lambda c: -st[c])
        b["top1"] += int(ranked[0] == winner)
        b["top3"] += int(winner in ranked[:3])
        probs = all_trifecta_probs(st)
        tri_rank = [c for c, _ in sorted(probs.items(), key=lambda kv: -kv[1])]
        actual = tuple(s.order[:3])
        pos = tri_rank.index(actual) + 1 if actual in tri_rank else 9999
        b["tri10"] += int(pos <= 10)
        b["n"] += 1
        b["last"] = s.date  # date昇順のため最後の観測が週内最終日

    if not order:
        return {"status": "pending", "note": "検証データ不足"}

    if weeks:
        order = order[-weeks:]
    rows = []
    for wk in order:
        b = agg[wk]
        n = b["n"]
        rows.append({
            "week_start": wk,
            "date_from": b["first"],
            "date_to": b["last"],
            "n_races": n,
            "top1_rate": round(b["top1"] / n, 4),
            "top3_rate": round(b["top3"] / n, 4),
            "tri10_rate": round(b["tri10"] / n, 4),
        })

    return {
        "status": "ok",
        "note": "検証期間（学習に使っていない直近区間）を週単位に区切ったモデル予測の的中実績の推移。買い目推奨ではありません。",
        "period": f"{rows[0]['date_from']}〜{rows[-1]['date_to']}",
        "n_weeks": len(rows),
        "n_races": sum(r["n_races"] for r in rows),
        "weeks": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="予測実績の推移（D13時系列拡張）")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--weeks", type=int, default=None, help="直近N週のみ表示（既定=全週）")
    args = ap.parse_args()
    h = build_accuracy_history(args.db, weeks=args.weeks)
    if h["status"] != "ok":
        print(h); return
    print(f"検証期間 {h['period']} / {h['n_weeks']}週 / {h['n_races']}レース")
    print(f"{'週(月曜始)':<12} {'レース':>5} {'1着':>7} {'上位3内':>8} {'三連単Top10':>11}")
    for r in h["weeks"]:
        print(f"{r['week_start']:<12} {r['n_races']:>5} "
              f"{r['top1_rate']*100:>6.1f}% {r['top3_rate']*100:>7.1f}% {r['tri10_rate']*100:>10.1f}%")


if __name__ == "__main__":
    main()
