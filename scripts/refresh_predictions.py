"""発走10分前の最新オッズで予測のEVだけを更新する軽量スクリプト（③b・GitHub Actions用）。

DB非依存: 学習済みモデル(pkl, コミット済)＋オッズ取得のみで、dashboard/data.json の
`predictions` セクションと `last_updated` を更新する。他セクション（data_status /
race_type_dist / calibration）は既存値を保持する（DBが無い実行環境でも動くように）。

  python scripts/refresh_predictions.py            # 今日の全ガールズ予測を最新オッズで更新
  python scripts/refresh_predictions.py --only-near 15   # 発走15分以内のレースだけ更新（Actions定期用）
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.collect.base import fetch, set_default_interval
from src.collect.gamboo_schedule import (
    build_kaisai_list_url, parse_kaisai_list, fetch_girls_race_numbers, kaisai_race_date,
)
from predict_race import predict_race_dict
from build_predictions import _venue_map

DEFAULT_OUT = ROOT / "dashboard" / "data.json"
JST = timezone(timedelta(hours=9))


def _minutes_to_deadline(deadline: str, now: datetime) -> float | None:
    """締切(HH:MM, JST)までの分。過ぎていれば負。"""
    if not deadline or ":" not in deadline:
        return None
    h, m = (int(x) for x in deadline.split(":"))
    dl = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return (dl - now).total_seconds() / 60.0


def main() -> None:
    ap = argparse.ArgumentParser(description="最新オッズで予測EVを更新")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--only-near", type=float,
                    help="締切まで N 分以内のレースだけ更新（Actions定期実行用）")
    args = ap.parse_args()
    set_default_interval(0.5)

    out = Path(args.out)
    doc = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    now = datetime.now(JST)
    target = now.date()

    res = fetch(build_kaisai_list_url(target.year, target.month, target.day))
    kaisai_list = [k for k in parse_kaisai_list(res.text)
                   if k.is_girls and kaisai_race_date(k.kaisai_day_code) == target]
    venues = _venue_map(res.text)

    races = []
    for k in kaisai_list:
        venue = venues.get(k.kaisai_code, k.venue_code)
        for rno in fetch_girls_race_numbers(k):
            try:
                d = predict_race_dict(k.kaisai_code, k.kaisai_day_code, rno, venue=venue)
            except Exception as e:
                print(f"  {venue} R{rno} 失敗: {e}")
                continue
            # --only-near: 締切が近いレースだけ最新化（それ以外は既存を保持）
            if args.only_near is not None:
                mins = _minutes_to_deadline(d.get("deadline", ""), now)
                if mins is None or mins < -5 or mins > args.only_near:
                    continue
            races.append(d)

    if args.only_near is not None and doc.get("predictions", {}).get("races"):
        # 既存レースに、近接レースだけ差し替えマージ
        merged = {(r["venue"], r["race_no"]): r for r in doc["predictions"]["races"]}
        for r in races:
            merged[(r["venue"], r["race_no"])] = r
        races = sorted(merged.values(), key=lambda r: (r["venue"], r["race_no"]))

    doc.setdefault("predictions", {})
    doc["predictions"].update({
        "status": "ok" if races else doc["predictions"].get("status", "pending"),
        "date": target.isoformat(),
        "model": "PL線形(拡張20特徴)",
        "note": "着順予測の確率です。EVは最新オッズ×モデルの参考値でエッジ未確立（実弾投入は非推奨）。",
        "last_updated": now.strftime("%Y-%m-%d %H:%M JST"),
        "races": races if races else doc["predictions"].get("races", []),
    })
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"更新: {out}  レース{len(doc['predictions']['races'])}  {now:%H:%M JST}")


if __name__ == "__main__":
    main()
