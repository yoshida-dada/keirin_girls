"""dashboard/data.json を「予測AI」の実データで生成する（方針A・S6運用UI）。

build_dashboard_data の data_status に加え、学習済みモデルで:
  - predictions   : 指定日のガールズ各レースの予測（レースタイプ・各車1着確率・三連単上位）※ネットワーク
  - race_type_dist: 収集済みデータ全レースをモデル分類した軸堅/標準/混戦の構成比（実データ）
  - calibration   : 時系列分割の検証で測った1着確率の reliability curve と Brier（実データ・課題B）
を出力する。buckets/recommendations/cumulative_roi は「エッジ未確立」につき pending のまま
（黒字買い目は提示しない）。

  python scripts/build_predictions.py                 # race_type_dist + calibration のみ（高速）
  python scripts/build_predictions.py --predict       # 本日のガールズ予測も生成（ネットワーク）
  python scripts/build_predictions.py --date 2026-07-14 --predict --max-races 40
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # build_dashboard_data / predict_race

import build_dashboard_data as bdd
from bs4 import BeautifulSoup
from src.collect.base import fetch, set_default_interval
from src.collect.gamboo_schedule import (
    build_kaisai_list_url, parse_kaisai_list, fetch_girls_race_numbers, kaisai_race_date,
)
from src.model.persist import load_model
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.race_type import classify_race
from src.model.evaluate import time_split
from src.backtest.calibration import reliability_curve, brier_score
from predict_race import predict_race_dict

DEFAULT_OUT = ROOT / "dashboard" / "data.json"


def _venue_map(html: str) -> dict[str, str]:
    """開催一覧HTML → {開催コード: 会場名}。"""
    import re
    soup = BeautifulSoup(html, "html.parser")
    ul = soup.find("ul", class_="kaisai_list")
    out = {}
    if not ul:
        return out
    for li in ul.find_all("li", recursive=False):
        a = li.find("a", href=re.compile(r"/race-list/"))
        if not a:
            continue
        parts = a["href"].split("/race-list/")[1].strip("/").split("/")
        name = li.get_text(" ", strip=True)
        m = re.search(r"([^\s]+?競輪)", name)
        out[parts[0]] = m.group(1) if m else name[:8]
    return out


def build_predictions_section(target: date) -> dict:
    """指定日のガールズ各レースをモデル予測して返す（ネットワークアクセスあり）。"""
    set_default_interval(0.5)
    res = fetch(build_kaisai_list_url(target.year, target.month, target.day))
    # 開催一覧は初日〜最終日の全日程を含むため、実施日が target と一致する開催日のみに絞る
    # （初日=昨日/2日目=今日 の混在を防ぐ。過去レースへの予測を出さない）。
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
                print(f"  {venue} R{rno} 予測失敗: {e}")
                continue
            races.append(d)
    from datetime import datetime, timezone, timedelta
    jst = timezone(timedelta(hours=9))
    return {
        "status": "ok" if races else "pending",
        "date": target.isoformat(),
        "model": "PL線形(拡張20特徴)",
        "note": "着順予測の確率です。EVは最新オッズ×モデルの参考値でエッジ未確立（実弾投入は非推奨）。",
        "last_updated": datetime.now(jst).strftime("%Y-%m-%d %H:%M JST"),
        "races": races,
    }


def build_model_sections(db_path: Path) -> dict:
    """収集済みデータと学習済みモデルから race_type_dist と calibration を実データ化する。"""
    import numpy as np
    model = load_model()
    samples = load_samples(db_path, features=PL_FEATURES_FULL)
    # Eloモデル(rel_elo付き)のときは各サンプルのXにレース内相対Eloを足す
    if "rel_elo" in (model.feature_names or []):
        from src.model.elo import compute_pre_race_elo, DEFAULT_ELO
        pre = compute_pre_race_elo(db_path)
        for s in samples:
            elos = np.array([pre.get((s.race_id, c), DEFAULT_ELO) for c in s.car_numbers])
            s.X = np.hstack([s.X, (elos - elos.mean()).reshape(-1, 1)])
    # レースタイプ分布（全サンプルをモデル分類）
    counts = {"軸堅": 0, "標準": 0, "混戦": 0}
    for s in samples:
        st = model.strengths(s.X, s.car_numbers)
        counts[classify_race(st).label] += 1
    race_type_dist = {"status": "ok", "note": "収集データを学習済みモデルで分類した構成比。",
                      "counts": [{"type": t, "n": n} for t, n in counts.items()]}
    # キャリブレーション（時系列分割の検証側で1着確率の reliability）
    _, test = time_split(samples, 0.25)
    pairs = []
    for s in test:
        st = model.strengths(s.X, s.car_numbers)
        winner = s.order[0]
        for car, p in st.items():
            pairs.append((p, 1 if car == winner else 0))
    bins = [{"lo": round(b.lo, 2), "hi": round(b.hi, 2),
             "mean_pred": round(b.mean_pred, 4) if b.mean_pred is not None else None,
             "emp_freq": round(b.emp_freq, 4) if b.emp_freq is not None else None,
             "n": b.count} for b in reliability_curve(pairs, 10)]
    calibration = {"status": "ok",
                   "note": "検証期間の1着確率の較正（対角線に近いほど良い, 課題B）。",
                   "brier": round(brier_score(pairs), 5), "n": len(pairs), "bins": bins}
    return {"race_type_dist": race_type_dist, "calibration": calibration}


def main() -> None:
    ap = argparse.ArgumentParser(description="予測AIのdata.json生成")
    ap.add_argument("--db", default=str(bdd.DEFAULT_DB))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--date", help="予測日 YYYY-MM-DD（既定=今日）")
    ap.add_argument("--predict", action="store_true", help="本日のガールズ予測を生成（ネットワーク）")
    args = ap.parse_args()

    db_path = Path(args.db)
    doc = bdd.build(db_path)                     # data_status + pending 一式
    doc["model_ready"] = True
    doc.update(build_model_sections(db_path))    # race_type_dist / calibration を実データ化
    if args.predict:
        target = date.fromisoformat(args.date) if args.date else date.today()
        print(f"{target} のガールズ予測を生成中…")
        doc["predictions"] = build_predictions_section(target)
        print(f"  予測レース数: {len(doc['predictions']['races'])}")

    out = Path(args.out)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    rt = {c["type"]: c["n"] for c in doc["race_type_dist"]["counts"]}
    print(f"生成: {out}")
    print(f"  レースタイプ分布: {rt}  / Brier: {doc['calibration']['brier']}")


if __name__ == "__main__":
    main()
