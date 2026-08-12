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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # build_dashboard_data / predict_race

import build_dashboard_data as bdd
from bs4 import BeautifulSoup
from src.collect.base import fetch, set_default_interval
from src.collect.gamboo_schedule import (
    build_kaisai_list_url, parse_kaisai_list, fetch_girls_race_numbers, kaisai_race_date,
    fetch_race_numbers,
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


def build_races_for_date(target: date, include: str = "girls") -> list:
    """指定日の各レースをモデル予測して返す（ネットワークアクセスあり）。

    include: "girls"（既定・従来どおり） / "men" / "all"
      男子は1日約90レースあり、ガールズ(約8)の10倍以上になる。所要時間とpayloadが
      桁違いになるので、既定はガールズのままにして明示指定でのみ男子を含める。
    """
    res = fetch(build_kaisai_list_url(target.year, target.month, target.day))
    # 開催一覧は初日〜最終日の全日程を含むため、実施日が target と一致する開催日のみに絞る
    # （初日=昨日/2日目=今日 の混在を防ぐ）。
    kaisai_list = [k for k in parse_kaisai_list(res.text)
                   if kaisai_race_date(k.kaisai_day_code) == target]
    if include == "girls":
        kaisai_list = [k for k in kaisai_list if k.is_girls]
    venues = _venue_map(res.text)
    races = []
    for k in kaisai_list:
        venue = venues.get(k.kaisai_code, k.venue_code)
        girls_nos = set(fetch_girls_race_numbers(k))
        if include == "girls":
            nos = sorted(girls_nos)
        else:
            allr = set(fetch_race_numbers(k))
            nos = sorted(allr - girls_nos) if include == "men" else sorted(allr)
        for rno in nos:
            try:
                d = predict_race_dict(k.kaisai_code, k.kaisai_day_code, rno, venue=venue)
            except Exception as e:
                print(f"  {venue} R{rno} 予測失敗: {e}")
                continue
            races.append(d)
    return races


def prune_combos(races: list, keep_within_min: float, now: datetime) -> int:
    """締切が遠いレースの combos(全オッズ) を落とす。落とした数を返す。

    combos は1レースの約4割を占める最大の重量物（7車210点で約6.8KB、9車504点で約2.4倍）。
    しかし発走が何時間も先のレースのオッズは実用価値が無く、**data.json は数分ごとに
    コミットされる**ため、全レース分を載せるとリポジトリが急速に肥大する
    （実測: 既に2,707コミット・直近24hで58回push）。
    近接レースだけ残せば、ライブEVの機能は保ったまま容量と差分を大幅に抑えられる。
    """
    dropped = 0
    for r in races:
        if not r.get("combos"):
            continue
        dl, rd = r.get("deadline"), r.get("date")
        keep = False
        if dl and ":" in str(dl):
            try:
                h, m = (int(x) for x in str(dl).split(":"))
                t = now.replace(hour=h, minute=m, second=0, microsecond=0)
                if rd:
                    y = date.fromisoformat(str(rd))
                    t = t.replace(year=y.year, month=y.month, day=y.day)
                mins = (t - now).total_seconds() / 60
                keep = -30 <= mins <= keep_within_min
            except ValueError:
                keep = False
        if not keep:
            r["combos"] = []
            r["combos_pruned"] = True      # UIが「オッズ未取得」と区別できるように
            dropped += 1
    return dropped


def build_predictions_section(target: date, window: int = 1, include: str = "girls") -> dict:
    """target を起点に前後 window 日ぶんのガールズ予測を返す（既定=前後1日＝計3日）。

    レースは (date, venue, race_no) で一意。同一会場でも日をまたぐと R番号が重複するため、
    以降の突き合わせ（refresh/fetch_results/ダッシュボード）は必ず date を含めて行う。
    """
    set_default_interval(0.5)
    days = [target + timedelta(days=i) for i in range(-window, window + 1)]
    races = []
    for d in days:
        got = build_races_for_date(d, include=include)
        ng = sum(1 for r in got if r.get("is_girls"))
        print(f"  {d}: {len(got)}レース（ガールズ{ng} / 男子{len(got)-ng}）")
        races.extend(got)
    races.sort(key=lambda r: (r.get("date", ""), r.get("venue", ""), r.get("race_no") or 0))
    jst = timezone(timedelta(hours=9))
    return {
        "status": "ok" if races else "pending",
        "date": target.isoformat(),                     # 起点日（＝「今日」）
        "dates": [d.isoformat() for d in days],         # 表示対象日（昨日/今日/明日）
        "model": "LightGBM lambdarank(拡張20+Elo+展開10+並び5:中団込)",
        "note": "着順予測の確率です。EVは最新オッズ×モデルの参考値でエッジ未確立（実弾投入は非推奨）。",
        "last_updated": datetime.now(jst).strftime("%Y-%m-%d %H:%M JST"),
        "races": races,
    }


def build_model_sections(db_path: Path) -> dict:
    """収集済みデータと学習済みモデルから race_type_dist と calibration を実データ化する。"""
    model = load_model()
    samples = load_samples(db_path, features=PL_FEATURES_FULL)
    # モデルの feature_names に合わせて rel_elo / 展開特徴 を as-of 付与（共通関数・skew防止）
    from src.model.feature_augment import augment_samples
    samples = augment_samples(samples, db_path, model.feature_names)
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
    ap.add_argument("--date", help="起点日 YYYY-MM-DD（既定=今日）")
    ap.add_argument("--predict", action="store_true", help="ガールズ予測を生成（ネットワーク）")
    ap.add_argument("--window", type=int, default=1,
                    help="起点日の前後何日を含めるか（既定=1＝昨日/今日/明日の3日）")
    ap.add_argument("--include", choices=["girls", "men", "all"], default="girls",
                    help="予測対象。men/all は1日約90レースになり所要時間もpayloadも桁違い")
    ap.add_argument("--combos-within", type=float, default=90.0,
                    help="オッズ全点(combos)を残す締切までの分数。遠いレースは落として容量を抑える")
    ap.add_argument("--compact", action="store_true",
                    help="data.json をインデント無しで書く（サイズが約4割になる）")
    args = ap.parse_args()

    db_path = Path(args.db)
    doc = bdd.build(db_path)                     # data_status + pending 一式
    doc["model_ready"] = True
    doc.update(build_model_sections(db_path))    # race_type_dist / calibration を実データ化
    from verify_predictions import build_accuracy_section
    doc["prediction_accuracy"] = build_accuracy_section(db_path)   # D13: 予測実績
    from accuracy_history import build_accuracy_history
    doc["accuracy_history"] = build_accuracy_history(db_path)      # D13時系列: 週次推移
    from results_history import build_results_history
    doc["results_history"] = build_results_history(db_path, days=180)   # 成績: 過去約半年の開催結果
    if args.predict:
        target = date.fromisoformat(args.date) if args.date else date.today()
        _lbl = {"girls": "ガールズ", "men": "男子", "all": "男女全"}[args.include]
        print(f"{target} を起点に前後{args.window}日の{_lbl}予測を生成中…")
        # 確定済みレース（result あり）は**予測ごと丸ごと据え置く**。
        # 発走後にモデルが変わると「事前に出した予測」が書き換わり、的中実績が遡って
        # 良く見えてしまう。result だけ引き継いで予測を作り直すと、表示中の予測と
        # fetch_results が記録した hit の整合も崩れる。
        prev = {}
        out_path = Path(args.out)
        if out_path.exists():
            try:
                old = json.loads(out_path.read_text(encoding="utf-8"))
                for r in (old.get("predictions") or {}).get("races") or []:
                    if r.get("result"):
                        prev[(r.get("date"), r.get("venue"), r.get("race_no"))] = r
            except Exception:
                pass
        doc["predictions"] = build_predictions_section(target, window=args.window,
                                                       include=args.include)
        races, kept = [], 0
        for r in doc["predictions"]["races"]:
            old_r = prev.get((r.get("date"), r.get("venue"), r.get("race_no")))
            if old_r:                       # 確定済み→当時の予測をそのまま残す
                races.append(old_r)
                kept += 1
            else:
                races.append(r)
        doc["predictions"]["races"] = races
        if kept:
            print(f"  確定済みレースは当時の予測を据え置き: {kept}レース")
        # 締切が遠いレースのオッズ全点を落とす（容量とgit差分の抑制。機能は落ちない）
        if args.combos_within > 0:
            n_dropped = prune_combos(doc["predictions"]["races"], args.combos_within,
                                     datetime.now(timezone(timedelta(hours=9))))
            print(f"  オッズ全点を保持: 締切{args.combos_within:.0f}分以内 "
                  f"（{n_dropped}レース分を除外）")
        print(f"  予測レース数: {len(doc['predictions']['races'])}")

    out = Path(args.out)
    kw = ({"separators": (",", ":")} if args.compact else {"indent": 2})
    out.write_text(json.dumps(doc, ensure_ascii=False, **kw), encoding="utf-8")
    rt = {c["type"]: c["n"] for c in doc["race_type_dist"]["counts"]}
    print(f"生成: {out}  {out.stat().st_size/1024/1024:.2f}MB"
          f"{'（compact）' if args.compact else ''}")
    print(f"  レースタイプ分布: {rt}  / Brier: {doc['calibration']['brier']}")


if __name__ == "__main__":
    main()
