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
    build_kaisai_list_url, parse_kaisai_list, fetch_race_numbers_for, kaisai_race_date,
)
from predict_race import predict_race_dict
from build_predictions import _venue_map
from src.collect.snapshot import build_race_id
from db.repository import SnapshotRepo

DEFAULT_OUT = ROOT / "dashboard" / "data.json"
SNAPSHOT_DB = ROOT / "data" / "odds_snapshots.sqlite"   # オッズ時系列(haircut/変動特徴の土台)
JST = timezone(timedelta(hours=9))


def _store_snapshot(repo, race_id: str, race: dict, now: datetime) -> None:
    """予測レースdictの combos から確定前オッズを取り出し、取得時刻付きで時系列保存する。"""
    if repo is None:
        return
    odds = {(c[0], c[1], c[2]): c[3] for c in race.get("combos", []) if c[3] is not None}
    if odds:
        try:
            repo.save_snapshot(race_id, odds, now)
        except Exception:
            pass


def _minutes_to_deadline(deadline: str, now: datetime, race_date: str | None = None) -> float | None:
    """締切(HH:MM, JST)までの分。過ぎていれば負。

    前後1日を同時に持つため、レースの日付を必ず加味する（日付を無視すると昨日/明日の
    レースが「今日の同時刻」と誤判定され、締切直前のライブ更新対象に紛れ込む）。
    """
    if not deadline or ":" not in deadline:
        return None
    h, m = (int(x) for x in deadline.split(":"))
    dl = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if race_date:
        try:
            d = date.fromisoformat(str(race_date))
            dl = dl.replace(year=d.year, month=d.month, day=d.day)
        except ValueError:
            return None
    return (dl - now).total_seconds() / 60.0


def main() -> None:
    ap = argparse.ArgumentParser(description="最新オッズで予測EVを更新")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--out-men", help="男子の書き出し先（既定=--out と同じ場所の data_men.json）")
    ap.add_argument("--only-near", type=float,
                    help="締切まで N 分以内のレースだけ更新（Actions定期実行用）")
    ap.add_argument("--include", choices=["girls", "men", "all"], default="girls",
                    help="更新対象（既定=girls）。data.json に男子を載せるなら all にする")
    ap.add_argument("--men-only-near", type=float,
                    help="男子だけ別の窓（分）にする。未指定なら --only-near と同じ。"
                         "男子は同時進行の会場数が多く、広い窓で1分回すと取得数が跳ね上がる")
    args = ap.parse_args()
    set_default_interval(0.5)

    # 男女でファイルを分ける（ダッシュボードも別ページ）。締切窓の判定は既存の
    # data.json 側の締切キャッシュを使うので、読み込みは両方まとめて行う。
    out = Path(args.out)
    out_men = Path(args.out_men) if args.out_men else out.with_name("data_men.json")
    doc = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    doc_men = json.loads(out_men.read_text(encoding="utf-8")) if out_men.exists() else {}
    now = datetime.now(JST)
    target = now.date()
    try:
        snap_repo = SnapshotRepo(SNAPSHOT_DB)     # オッズ時系列を蓄積（haircut/変動特徴の土台）
    except Exception:
        snap_repo = None

    res = fetch(build_kaisai_list_url(target.year, target.month, target.day))
    kaisai_list = [k for k in parse_kaisai_list(res.text)
                   if kaisai_race_date(k.kaisai_day_code) == target]
    if args.include == "girls":
        kaisai_list = [k for k in kaisai_list if k.is_girls]
    venues = _venue_map(res.text)

    # 既存data.jsonの締切をキャッシュし、窓外レースは取得自体をスキップ（1分ループを軽くする）
    # キーは (date, venue, race_no)。日付を外すと同一会場のR番号が日をまたいで衝突する。
    tstr = target.isoformat()
    known = {(r.get("date") or tstr, r.get("venue"), r.get("race_no")): r
             for d in (doc, doc_men)
             for r in d.get("predictions", {}).get("races", [])}
    dl_cache = {k2: r.get("deadline") for k2, r in known.items()}

    men_near = args.men_only_near if args.men_only_near is not None else args.only_near

    def _plan(k) -> list[int]:
        """この開催で今回見るレース番号。

        --only-near のときは**レース一覧ページを取りに行かない**。朝のビルドが data.json に
        全レースを入れてあるので、そこから開催中の会場のR番号を復元すれば足りる。
        毎分×十数会場ぶんのレース一覧フェッチ（男子込みで1日1万回超）を丸ごと省ける。
        """
        v = venues.get(k.kaisai_code, k.venue_code)
        if args.only_near is None:
            return fetch_race_numbers_for(k, args.include)
        nos = sorted(no for (d, vv, no) in known if d == tstr and vv == v and no is not None)
        return nos or fetch_race_numbers_for(k, args.include)   # 朝のビルドが無い日は従来通り

    races = []
    for k in kaisai_list:
        venue = venues.get(k.kaisai_code, k.venue_code)
        for rno in _plan(k):
            r0 = known.get((tstr, venue, rno))
            if args.include == "men" and r0 is not None and r0.get("is_girls"):
                continue
            # 窓は男女で分けられる（男子は同時開催が多く、広い窓だと1分あたりの取得数が跳ねる）
            near = men_near if (r0 is not None and r0.get("is_girls") is False) else args.only_near
            if near is not None:
                dl = dl_cache.get((tstr, venue, rno))
                if dl:                                   # 締切既知→窓外なら取得しない
                    m = _minutes_to_deadline(dl, now, tstr)
                    if m is None or m < -5 or m > near:
                        continue
            try:
                d = predict_race_dict(k.kaisai_code, k.kaisai_day_code, rno, venue=venue)
            except Exception as e:
                print(f"  {venue} R{rno} 失敗: {e}")
                continue
            if near is not None:                         # 取得後の締切で最終判定
                mins = _minutes_to_deadline(d.get("deadline", ""), now, d.get("date"))
                if mins is None or mins < -5 or mins > near:
                    continue
            _store_snapshot(snap_repo, build_race_id(k.kaisai_day_code, rno), d, now)
            races.append(d)

    def _key(r: dict) -> tuple:
        return (r.get("date") or tstr, r.get("venue") or "", r.get("race_no") or 0)

    dates = [(target + timedelta(days=i)).isoformat() for i in (-1, 0, 1)]

    def _write(path: Path, d: dict, got: list) -> None:
        """既存レースを保持しつつ今回取得ぶんを差し替えて書く。取得0なら既存のまま。"""
        if not d.get("predictions") and not got:
            return                       # そのファイル自体が無い（例: ガールズ非開催日）
        if d.get("predictions", {}).get("races"):
            # 既存レース（前後1日を含む）を保持し、今回取得したぶんだけ差し替えマージ。
            # only_near 指定の有無にかかわらず、他日のレースを消さない。
            merged = {_key(r): r for r in d["predictions"]["races"]}
            for r in got:
                merged[_key(r)] = r
            got = sorted(merged.values(), key=_key)
        d.setdefault("predictions", {})
        d["predictions"].update({
            "status": "ok" if got else d["predictions"].get("status", "pending"),
            "date": target.isoformat(),
            # 表示対象日（前後1日）。日付が変わっても morning build を待たずに追従させる。
            "dates": dates,
            # model は build_predictions が入れた正しい値を尊重する（ここで上書きしない）
            "model": d["predictions"].get("model") or "LightGBM lambdarank",
            "note": "着順予測の確率です。EVは最新オッズ×モデルの参考値でエッジ未確立（実弾投入は非推奨）。",
            "last_updated": now.strftime("%Y-%m-%d %H:%M JST"),
            "races": got if got else d["predictions"].get("races", []),
        })
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"更新: {path}  レース{len(d['predictions']['races'])}  {now:%H:%M JST}")

    _write(out, doc, [r for r in races if r.get("is_girls") is not False])
    _write(out_men, doc_men, [r for r in races if r.get("is_girls") is False])


if __name__ == "__main__":
    main()
