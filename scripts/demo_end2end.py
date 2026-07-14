"""実レース End2End デモ（S1→S3→S4を1本通す）。

1つのガールズ(L級)レースについて、GambooBET出走表 → 各選手の登録番号を競輪ステーションで
名前解決 → 直近成績 → 競走得点ベースの強さ → Plackett-Luce 210通り → レースタイプ分類 →
確定前オッズ → EVテーブル まで通しで動かし、全部品が繋がることを可視化する。

  # 当日のガールズ戦を自動探索して実行:
  python scripts/demo_end2end.py
  # レースを指定（負荷を抑えたいとき）:
  python scripts/demo_end2end.py --kaisai 6220260713 --day 62202607130100 --race 1
  # 選手成績の名前解決をスキップ（出走表の競走得点だけで動かす）:
  python scripts/demo_end2end.py --no-stats
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collect.gamboo_schedule import fetch_girls_kaisai, fetch_race_numbers, Kaisai
from src.collect.gamboo_racecard import fetch_race_card, is_girls_race
from src.collect.gamboo_odds import fetch_trifecta_odds
from src.collect.keirin_station import resolve_rider_id, fetch_player_detail
from src.model.strength import strengths_from_entries
from src.model.plackett_luce import all_trifecta_probs
from src.model.race_type import classify_race
from src.ev.market import implied_trifecta_probs, blend_loglinear
from src.ev.ev_engine import build_trifecta_ev_table, format_combo


def find_first_girls_race(target: date, scan_per_kaisai: int = 6):
    """当日のガールズ開催を走査し、最初のL級レースを返す。"""
    for k in fetch_girls_kaisai(target.year, target.month, target.day):
        for rno in fetch_race_numbers(k)[:scan_per_kaisai]:
            entries = fetch_race_card(k.kaisai_code, k.kaisai_day_code, rno)
            if is_girls_race(entries):
                return k, rno, entries
    return None, None, None


def run(kaisai: Kaisai, race_no: int, entries, use_stats: bool) -> None:
    print(f"\n=== 会場{kaisai.venue_code} R{race_no}  ({len(entries)}車 L級) ===\n")

    # --- 出走表 ＋（任意で）選手成績の名前解決 ---
    stats = {}
    if use_stats:
        for e in entries:
            try:
                rid = resolve_rider_id(e.rider_name, e.prefecture, e.term)
                stats[e.car_number] = fetch_player_detail(rid) if rid else None
            except Exception:
                stats[e.car_number] = None

    print(f"{'車':>2} {'選手':<10}{'得点':>6}{'脚質':>4}{'登録番号':>9}{'勝率':>7}{'逃/捲/差/マ':>14}")
    for e in entries:
        ps = stats.get(e.car_number)
        rid = ps.rider_id if ps else "-"
        wr = f"{ps.win_rate*100:.0f}%" if ps and ps.win_rate is not None else "-"
        km = (f"{ps.escape_rate:.0%}/{ps.dash_rate:.0%}/{ps.closing_rate:.0%}/{ps.mark_rate:.0%}"
              if ps and ps.escape_rate is not None else "-")
        print(f"{e.car_number:>2} {e.rider_name:<10}{e.racing_score or 0:>6.1f}{e.leg_type:>4}"
              f"{rid:>9}{wr:>7}{km:>14}")

    # --- 強さ(競走得点ベース) → PL210通り → レースタイプ ---
    strengths = strengths_from_entries(entries)
    model = all_trifecta_probs(strengths)
    rt = classify_race(strengths)
    print(f"\n強さ(1着確率): " + "  ".join(f"{c}:{p:.0%}" for c, p in
                                        sorted(strengths.items(), key=lambda x: -x[1])))
    print(f"レースタイプ: 【{rt.label}】 トップ1着{rt.top1_win_prob:.0%} / "
          f"エントロピー{rt.entropy_norm:.2f}")

    # --- オッズ → 市場ブレンド → EV ---
    odds, deadline = fetch_trifecta_odds(kaisai.kaisai_code, kaisai.kaisai_day_code, race_no)
    if not odds:
        print(f"\nオッズ未取得（締切{deadline}）。EV算出はスキップ。")
        return
    implied = implied_trifecta_probs(odds)
    blended = blend_loglinear(model, implied, alpha=0.8)
    guards = {"shrink_to_market": 0.0, "min_prob": 0.005, "max_odds": 500.0}
    table = build_trifecta_ev_table(blended, odds, ev_threshold=1.10, guards=guards)
    print(f"\nオッズ{len(odds)}点（締切{deadline}） / 購入候補{len(table['buy'])}点（EV≥1.10）")
    print(f"{'買い目':<10}{'モデル%':>8}{'オッズ':>9}{'EV':>7}")
    for r in table["buy"][:10]:
        print(f"{format_combo(r.combo):<10}{r.model_prob*100:>7.2f}{r.odds:>9.1f}{r.ev_gross:>7.2f}")
    if not table["buy"]:
        top = table["all"][:5]
        print("（+EVなし。EV上位: " + ", ".join(f"{format_combo(r.combo)}={r.ev_gross:.2f}" for r in top) + "）")


def main() -> None:
    ap = argparse.ArgumentParser(description="実レース End2End デモ")
    ap.add_argument("--date")
    ap.add_argument("--kaisai"); ap.add_argument("--day"); ap.add_argument("--race", type=int)
    ap.add_argument("--no-stats", action="store_true", help="選手成績の名前解決をスキップ")
    args = ap.parse_args()
    target = date.fromisoformat(args.date) if args.date else date.today()

    if args.kaisai and args.day and args.race:
        kaisai = Kaisai(args.kaisai, args.day, args.kaisai[:2], True)
        entries = fetch_race_card(kaisai.kaisai_code, kaisai.kaisai_day_code, args.race)
        race_no = args.race
        if not is_girls_race(entries):
            print("※指定レースはL級ではありません（デモは続行）")
    else:
        print("当日のガールズ(L級)レースを探索中…")
        kaisai, race_no, entries = find_first_girls_race(target)
        if kaisai is None:
            print("本日のガールズL級レースが見つかりませんでした。")
            return
    run(kaisai, race_no, entries, use_stats=not args.no_stats)


if __name__ == "__main__":
    main()
