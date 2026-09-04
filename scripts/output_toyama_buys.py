"""本日の富山競輪 全レースの荒れ用買い目（効率=mix18 / バランス=mix24）と結果を出す。

条件分岐: 万車券率<30%=現行(build_branches統合), ≥30%=mixで広げた荒れ買い目。
効率=荒れをmix18点、バランス=荒れをmix24点。結果(実払戻)で的中判定。本番モデル。

  PYTHONIOENCODING=utf-8 python scripts/output_toyama_buys.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.collect.base import fetch, set_default_interval
from src.collect.gamboo_schedule import (build_kaisai_list_url, parse_kaisai_list,
                                         kaisai_race_date, fetch_race_numbers_for)
from src.collect.gamboo_result import fetch_result
from scripts.build_predictions import _venue_map
from predict_race import predict_race_dict

THR = 0.30
ARARE_EFF, ARARE_BAL = 18, 24


def _merged_from_dict(dev):
    cs = set()
    for f in ((dev or {}).get("merged") or {}).get("forms", []):
        cs |= {(a, b, c) for a in f["first"] for b in f["second"] for c in f["third"]
               if len({a, b, c}) == 3}
    return cs


def _mix_ranked(combos):
    # combos = [[a,b,c,odds,prob,ev], ...] prob は本番分布(mix/himo)
    good = [(int(c[0]), int(c[1]), int(c[2]), c[4] or 0) for c in combos if c[4] is not None]
    return [(a, b, c) for a, b, c, _ in sorted(good, key=lambda x: -x[3])]


def _fmt(combos, k=None):
    xs = list(combos)[:k] if k else list(combos)
    return " ".join(f"{a}-{b}-{c}" for a, b, c in xs)


def main():
    set_default_interval(0.5)
    today = date.today()
    res = fetch(build_kaisai_list_url(today.year, today.month, today.day))
    ven = _venue_map(res.text)
    ks = [k for k in parse_kaisai_list(res.text)
          if kaisai_race_date(k.kaisai_day_code) == today
          and "富山" in (ven.get(k.kaisai_code, "") or "") and not k.is_girls]
    if not ks:
        print("本日 富山の男子開催なし"); return
    k = ks[0]
    v = ven.get(k.kaisai_code, "富山競輪")
    print(f"{v} {today}（開催{k.kaisai_day_code[10:12]}日目）\n"
          f"条件分岐: 万車券率<30%=現行 / ≥30%=荒れ買い目（効率mix{ARARE_EFF}点 / バランスmix{ARARE_BAL}点）\n")

    tot = {"eff": {"pts": 0, "hit": 0, "ret": 0}, "bal": {"pts": 0, "hit": 0, "ret": 0},
           "cur": {"pts": 0, "hit": 0, "ret": 0}, "n": 0, "arare": 0, "done": 0}
    for rno in fetch_race_numbers_for(k, "men"):
        try:
            d = predict_race_dict(k.kaisai_code, k.kaisai_day_code, rno, venue=v)
        except Exception as e:
            print(f"R{rno} 失敗: {e}"); continue
        u = d.get("upset_prob")
        current = _merged_from_dict(d.get("dev_branches"))
        mixr = _mix_ranked(d.get("combos") or [])
        role = (d.get("race_name") or d.get("race_type") or "")
        arare = (u is not None and u >= THR)
        eff = set(mixr[:ARARE_EFF]) if arare else set(current)
        bal = set(mixr[:ARARE_BAL]) if arare else set(current)
        # 結果
        winner = pay = None
        try:
            rows, payout = fetch_result(k.kaisai_code, k.kaisai_day_code, rno)
            fin = sorted([r for r in rows if r.position in (1, 2, 3)], key=lambda r: r.position)
            if len(fin) == 3:
                winner = (fin[0].car_number, fin[1].car_number, fin[2].car_number)
            if payout:
                pay = payout.payout
        except Exception:
            pass
        utxt = f"{u*100:.0f}%" if u is not None else "—"
        tag = "【荒れ】" if arare else "【堅】"
        rtxt = (f"結果 {winner[0]}-{winner[1]}-{winner[2]}"
                + (f" ({pay:,}円)" if pay else "")) if winner else "未確定"
        print(f"R{rno} [{role}] 万車券率{utxt} {tag}  {rtxt}")
        tot["n"] += 1
        if arare:
            tot["arare"] += 1
        if arare:
            print(f"   効率(mix{ARARE_EFF}, {len(eff)}点): {_fmt(mixr, ARARE_EFF)}")
            print(f"   バランス(mix{ARARE_BAL}, {len(bal)}点): {_fmt(mixr, ARARE_BAL)}")
        else:
            print(f"   現行({len(current)}点): {_fmt(sorted(current))}")
        if winner:
            tot["done"] += 1
            for key, s in (("eff", eff), ("bal", bal), ("cur", set(current))):
                tot[key]["pts"] += len(s)
                if tuple(winner) in s:
                    tot[key]["hit"] += 1
                    tot[key]["ret"] += pay or 0
            mk = lambda s: "○" if tuple(winner) in s else "×"
            print(f"   → 効率{mk(eff)} / バランス{mk(bal)} / 現行(参考){mk(set(current))}")
        print()

    print("=" * 56)
    print(f"【集計】{tot['n']}レース（荒れ{tot['arare']} / 確定{tot['done']}）")
    def line(nm, key):
        t = tot[key]
        roi = t["ret"] / (t["pts"] * 100) * 100 if t["pts"] else 0
        print(f"  {nm:<12} 総点数{t['pts']:>4} 的中{t['hit']}/{tot['done']} 払戻{t['ret']:>7,}円 回収率{roi:.1f}%")
    line("効率(mix18)", "eff")
    line("バランス(mix24)", "bal")
    line("現行(参考)", "cur")


if __name__ == "__main__":
    main()
