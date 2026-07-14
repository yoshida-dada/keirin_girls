"""1レースの予測を出す（方針A: 予測AIの中核）。

GambooBETのオッズページから出走表＋直近4ヶ月＋オッズを取得し、学習済みモデルで
各車の1着確率・三連単210通り確率・レースタイプを出す。オッズがあればEVも表示する。

  python scripts/predict_race.py --kaisai 6220260713 --day 62202607130100 --race 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collect.base import fetch, set_default_interval
from src.collect.gamboo_odds import build_odds_url, parse_trifecta_odds, parse_deadline
from src.collect.gamboo_racecard import parse_race_card, parse_recent_form, is_girls_race
from src.model.persist import load_model, strengths_from_model
from src.model.plackett_luce import all_trifecta_probs
from src.model.race_type import classify_race
from src.ev.ev_engine import format_combo


def predict_race_dict(kaisai_code: str, day_code: str, race_no: int,
                      venue: str = "") -> dict:
    """1レースの予測を構造化データで返す（CLI/ダッシュボード共用）。ネットワークアクセスあり。"""
    set_default_interval(0.5)
    html = fetch(build_odds_url(kaisai_code, day_code, race_no)).text
    entries = parse_race_card(html)
    recent = parse_recent_form(html)
    odds = parse_trifecta_odds(html)
    deadline = parse_deadline(html)

    model = load_model()
    strengths = strengths_from_model(model, entries, recent)
    source = "学習済みモデル(拡張20特徴)"
    if not strengths:
        from src.model.strength import strengths_from_entries
        strengths = strengths_from_entries(entries)
        source = "ベースライン(競走得点)"
    rt = classify_race(strengths)
    probs = all_trifecta_probs(strengths)

    riders = []
    for e in sorted(entries, key=lambda e: -strengths.get(e.car_number, 0)):
        f = recent.get(e.car_number)
        riders.append({
            "car": e.car_number, "name": e.rider_name,
            "score": e.racing_score, "leg": e.leg_type,
            "win_rate": (f.win_rate if f else None),
            "win_prob": round(strengths.get(e.car_number, 0), 4),
        })
    top_tri = [{"combo": format_combo(c), "prob": round(p, 4),
                "odds": odds.get(c), "need_odds": round(1 / p, 1) if p > 0 else None}
               for c, p in sorted(probs.items(), key=lambda kv: -kv[1])[:8]]
    return {
        "venue": venue, "race_no": race_no, "deadline": deadline,
        "is_girls": is_girls_race(entries), "field_size": len(entries),
        "race_type": rt.label, "top1_prob": round(rt.top1_win_prob, 4),
        "entropy": round(rt.entropy_norm, 4), "source": source,
        "riders": riders, "top_trifecta": top_tri,
        "has_odds": bool(odds),
    }


def predict(kaisai_code: str, day_code: str, race_no: int) -> None:
    """CLI表示（predict_race_dict の結果を整形して出す）。"""
    d = predict_race_dict(kaisai_code, day_code, race_no)
    print(f"\n=== R{race_no}  {'L級(ガールズ)' if d['is_girls'] else '一般'} "
          f"{d['field_size']}車  締切{d['deadline']} ===")
    print(f"レースタイプ: 【{d['race_type']}】（トップ1着{d['top1_prob']:.0%} / "
          f"エントロピー{d['entropy']:.2f}）  予測: {d['source']}")

    print(f"\n{'車':>2} {'選手':<10}{'得点':>6}{'脚質':>4}{'勝率':>6}  1着確率")
    for r in d["riders"]:
        wr = f"{r['win_rate']*100:.0f}%" if r["win_rate"] is not None else "-"
        bar = "█" * round(r["win_prob"] * 40)
        print(f"{r['car']:>2} {r['name']:<10}{r['score'] or 0:>6.1f}{r['leg']:>4}{wr:>6}"
              f"  {r['win_prob']:>5.1%} {bar}")

    print("\n三連単 モデル確率 上位:")
    print(f"{'買い目':<10}{'確率':>7}{'必要オッズ':>10}{'市場オッズ':>10}")
    for t in d["top_trifecta"]:
        o = f"{t['odds']:.1f}倍" if t["odds"] else "-"
        print(f"{t['combo']:<10}{t['prob']:>6.2%}{t['need_odds']:>9.1f}倍{o:>10}")
    print("\n※着順予測の確率です。エッジ未確立のため買い目・EVは提示しません（実弾投入は非推奨）。")


def main() -> None:
    ap = argparse.ArgumentParser(description="1レースの予測")
    ap.add_argument("--kaisai", required=True)
    ap.add_argument("--day", required=True)
    ap.add_argument("--race", type=int, required=True)
    args = ap.parse_args()
    predict(args.kaisai, args.day, args.race)


if __name__ == "__main__":
    main()
