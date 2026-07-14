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
from src.model.persist import load_model, strengths_from_model, load_elo_state
from src.model.plackett_luce import all_trifecta_probs
from src.model.race_type import classify_race
from src.ev.market import implied_trifecta_probs, blend_loglinear
from src.ev.ev_engine import build_trifecta_ev_table, format_combo


def predict_race_dict(kaisai_code: str, day_code: str, race_no: int,
                      venue: str = "") -> dict:
    """1レースの予測を構造化データで返す（CLI/ダッシュボード共用）。ネットワークアクセスあり。"""
    set_default_interval(0.5)
    html = fetch(build_odds_url(kaisai_code, day_code, race_no)).text
    entries = parse_race_card(html)
    recent = parse_recent_form(html)
    odds = parse_trifecta_odds(html)
    # 9999.9 はGambooBETの表示上限（実質オッズなし＝ほぼ無投票）なので除外する。
    odds = {k: v for k, v in odds.items() if v and v < 9999}
    deadline = parse_deadline(html)

    model = load_model()
    elo_state = load_elo_state() if "rel_elo" in (model.feature_names or []) else None
    strengths = strengths_from_model(model, entries, recent, elo_state)
    source = "学習済みモデル(拡張+Elo)" if elo_state is not None else "学習済みモデル(拡張20特徴)"
    if not strengths:
        from src.model.strength import strengths_from_entries
        strengths = strengths_from_entries(entries)
        source = "ベースライン(競走得点)"
    rt = classify_race(strengths)
    probs = all_trifecta_probs(strengths)

    # 一着固定の合成オッズ: 車cを1着に固定した三連単(c,*,*)全通りを合成した実効オッズ
    #   合成オッズ_c = 1 / Σ(1/オッズ)   … cを1着で買い切ったときの実効配当倍率
    # モデル勝率 win_prob と突き合わせると市場が各車の「勝ち」を割高/割安に見ているか分かる。
    #   win_ev = win_prob × 合成オッズ（>1でモデル的に割安=1着を過小評価）
    def _synth_1st(car: int) -> float | None:
        inv = sum(1.0 / o for k, o in odds.items() if k[0] == car and o and o > 0)
        return round(1.0 / inv, 2) if inv > 0 else None

    riders = []
    for e in sorted(entries, key=lambda e: -strengths.get(e.car_number, 0)):
        f = recent.get(e.car_number)
        wp = round(strengths.get(e.car_number, 0), 4)
        synth = _synth_1st(e.car_number) if odds else None
        riders.append({
            "car": e.car_number, "name": e.rider_name,
            "score": e.racing_score, "leg": e.leg_type,
            "win_rate": (f.win_rate if f else None),
            "win_prob": wp,
            "synth_odds_1st": synth,                         # 一着固定の合成オッズ
            "fair_odds_1st": round(1 / wp, 2) if wp > 0 else None,  # モデル勝率の必要オッズ
            "win_ev": round(wp * synth, 2) if synth else None,     # >1=市場が1着を過小評価
        })
    top_tri = [{"combo": format_combo(c), "prob": round(p, 4),
                "odds": odds.get(c), "need_odds": round(1 / p, 1) if p > 0 else None}
               for c, p in sorted(probs.items(), key=lambda kv: -kv[1])[:8]]

    # オッズテーブル用の全210通り: [1着,2着,3着,オッズ,確率,EV]（EV=確率×オッズ, 1超で妙味）。
    # ダッシュボードは1着車で絞って 2着×3着 マトリクスを描き、EV>1 をハイライトする。
    combos = []
    for (a, b, c), p in probs.items():
        o = odds.get((a, b, c))
        ev = round(p * o, 2) if o else None
        combos.append([a, b, c, o, round(p, 5), ev])

    # 最新オッズに基づくEV判定（発走10分前更新で使う）。エッジ未確立のため参考値。
    ev = {"status": "no_odds", "threshold": 1.10, "n_buy": 0, "buys": [],
          "note": "最新オッズ×モデル確率のEV参考値。エッジ未確立のため実弾投入は非推奨。"}
    if odds:
        implied = implied_trifecta_probs(odds)
        blended = blend_loglinear(probs, implied, alpha=0.8)
        table = build_trifecta_ev_table(
            blended, odds, ev_threshold=1.10,
            guards={"shrink_to_market": 0.0, "min_prob": 0.005, "max_odds": 500.0})
        buys = [{"combo": format_combo(r.combo), "prob": round(r.model_prob, 4),
                 "odds": r.odds, "ev": round(r.ev_gross, 2)} for r in table["buy"][:8]]
        ev.update(status="ok", n_buy=len(table["buy"]), buys=buys)

    from datetime import datetime, timezone, timedelta
    jst = timezone(timedelta(hours=9))
    return {
        "venue": venue, "race_no": race_no, "deadline": deadline,
        "is_girls": is_girls_race(entries), "field_size": len(entries),
        "race_type": rt.label, "top1_prob": round(rt.top1_win_prob, 4),
        "entropy": round(rt.entropy_norm, 4), "source": source,
        "riders": riders, "top_trifecta": top_tri, "ev": ev,
        "combos": combos,                          # 全210通り（オッズテーブル用）
        "has_odds": bool(odds),
        "updated_at": datetime.now(jst).strftime("%Y-%m-%d %H:%M"),
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
