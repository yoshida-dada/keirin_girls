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
from src.model.himo_adjust import corrected_trifecta_probs
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

    # 現時点の選手成績（通算/直近5走/当地/中何日）と対戦成績を氏名で引く（本日レースはDB外＝混ざらない）
    from config.settings import DATA_DIR
    from src.features.rider_history import current_stats, head_to_head, style_counts, meet_results
    from datetime import date as _date
    db_path = str(DATA_DIR / "keirin.sqlite")
    venue_code = kaisai_code[:2]
    try:
        stats = current_stats(db_path)
    except Exception:
        stats = {}
    car_name = {e.car_number: e.rider_name for e in entries}
    try:
        h2h = head_to_head(db_path, car_name) if stats else None
    except Exception:
        h2h = None
    try:
        styles = style_counts(db_path) if stats else {}
    except Exception:
        styles = {}
    try:  # 今場所成績（当該開催の前日までの各走）。当日自身は除外（before=当該レース実施日）。
        from src.collect.gamboo_schedule import kaisai_race_date
        _rdate = kaisai_race_date(day_code).isoformat()
        meets = meet_results(db_path, tuple(car_name.values()), kaisai_code,
                             before=_rdate) if stats else {}
    except Exception:
        meets = {}

    def _days_since(name: str) -> int | None:
        ld = (stats.get(name) or {}).get("last_date")
        if not ld:
            return None
        try:
            return (_date.today() - _date.fromisoformat(ld)).days
        except ValueError:
            return None

    model = load_model()
    _mfeats = model.feature_names or []
    elo_state = load_elo_state() if "rel_elo" in _mfeats else None
    tactics_ctx = None
    from src.features.tactics_features import TACTIC_NAMES
    if any(n in _mfeats for n in TACTIC_NAMES):    # 展開特徴付きモデル: 現時点as-of historyを引く
        from src.features.rider_tactics import current_tactics
        tactics_ctx = current_tactics(db_path)
    narabi_ctx = None
    from src.features.rider_narabi import NARABI_KEYS
    if any(n in _mfeats for n in NARABI_KEYS):     # 並び予想付きモデル: 出走表ページから即取得
        from src.collect.gamboo_racecard import parse_narabi
        narabi_ctx = parse_narabi(html)
    strengths = strengths_from_model(model, entries, recent, elo_state,
                                     tactics_ctx=tactics_ctx, narabi_ctx=narabi_ctx)
    _mtype = "LightGBM" if type(model).__name__ == "GBDTModel" else "PL線形"
    source = (f"学習済みモデル({_mtype}+Elo)" if elo_state is not None
              else f"学習済みモデル({_mtype}拡張20特徴)")
    if not strengths:
        from src.model.strength import strengths_from_entries
        strengths = strengths_from_entries(entries)
        source = "ベースライン(競走得点)"
    rt = classify_race(strengths)
    # 条件付き紐補正: 2着分布を平坦化(PLの○過大評価是正)＋◎の並び番手を加点（精度改善, himo_adjust）。
    # 並び予想があれば {車番: 隊列位置} を渡す。無ければ温度平坦化のみ適用。
    narabi_pos = ({car: i for i, car in enumerate(narabi_ctx["order"])}
                  if narabi_ctx and narabi_ctx.get("order") else None)
    probs = corrected_trifecta_probs(strengths, narabi_pos)

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
        st = stats.get(e.rider_name) or {}
        cwr = st.get("career_win_rate")
        r5 = st.get("recent5_avg_finish")
        vwr = (st.get("venue") or {}).get(venue_code)
        vst = (st.get("venue_starts") or {}).get(venue_code)
        sc = styles.get(e.rider_name) or {}
        mr = meets.get(e.rider_name) or []
        from src.features.venue_region import is_home_pref, is_home_district
        riders.append({
            "car": e.car_number, "name": e.rider_name,
            "score": e.racing_score, "leg": e.leg_type,
            "pref": (e.prefecture or "").strip() or None,        # 登録府県
            "home": is_home_pref(e.prefecture, venue_code),      # 地元(同県)開催か
            "home_dist": is_home_district(e.prefecture, venue_code),  # 同地区開催か
            "win_rate": (f.win_rate if f else None),
            "win_prob": wp,
            "synth_odds_1st": synth,                         # 一着固定の合成オッズ
            "fair_odds_1st": round(1 / wp, 2) if wp > 0 else None,  # モデル勝率の必要オッズ
            "win_ev": round(wp * synth, 2) if synth else None,     # >1=市場が1着を過小評価
            # 収集済み全履歴からの現時点成績（as-of最新, 本日レースは含まない）
            "career_win_rate": round(cwr, 4) if cwr is not None else None,
            "career_starts": st.get("career_starts"),
            "recent5_finish": round(r5, 2) if r5 is not None else None,
            "venue_win_rate": round(vwr, 4) if vwr is not None else None,
            "venue_starts": vst,
            "days_since": _days_since(e.rider_name),
            # 脚質プロファイル（直近1年）: S/B回数と1着決まり手(逃/捲/差)回数
            "s_cnt": sc.get("s"), "b_cnt": sc.get("b"),
            "nige": sc.get("nige"), "makuri": sc.get("makuri"), "sashi": sc.get("sashi"),
            "style_races": sc.get("races"),
            # 今場所成績: [日付, R番号, 着順, 上りタイム] の配列（前日までの各走）
            "meet": [[m["date"], m["race_no"], m["position"], m["last_lap"]] for m in mr],
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

    # 参考フォーメーション（◎頭固定・補正確率top-K）。実弾非推奨・回収率<100%（黒字ゾーンは無い）。
    from src.betting.reference_formation import build_reference
    reference = build_reference(strengths, narabi_pos, venue_code,
                               bool(riders and riders[0].get("home")))

    # 展開予想（記者の並び予想の隊列＋モデルの一言読み）。ガールズは並び通りになるとは限らない。
    development = None
    if narabi_ctx and narabi_ctx.get("order"):
        _order = narabi_ctx["order"]
        _legs = narabi_ctx.get("legs") or {}
        _fav = max(strengths, key=strengths.get) if strengths else None
        _fpos = narabi_pos.get(_fav) if narabi_pos else None
        _marker = None
        if _fpos is not None:
            for _c, _pp in (narabi_pos or {}).items():
                if _c != _fav and _pp == _fpos + 1:
                    _marker = _c
                    break
        line = [{"car": c, "leg": _legs.get(c), "is_fav": (c == _fav), "pos": i}
                for i, c in enumerate(_order)]
        # モデルの一言展開読み（himo知見: ◎主導権→番手が連れる／◎飛び→中団の自力型）
        if _fpos == 0:
            read = f"◎{_fav}番が予想先頭＝主導権を握る展開が本線。"
            if _marker:
                read += f"番手{_marker}番が連れて2着有力。"
        elif _fpos is not None:
            read = f"◎{_fav}番は隊列{_fpos+1}番手。前の{_order[0]}番の主導権をどう捉えるかがカギ。"
        else:
            read = f"◎{_fav}番の並び位置は不明。"
        read += "◎が飛ぶ場合は中団の自力型(捲り)が抜ける展開に注意。"
        development = {
            "source": "並び予想（記者予想の隊列, 発走前確定情報）",
            "line": line, "fav": _fav, "marker": _marker,
            "note": read,
            "caveat": "ガールズはライン概念が薄く、実際の主導権は並び予想通りにならないことも多い（予想先頭の実バック取得率≒20%）。",
        }

    from datetime import datetime, timezone, timedelta
    jst = timezone(timedelta(hours=9))
    return {
        "venue": venue, "race_no": race_no, "deadline": deadline,
        "is_girls": is_girls_race(entries), "field_size": len(entries),
        "race_type": rt.label, "top1_prob": round(rt.top1_win_prob, 4),
        "entropy": round(rt.entropy_norm, 4), "source": source,
        "riders": riders, "top_trifecta": top_tri, "ev": ev,
        "development": development,                 # 展開予想（並び予想の隊列＋モデル読み）
        "combos": combos,                          # 全210通り（オッズテーブル用）
        "reference": reference,                    # 参考フォーメーション（実弾非推奨・回収率<100%）
        "h2h": h2h,                                # 出走者同士の過去対戦成績マトリクス
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
