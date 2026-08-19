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
from src.collect.gamboo_odds import (
    build_odds_url, parse_trifecta_odds, parse_deadline, parse_race_meta)
from src.collect.gamboo_racecard import parse_race_card, parse_recent_form, is_girls_race
from src.model.persist import load_model, strengths_from_model, load_elo_state
from src.model.plackett_luce import all_trifecta_probs
from src.model.himo_adjust import corrected_trifecta_probs
from src.model.race_type import classify_race
from src.model.upset import man_prob
from src.ev.market import implied_trifecta_probs, blend_loglinear
from src.ev.ev_engine import build_trifecta_ev_table, format_combo


def predict_race_dict(kaisai_code: str, day_code: str, race_no: int,
                      venue: str = "") -> dict:
    """1レースの予測を構造化データで返す（CLI/ダッシュボード共用）。ネットワークアクセスあり。"""
    set_default_interval(0.5)
    # オッズは下でこのHTMLから読む。**取得時刻はフェッチの瞬間で取る**。
    # 締切までの残り時間でオッズは動くので、いつ時点の値かを表示に出さないと
    # 合成オッズが何を指しているか読めない（updated_at は処理完了時刻で別物）。
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    odds_at = _dt.now(_tz(_td(hours=9))).strftime("%H:%M")
    html = fetch(build_odds_url(kaisai_code, day_code, race_no)).text
    entries = parse_race_card(html)
    recent = parse_recent_form(html)
    odds = parse_trifecta_odds(html)
    # 9999.9 はGambooBETの表示上限（実質オッズなし＝ほぼ無投票）なので除外する。
    odds = {k: v for k, v in odds.items() if v and v < 9999}
    deadline = parse_deadline(html)
    meta = parse_race_meta(html)      # 開催格/開催名/レース名（同じHTMLから。追加フェッチなし）
    # レース種別(勝ち上がり): race_name(例"Ａ級予選"/"Ｓ級決勝")から種別語を判定。追加フェッチ無し。
    _rn = meta.get("race_name")
    _rrole = next((r for r in ["準決勝", "決勝", "予選", "選抜", "特選", "一般"]
                   if _rn and r in _rn), None)
    race_role = {"class_role": _rn, "role": _rrole, "grade": meta.get("grade")}
    # 翌日ぶんは出走表が未公開のことがある（前日夕方に順次published）。空のまま進むと
    # 特徴量組み立てが "None of ['car_number'] are in the columns" で落ち、原因が読めない。
    if not entries:
        raise ValueError("出走表が未公開（翌日ぶんの可能性）")

    # 現時点の選手成績（通算/直近5走/当地/中何日）と対戦成績を氏名で引く（本日レースはDB外＝混ざらない）
    from config.settings import DATA_DIR
    from src.features.rider_history import current_stats, head_to_head, style_counts, meet_results
    from datetime import date as _date
    db_path = str(DATA_DIR / "keirin.sqlite")
    venue_code = kaisai_code[:2]
    # 選手成績は氏名でDBを引く。**男子はガールズDBに1走も入っていない**ので、ここで
    # DBを切り替えないと通算/直近5走/当地/S/B/決まり手が全員 None になり表が空になる。
    _girls = is_girls_race(entries)
    men_db = str(DATA_DIR / "keirin_men.sqlite")
    hist_db = db_path if _girls else (men_db if Path(men_db).exists() else db_path)
    try:
        stats = current_stats(hist_db)
    except Exception:
        stats = {}
    car_name = {e.car_number: e.rider_name for e in entries}
    try:
        h2h = head_to_head(hist_db, car_name) if stats else None
    except Exception:
        h2h = None
    try:
        styles = style_counts(hist_db) if stats else {}
    except Exception:
        styles = {}
    try:  # 今場所成績（当該開催の前日までの各走）。当日自身は除外（before=当該レース実施日）。
        from src.collect.gamboo_schedule import kaisai_race_date
        _rdate = kaisai_race_date(day_code).isoformat()
        meets = meet_results(hist_db, tuple(car_name.values()), kaisai_code,
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

    # レース種別でモデルを切り替える。特徴セットが違う（ガールズ38列 / 男子39列）ので、
    # 取り違えると無言で誤った推論になる。男子モデルが無ければフォールバックせず落とす。
    from src.model.feature_sets import load_for
    model, elo_state, _mlabel = load_for(_girls)
    # 展開特徴(as-of history)と Elo も選手成績と同じ母集団から引く
    feat_db = hist_db

    # 後続（展開予想・紐補正・展開AI）でも使うので、モデルの有無に関わらず先に用意する
    from src.collect.gamboo_racecard import parse_narabi
    narabi_ctx = parse_narabi(html)
    tactics_ctx = None
    strengths = {}
    _mfeats = (model.feature_names or []) if model is not None else []
    if model is not None:
        if "rel_elo" not in _mfeats:
            elo_state = None
        from src.features.tactics_features import TACTIC_NAMES
        if any(n in _mfeats for n in TACTIC_NAMES):   # 展開特徴付き: 現時点as-of historyを引く
            from src.features.rider_tactics import current_tactics
            tactics_ctx = current_tactics(feat_db)
        strengths = strengths_from_model(model, entries, recent, elo_state,
                                         tactics_ctx=tactics_ctx, narabi_ctx=narabi_ctx,
                                         venue_code=venue_code)
    _mtype = "LightGBM" if type(model).__name__ == "GBDTModel" else "PL線形"
    source = f"学習済みモデル({_mlabel}・{_mtype} {len(_mfeats)}特徴)"
    if not strengths:
        from src.model.strength import strengths_from_entries
        strengths = strengths_from_entries(entries)
        source = f"ベースライン(競走得点)※{_mlabel}モデルで推論できず"
    rt = classify_race(strengths)
    # 条件付き紐補正: 2着分布を平坦化(PLの○過大評価是正)＋◎の並び番手を加点（精度改善, himo_adjust）。
    # 並び予想があれば {車番: 隊列位置} を渡す。無ければ温度平坦化のみ適用。
    # 男子は A-3 で再推定済み（hold-out 5,498R: 三連単log-loss -0.104 / 2着top1 +1.6pt）。
    # 番手は**記者の並び予想のライン基準**で判定する（隊列の直後だと次ラインの先頭＝敵を拾う）。
    from src.model.stats_profile import profile as _stats_profile
    _sp = _stats_profile(_girls)
    narabi_pos = ({car: i for i, car in enumerate(narabi_ctx["order"])}
                  if narabi_ctx and narabi_ctx.get("order") else None)
    # 番手は**記者の並び予想のライン構成**で判定する（男子）。隊列の直後で判定すると
    # ラインの最後尾で次ラインの先頭＝敵を番手扱いしてしまう（marker_of 参照）。
    _narabi_lines = (narabi_ctx or {}).get("lines") or None
    # パラメータは性別ごと（ガールズ DEFAULT_PARAMS / 男子 MEN_PARAMS）。取り違えると
    # 無言で別の補正が掛かるので、必ず stats_profile から受け取る。
    # ---- 本表示の三連単分布 ----
    # **分岐混合 Σ P(B=b)·P(順位|B=b) を第一候補にする**（2026-08-18 配線）。
    # 紐補正は着順ごとの周辺重みの調整で「同一ラインが揃って上位」という同時共起を作れず、
    # ライン決着を実測55.6%に対し35.3%と20pt過小に出していた。混合は56.8%（+1.2pt）で、
    # tri10 も 35.23→41.08% と改善する（scripts/validate_joint.py）。
    # ラインが無い（ガールズ・並び予想なし）／展開AIが引けない場合は紐補正へフォールバック。
    _pB = None                      # 展開AIの P(B)。後段の backstretch 表示でも使い回す
    _dists = None                   # 分岐ごとの分布。build_branches へ渡して二重計算を避ける
    _mix = {}
    if not _girls and _narabi_lines and strengths:
        from src.model.backstretch import load_backstretch
        _bs_model = load_backstretch(_girls)
        if _bs_model is not None:
            _pB = strengths_from_model(_bs_model, entries, recent, elo_state,
                                       tactics_ctx=tactics_ctx, narabi_ctx=narabi_ctx,
                                       venue_code=venue_code)
            if _pB:
                from src.model.development_branches import branch_mixture
                _mix, _dists = branch_mixture(strengths, _narabi_lines, _pB)
    probs = _mix or (corrected_trifecta_probs(strengths, narabi_pos, _sp.himo_params,
                                              lines=_narabi_lines)
                     if _sp.himo_params else all_trifecta_probs(strengths))
    prob_source = "分岐混合" if _mix else ("紐補正" if _sp.himo_params else "素のPL")

    # 一着固定の合成オッズ: 車cを1着に固定した三連単(c,*,*)全通りを合成した実効オッズ
    #   合成オッズ_c = 1 / Σ(1/オッズ)   … cを1着で買い切ったときの実効配当倍率
    # モデル勝率 win_prob と突き合わせると市場が各車の「勝ち」を割高/割安に見ているか分かる。
    #   win_ev = win_prob × 合成オッズ（>1でモデル的に割安=1着を過小評価）
    def _synth_1st(car: int) -> float | None:
        inv = sum(1.0 / o for k, o in odds.items() if k[0] == car and o and o > 0)
        return round(1.0 / inv, 2) if inv > 0 else None

    # 並び予想のライン境界 → {車番: (line_id, pos_in_line)}。ガールズは lines が空で全て None。
    _line_of = {car: (li, pi)
                for li, line in enumerate((narabi_ctx or {}).get("lines") or [])
                for pi, car in enumerate(line)}
    # 選手の位置別成績・戦法系の実績（表示用）。**as-of（当該レース日より前）で集計する**。
    # ガールズはライン概念が無いので男子だけ。飛びつき成功率は位置取り推移が無く出せない。
    lstats = {}
    if not _girls:
        try:
            from src.features.rider_line_stats import compute_line_stats
            from src.collect.gamboo_schedule import kaisai_race_date
            lstats = compute_line_stats(hist_db, kaisai_race_date(day_code).isoformat())
        except Exception:
            lstats = {}

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
        _lp = _line_of.get(e.car_number)
        riders.append({
            "car": e.car_number, "name": e.rider_name,
            "score": e.racing_score, "leg": e.leg_type,
            "class_rank": e.class_rank,                        # 級班（男子はS1/S2/A1/A2/A3）
            "line_id": _lp[0] if _lp else None,                # 所属ライン（並び予想由来）
            "pos_in_line": _lp[1] if _lp else None,            # 0=ライン先頭 1=番手 2=3番手
            "narabi_leg": (narabi_ctx or {}).get("legs", {}).get(e.car_number),
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
            # ライン内位置別成績＋戦法系の実績（表示専用・as-of）。n が薄いものは
            # rate が None で返る（分母は入っている）
            "line_stats": lstats.get(e.rider_name),
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
    # 的中率/回収率の実測はガールズout-of-sample。男子では出さない（pocket_stats=False）
    reference = None
    if _sp.pocket_stats:
        from src.betting.reference_formation import build_reference
        reference = build_reference(strengths, narabi_pos, venue_code,
                                    bool(riders and riders[0].get("home")))

    # 男子のライン別の強さ（ライン形式の並び表示＋数値化）。ガールズは lines が空で None。
    # 数値は上の probs（このレースのモデル三連単分布）から積み上げるので、表示の
    # ライン評価と買い目確率が食い違わない。
    from src.model.line_strength import build_lines
    line_strength = build_lines(
        (narabi_ctx or {}).get("lines") or [], strengths, probs,
        names={e.car_number: e.rider_name for e in entries},
        scores={e.car_number: e.racing_score for e in entries},
        legs=(narabi_ctx or {}).get("legs") or {},
        classes={e.car_number: e.class_rank for e in entries},
        seri=(narabi_ctx or {}).get("seri") or [])

    # 展開予想（記者の並び予想の隊列＋モデルの一言読み）。ガールズは並び通りになるとは限らない。
    development = None
    dev_branches = None          # 男子の展開分岐（下の _backstretch 確定後に作る）
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
        # 明示的な展開パターン（複数）: 勝者の隊列位置でモデルの1着確率を分解。P合計=1。
        _defs = [("先行・主導権型", "🚴", lambda p: p == 0),
                 ("番手抜け出し型", "🔗", lambda p: p == 1),
                 ("中団抜け出し型(捲り)", "🌀", lambda p: p is not None and 2 <= p <= 4),
                 ("後方一気型", "⚡", lambda p: p is not None and p >= 5)]
        patterns = []
        _assigned = set()
        for _lbl, _ic, _cond in _defs:
            _cars = [(c, strengths[c]) for c in strengths
                     if _cond((narabi_pos or {}).get(c))]
            if not _cars:
                continue
            _cars.sort(key=lambda cp: -cp[1])
            patterns.append({"label": _lbl, "icon": _ic,
                             "prob": round(sum(p for _, p in _cars), 4),
                             "lead_car": _cars[0][0], "cars": [c for c, _ in _cars],
                             "has_fav": any(c == _fav for c, _ in _cars)})
            _assigned |= {c for c, _ in _cars}
        _rest = [(c, strengths[c]) for c in strengths if c not in _assigned]
        if _rest:
            _rest.sort(key=lambda cp: -cp[1])
            patterns.append({"label": "位置不明", "icon": "・",
                             "prob": round(sum(p for _, p in _rest), 4),
                             "lead_car": _rest[0][0], "cars": [c for c, _ in _rest],
                             "has_fav": any(c == _fav for c, _ in _rest)})
        patterns.sort(key=lambda x: -x["prob"])
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
        # 推定主導権（展開AI＝最終バック先頭Bの予測）。男女で別モデル・別特徴。
        #   ガールズ 55.2%的中（記者予想22.1%）/ 男子 62.5%（記者先頭49.4%・B回数最大56.3%を
        #   5/5foldで上回る）。男子は主導権がほぼ決着構造を決めるので表示価値が高い。
        _backstretch = None
        # P(B) は本表示の分布を作る時に既に引いてある（_pB）。ここで引き直すと
        # 表示している分岐確率と分布の条件付けが食い違う恐れがあるので使い回す。
        pB = _pB
        if pB is None and _girls:
            from src.model.backstretch import load_backstretch
            _bs = load_backstretch(_girls)
            if _bs is not None:
                pB = strengths_from_model(_bs, entries, recent, elo_state,
                                          tactics_ctx=tactics_ctx, narabi_ctx=narabi_ctx,
                                          venue_code=venue_code)
        if pB:
            _rk = sorted(pB.items(), key=lambda kv: -kv[1])
            _rfront = _order[0] if _order else None
            _backstretch = {
                "lead_car": _rk[0][0], "lead_p": round(_rk[0][1], 4),
                "second_car": _rk[1][0] if len(_rk) > 1 else None,
                "second_p": round(_rk[1][1], 4) if len(_rk) > 1 else None,
                "reporter_front": _rfront,
                "diverges": bool(_rfront is not None and _rk[0][0] != _rfront),
                # walk-forward out-of-sample 実測（男女で別値）
                "hit_rate": 0.552 if _girls else 0.625,
                "probs": {int(c): round(p, 4) for c, p in _rk},
            }
        # 展開分岐（男子）: 主導権の候補ごとに条件付き着順分布と買い目を出す。
        # **_backstretch が確定した後**でなければならない（分岐の確率はP(B)そのもの）。
        if not _girls and _backstretch and (narabi_ctx or {}).get("lines"):
            from src.model.development_branches import build_branches
            dev_branches = build_branches(
                strengths, narabi_ctx["lines"],
                {int(c): p for c, p in (_backstretch.get("probs") or {}).items()},
                names={e.car_number: e.rider_name for e in entries},
                # 合成オッズは締切間近の更新でオッズが揃った時だけ入る（発売前は None）
                odds=odds or None,
                # 本表示の分布を作った時の分岐をそのまま渡す（二重計算で食い違わせない）
                dists=_dists)

        # ペース読み（先行型=レース内でb_countが最多の40%以上の人数。スケール非依存で analyze_pace_composition と同一定義）。
        # 先行型が多いほどハイペース化し逃げが飛び捲り・差しが台頭（±5pt程度）。表示専用・着順には非影響。
        _bvals = [((recent.get(e.car_number).b_count or 0) if recent.get(e.car_number) else 0) for e in entries]
        _mx = max(_bvals) if _bvals else 0
        _nfront = sum(1 for v in _bvals if _mx >= 2 and v >= 0.4 * _mx)
        # 決まり手と主導権の信頼度は「ペース×バンク」の実測から引く。バンクの影響は
        # ペースの3〜4倍あり（逃げ率 333m 35.3% vs 500m 12.8% に対しペース差は6.8pt）、
        # 無視すると短走路で逃げを過小評価する。標本の薄いセルはバンク→ペース→全体へ後退。
        _plv = "ハイ" if _nfront >= 4 else ("ミドル〜ハイ" if _nfront == 3 else "スロー〜ミドル")
        from src.model.kimarite_hint import hint as _khint, pace_note as _knote
        _h = _khint(_nfront, venue_code, is_girls=_girls)
        if _h:
            _pkm, _brel = _h["kimarite_hint"], _h.get("b_reliability")
            _pnote = _knote(_nfront, _pkm)
            _basis, _bank = _h.get("basis"), _h.get("bank")
        else:                      # 統計JSONが無い場合の従来値（ペースのみ）
            _pkm = {"逃": 17, "捲": 49, "差": 34} if _nfront >= 4 else (
                {"逃": 18, "捲": 49, "差": 34} if _nfront == 3 else {"逃": 23, "捲": 46, "差": 31})
            _brel = {"rentai": 55, "gaiji": 35, "note": "主導権はやや不安定"}
            _pnote, _basis, _bank = "捲り・差しがやや優勢", None, None
        pace = {"n_front": _nfront, "level": _plv, "note": _pnote,
                "kimarite_hint": _pkm, "b_reliability": _brel,
                "basis": _basis, "bank": _bank}
        development = {
            "source": "並び予想（記者予想の隊列, 発走前確定情報）",
            "line": line, "fav": _fav, "marker": _marker,
            "patterns": patterns,                  # 明示的な展開パターン（複数・確率付き）
            "backstretch": _backstretch,           # 推定主導権(展開AI・最終バック先頭B)
            "pace": pace,                          # ペース読み(先行型の数→ペース→決まり手傾向)
            "note": read,
            "caveat": "ガールズはライン概念が薄く、実際の主導権は並び予想通りにならないことも多い（予想先頭の実バック取得率≒20%）。展開パターンの確率はモデル1着確率を勝者の隊列位置で分解したもの。",
        }

    # バンク特性（諸元＋統計的な有利脚質）。静的テーブル＋統計JSONのみ＝DB非依存。
    from src.features.bank_profile import profile as _bank_profile
    bank_profile = _bank_profile(venue_code, is_girls=_girls)

    # 展開6パターンの上位3つ（発生確率＋紐の内訳）。履歴統計JSONのみ参照＝DB非依存。
    from src.model.dev_patterns import build_dev_patterns
    _pace_lv = ((development or {}).get("pace") or {}).get("level", "")
    dev_patterns = build_dev_patterns(rt.top1_win_prob, _pace_lv, riders,
                                      is_girls=_girls)

    from datetime import datetime, timezone, timedelta
    from src.collect.gamboo_schedule import kaisai_race_date as _krd
    jst = timezone(timedelta(hours=9))
    return {
        # date は開催日コードから引く（前後1日を同時表示するため全レースに必須。
        # (date, venue, race_no) が一意キー＝同一会場で日をまたぐと R番号が重複するため）
        "date": _krd(day_code).isoformat(),
        # venue_code も持たせる（会場名は「伊東競輪」表記でテーブルの「伊東温泉」と一致せず、
        # 名前からバンク長を引くと取りこぼす。後段は必ずこのコードを使う）
        "venue": venue, "venue_code": venue_code, "race_no": race_no, "deadline": deadline,
        "is_girls": _girls, "field_size": len(entries),
        # 男子用。級班はレース単位の階層（番手の価値が級班で反転するため表示に要る）、
        # lines は並び予想のライン境界（ガールズは空になる）
        "class_group": "/".join(sorted({e.class_rank for e in entries if e.class_rank})) or None,
        "lines": (narabi_ctx or {}).get("lines") or [],
        # 競り＝同じ位置を争うグループ。並び予想でカッコに入っている選手。
        # 直列に描くと番手が1人に確定して見えてしまうので、表示でカッコを復元する。
        "seri": (narabi_ctx or {}).get("seri") or [],
        "line_strength": line_strength,        # 男子: ライン別の強さ（表示用）
        "dev_branches": dev_branches,          # 男子: 展開分岐＋分岐ごとの買い目
        # 開催格(G1/F2等)・開催名・レース名。混戦度は格とレース名（決勝/予選）で傾向が変わる
        "grade": meta.get("grade"), "meet_name": meta.get("meet_name"),
        "race_name": meta.get("race_name"),
        "race_type": rt.label, "top1_prob": round(rt.top1_win_prob, 4),
        "entropy": round(rt.entropy_norm, 4), "source": source,
        # 三連単分布の作り方（分岐混合 / 紐補正 / 素のPL）。どれで出しているかが
        # ライン決着の読み方を変える（混合は実測どおり56%、紐補正は35%と過小）
        "prob_source": prob_source,
        # 波乱確率＝万車券率（払戻1万円以上）。旧表示の 1-top1_prob は「◎が1着を
        # 外す確率」で、実測の約2倍を波乱として出していた（src/model/upset.py 参照）。
        # 検証を通していない層（男子9車）では None が返る＝表示しない
        "upset_prob": man_prob(probs, is_girls=_girls, field_size=len(entries)),
        "riders": riders, "top_trifecta": top_tri, "ev": ev,
        "role": race_role,                          # レース種別(勝ち上がり)+勝ち上がり条件
        "development": development,                 # 展開予想（並び予想の隊列＋モデル読み）
        "dev_patterns": dev_patterns,               # 展開6パターンの上位3つ（実測分岐比）
        "bank_profile": bank_profile,               # バンク諸元＋統計的な有利脚質（表示用）
        "combos": combos,                          # 全210通り（オッズテーブル用）
        "reference": reference,                    # 参考フォーメーション（実弾非推奨・回収率<100%）
        "h2h": h2h,                                # 出走者同士の過去対戦成績マトリクス
        "has_odds": bool(odds),
        # オッズを取った時刻（JST・HH:MM）。合成オッズがいつ時点かを表示するため。
        # オッズが無ければ None（発売前）
        "odds_at": odds_at if odds else None,
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
