"""印(◎○▲△×)ベースの狙い撃ち戦略を実測する。

印 = モデル1着確率の降順に ◎(0) ○(1) ▲(2) △(3) ×(4)。out-of-sample(build_records)。

  軸堅: ◎頭固定の10点フォーメーション
    A) ◎→{○,▲}→{○,▲,△,×}  （2着n2×3着n4, 重複除外=6点）
    B) ◎→{△,×}→{○,▲}       （4点）  … 計10点
   この10点から「エッジのある点」を EV(=model_prob×odds) / エッジ比(model_prob/市場フェア確率q)
   で選定して実現ROIを見る（全10点買いも比較）。

  混戦: ◎が1着でない（◎が勝てない）買い目＝1着≠◎ の三連単でエッジがあるか。
   全部/エッジ選定のROIを、対照として「◎1着の買い目」と比較する。

  PYTHONIOENCODING=utf-8 python scripts/analyze_mark_formations.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.features.tactics_features import TACTIC_NAMES
from src.model.evaluate import time_split
from src.backtest.bucket_analysis import build_records
from src.backtest.selection import group_by_race, _first_place_probs
from src.ev.market import implied_trifecta_probs

BUDGET = 1000


def _settle(chosen_by_race):
    """{race_id:[ComboRecord]} を均等/ドッチングで決済。"""
    n_races = n_hits = n_bets = 0
    eq_stake = eq_ret = du_stake = du_ret = 0.0
    for recs in chosen_by_race.values():
        if not recs:
            continue
        n_races += 1
        n_bets += len(recs)
        inv = sum(1.0 / r.odds for r in recs if r.odds > 0)
        eq_stake += len(recs) * 100
        du_stake += BUDGET
        win = next((r for r in recs if r.is_win), None)
        if win:
            n_hits += 1
            eq_ret += win.payout
            if inv > 0:
                du_ret += (BUDGET * (1.0 / win.odds) / inv) * (win.payout / 100.0)
    if n_races == 0:
        return None
    return {"n_races": n_races, "n_bets": n_bets, "n_hits": n_hits,
            "hit_rate": n_hits / n_races, "pts": n_bets / n_races,
            "roi_eq": eq_ret / eq_stake if eq_stake else 0,
            "roi_du": du_ret / du_stake if du_stake else 0}


def _pr(label, s):
    if not s:
        print(f"{label:<30}(該当なし)"); return
    print(f"{label:<30}{s['n_races']:>6}{s['pts']:>7.1f}{s['n_hits']:>6}"
          f"{s['hit_rate']*100:>7.1f}%{s['roi_eq']*100:>8.1f}%{s['roi_du']*100:>9.1f}%")


def main():
    ap = argparse.ArgumentParser(description="印フォーメーション戦略のエッジ検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--test-frac", type=float, default=0.35)
    args = ap.parse_args()

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    feats31 = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
    samples = augment_samples(base, args.db, feats31)
    train, test = time_split(samples, args.test_frac)
    model = train_gbdt(train)
    records = build_records(args.db, model, [s.race_id for s in test])

    # レースごとに前処理: 印ランク・combo→record・市場フェア確率q
    races = {}
    for rid, recs in group_by_race(records).items():
        fp = _first_place_probs(recs)
        ranked = [c for c, _ in sorted(fp.items(), key=lambda kv: -kv[1])]
        if len(ranked) < 5:
            continue
        odds = {r.combo: r.odds for r in recs}
        races[rid] = {
            "recs": recs, "rtype": recs[0].race_type, "ranked": ranked,
            "rec_of": {r.combo: r for r in recs},
            "q": implied_trifecta_probs(odds),
        }
    from collections import Counter
    cnt = Counter(v["rtype"] for v in races.values())
    print(f"検証 {len(races)}レース（軸堅{cnt.get('軸堅',0)}/標準{cnt.get('標準',0)}/混戦{cnt.get('混戦',0)}）"
          f" / 31特徴lambdarank / out-of-sample")

    def hdr(title):
        print(f"\n【{title}】")
        print(f"{'戦略':<30}{'R数':>6}{'点/R':>7}{'的中':>6}{'的中率':>8}{'ROI均等':>8}{'ROIドッチ':>9}")

    def formation_10(info):
        """◎頭固定の10点(A+B)の combo集合を返す。"""
        h, m, s3, sk, bt = info["ranked"][:5]     # ◎○▲△×
        A = {(h, b, c) for b in (m, s3) for c in (m, s3, sk, bt) if len({h, b, c}) == 3}
        B = {(h, b, c) for b in (sk, bt) for c in (m, s3) if len({h, b, c}) == 3}
        return A | B

    def collect(rtypes, combo_pred, ev_thr=None, ratio=None):
        """条件に合う買い目を {race_id:[rec]} で返す。combo_pred(info,combo)->bool。"""
        out = {}
        for rid, info in races.items():
            if info["rtype"] not in rtypes:
                continue
            picks = []
            for r in info["recs"]:
                if not combo_pred(info, r.combo):
                    continue
                if ev_thr is not None and r.ev < ev_thr:
                    continue
                if ratio is not None:
                    q = info["q"].get(r.combo, 0.0)
                    if q <= 0 or r.model_prob / q < ratio:
                        continue
                picks.append(r)
            if picks:
                out[rid] = picks
        return out

    # ============ 軸堅: ◎頭固定10点フォーメーション ============
    hdr("軸堅: ◎頭固定10点フォーメーション（A:◎→○▲→○▲△× / B:◎→△×→○▲）")
    in10 = lambda info, combo: combo in formation_10(info)
    _pr("全10点買い", _settle(collect({"軸堅"}, in10)))
    for thr in (1.1, 1.2, 1.3, 1.5):
        _pr(f"10点内 EV≥{thr} 選定", _settle(collect({"軸堅"}, in10, ev_thr=thr)))
    for ratio in (1.3, 1.5, 2.0):
        _pr(f"10点内 エッジ比p/q≥{ratio}", _settle(collect({"軸堅"}, in10, ratio=ratio)))

    # ============ 混戦: ◎が1着でない買い目 ============
    hdr("混戦: ◎が1着でない三連単（1着≠◎）にエッジがあるか")
    anti = lambda info, combo: combo[0] != info["ranked"][0]
    fav = lambda info, combo: combo[0] == info["ranked"][0]
    _pr("1着≠◎ 全部買い", _settle(collect({"混戦"}, anti)))
    for thr in (1.1, 1.3, 1.5):
        _pr(f"1着≠◎ かつ EV≥{thr}", _settle(collect({"混戦"}, anti, ev_thr=thr)))
    for ratio in (1.3, 1.5, 2.0):
        _pr(f"1着≠◎ かつ p/q≥{ratio}", _settle(collect({"混戦"}, anti, ratio=ratio)))
    print("  --- 対照（混戦で ◎が1着 の買い目）---")
    _pr("1着=◎ 全部買い", _settle(collect({"混戦"}, fav)))
    _pr("1着=◎ かつ EV≥1.3", _settle(collect({"混戦"}, fav, ev_thr=1.3)))

    print("\n※ ROIは控除率≈25%のため<100%が基本。>100%かつ的中数が十分なら"
          "「相対的にマシ／エッジ候補」。少的中は分散ノイズに注意。")


if __name__ == "__main__":
    main()
