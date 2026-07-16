"""波乱確率（◎=モデル1着本命が1着を取れない確率 = 1 - P(◎1着)）の条件分析。

「勝率に織り込まれている」前提の上で:
  (1) モデルの波乱確率は較正されているか（予測波乱率 vs 実波乱率の全体ギャップ・信頼度カーブ）
  (2) どんな事前条件で波乱が起きやすい/起きにくいか（記述）
  (3) 条件別に「実波乱率 − 予測波乱率」のギャップが出るか（＝モデルの過信/過小、回収率のヒント）
を walk-forward(out-of-sample) で測る。ギャップ正=モデルが波乱を過小評価(◎過大評価→頭固定は危険)、
負=モデルが波乱を過大評価(◎は見た目より堅い→頭固定寄せ)。全条件で per-fold 符号一致数も出して頑健性を見る。

  PYTHONIOENCODING=utf-8 python scripts/analyze_upset.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.model.race_type import classify_race, _entropy_norm
from src.features.tactics_features import TACTIC_NAMES
from src.features.rider_narabi import NARABI_KEYS, compute_narabi_features
from src.features import venue_region as vr
from src.features import venue_meta as vm
from src.backtest.walkforward import fold_boundaries


def _aux(db: str):
    """(leg[(rid,car)], pref[(rid,car)], venue[rid]) を返す。"""
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    leg, pref = {}, {}
    for rid, car, lt, pf in c.execute("SELECT race_id,car_number,leg_type,prefecture FROM entries"):
        leg[(rid, car)] = lt
        pref[(rid, car)] = pf
    venue = {rid: v for rid, v in c.execute("SELECT race_id,venue_code FROM races")}
    c.close()
    return leg, pref, venue


def _bands(p):
    for lo, hi in [(0, .30), (.30, .40), (.40, .50), (.50, .60), (.60, .70), (.70, 1.01)]:
        if lo <= p < hi:
            return f"{int(lo*100):02d}-{int(hi*100):02d}%"
    return "?"


def _leg_group(lt: str | None) -> str:
    if not lt:
        return "不明"
    if any(k in lt for k in ("逃", "先")):
        return "逃/先行"
    if "捲" in lt or "自在" in lt or "両" in lt:
        return "捲/自在/両"
    if any(k in lt for k in ("追", "差", "マー")):
        return "追/差/マーク"
    return lt


def main():
    ap = argparse.ArgumentParser(description="波乱確率の条件分析")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    feats = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES) + list(NARABI_KEYS)
    samples = augment_samples(base, args.db, feats)
    leg, pref, venue = _aux(args.db)
    narabi = compute_narabi_features(args.db)
    bounds = fold_boundaries(len(samples), n_folds=args.folds, warmup_frac=0.40, window="expanding")

    recs = []   # per-race: dict(fold, p_fav, upset, conds{...})
    for fi, (a, b, c) in enumerate(bounds):
        model = train_gbdt(samples[a:b])
        for s in samples[b:c]:
            st = model.strengths(s.X, s.car_numbers)
            fav = max(st, key=st.get)
            ranked = sorted(st.values(), reverse=True)
            p_fav = ranked[0]
            gap2 = p_fav - (ranked[1] if len(ranked) > 1 else 0.0)
            rid = s.race_id
            nb = narabi.get((rid, fav), {})
            npos = nb.get("narabi_pos")
            v = venue.get(rid, "")
            conds = {
                "race_type": classify_race(st).label,
                "p_fav帯": _bands(p_fav),
                "entropy帯": ("低<.5" if _entropy_norm(st) < .5 else ("中.5-.75" if _entropy_norm(st) <= .75 else "高>.75")),
                "2着との差": ("僅差<.10" if gap2 < .10 else ("中.10-.25" if gap2 < .25 else "大>.25")),
                "◎脚質": _leg_group(leg.get((rid, fav))),
                "◎予想先頭": ("予想先頭" if nb.get("narabi_lead") else "非先頭"),
                "◎並び位置": ("先頭(0)" if npos == 0 else ("中団(2-4)" if (npos is not None and 2 <= npos <= 4) else ("番手(1)" if npos == 1 else "後方/無"))),
                "◎地元": ("地元" if vr.is_home_pref(pref.get((rid, fav)), v) else "非地元"),
                "バンク": (f"{vm.bank_length(v)}m" if vm.bank_length(v) else "?"),
            }
            recs.append({"fold": fi, "p_fav": p_fav, "upset": int(s.order[0] != fav), "conds": conds})

    n = len(recs)
    pred = sum(1 - r["p_fav"] for r in recs) / n
    act = sum(r["upset"] for r in recs) / n
    print(f"波乱確率＝1−P(◎1着)。out-of-sample {n}レース / {len(bounds)}fold\n")
    print(f"【全体較正】予測波乱率 {pred*100:.1f}% vs 実波乱率 {act*100:.1f}%  ギャップ {(act-pred)*100:+.1f}pt")
    print("  （0付近＝◎の負けは勝率に正しく織込み済み。＋＝モデルが波乱を過小評価）\n")

    # 予測波乱率デシルの信頼度カーブ
    print("【信頼度カーブ】予測波乱率帯ごとの実波乱率")
    dec = defaultdict(list)
    for r in recs:
        dec[min(9, int((1 - r["p_fav"]) * 10))].append(r["upset"])
    print(f"  {'予測波乱率':>10}{'n':>7}{'実波乱率':>10}{'gap':>8}")
    for k in sorted(dec):
        us = dec[k]
        pr = k * 10 + 5
        print(f"  {k*10:>3}-{k*10+10:>3}%{len(us):>7}{sum(us)/len(us)*100:>9.1f}%{(sum(us)/len(us)-pr/100)*100:>+7.1f}")

    # 条件別: 予測 vs 実 と fold符号一致
    for cond in ["race_type", "p_fav帯", "entropy帯", "2着との差", "◎脚質",
                 "◎予想先頭", "◎並び位置", "◎地元", "バンク"]:
        groups = defaultdict(list)
        for r in recs:
            groups[r["conds"][cond]].append(r)
        print(f"\n【条件: {cond}】 予測波乱率 vs 実波乱率（gap=実−予測, +で◎過大評価=波乱多い）")
        print(f"  {'値':<12}{'n':>6}{'予測':>8}{'実績':>8}{'gap':>8}  fold一致")
        rows = []
        for val, rs in groups.items():
            if len(rs) < 40:
                continue
            pv = sum(1 - x["p_fav"] for x in rs) / len(rs)
            av = sum(x["upset"] for x in rs) / len(rs)
            # per-fold gap 符号一致（n>=10のfoldのみ）
            agree = 0; tot = 0
            for fi in range(len(bounds)):
                fr = [x for x in rs if x["fold"] == fi]
                if len(fr) >= 10:
                    tot += 1
                    fpv = sum(1 - x["p_fav"] for x in fr) / len(fr)
                    fav_ = sum(x["upset"] for x in fr) / len(fr)
                    if (fav_ - fpv) * (av - pv) > 0:
                        agree += 1
            rows.append((av - pv, val, len(rs), pv, av, agree, tot))
        for g, val, ln, pv, av, agree, tot in sorted(rows, key=lambda z: -abs(z[0])):
            print(f"  {val:<12}{ln:>6}{pv*100:>7.1f}%{av*100:>7.1f}%{g*100:>+7.1f}  {agree}/{tot}")


if __name__ == "__main__":
    main()
