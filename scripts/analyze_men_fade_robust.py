"""男子・中波乱の「追込◎フェード98.2%」の頑健性検証（fold別・点数・消し方・組み合わせ）。

過去に単一分割の好結果(混戦◎2着固定114%等)が walk-forward で非再現だった前例があるため、
追込◎フェードが fold をまたいで安定して100%近辺かを確認する。安定なら初のエッジ候補。

  PYTHONIOENCODING=utf-8 python scripts/analyze_men_fade_robust.py
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
from src.features.rider_narabi import compute_narabi_features
from src.model.feature_sets import men_features
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS
from src.model.upset import threshold_for
from src.backtest.walkforward import fold_boundaries


def _aux(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    leg = {}
    for rid, car, lt in c.execute("SELECT race_id,car_number,leg_type FROM entries"):
        leg[(rid, car)] = lt
    payout = {}
    for rid, combo, pay in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta"):
        payout[rid] = (combo, pay)
    tmp = defaultdict(list)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        tmp[rid].append((li, pi, car))
    lines = {}
    for rid, rows in tmp.items():
        mx = max(li for li, _, _ in rows)
        ls = [[] for _ in range(mx + 1)]
        for li, pi, car in sorted(rows, key=lambda x: (x[0], x[1])):
            ls[li].append(car)
        lines[rid] = [x for x in ls if x]
    c.close()
    return leg, payout, lines


def _is_bante(fav, lines):
    for m in lines or []:
        if fav in m:
            return len(m) > 1 and m.index(fav) == 1
    return False


def _topk(dist, k, pred):
    it = [(o, p) for o, p in dist.items() if pred(o)]
    it.sort(key=lambda op: -op[1])
    return set(o for o, _ in it[:k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    thr = threshold_for(False, 7)
    base = load_samples(args.db, field_size=7, features=PL_FEATURES_FULL)
    samples = augment_samples(base, args.db, men_features())
    narabi = compute_narabi_features(args.db)
    leg, payout, lines = _aux(args.db)
    bounds = fold_boundaries(len(samples), n_folds=args.folds, warmup_frac=0.40, window="expanding")

    # 追込◎・中波乱のレコード（fold付き）
    recs = []
    for fi, (a, b, c) in enumerate(bounds):
        model = train_gbdt(samples[a:b])
        for s in samples[b:c]:
            if s.race_id not in payout:
                continue
            st = model.strengths(s.X, s.car_numbers)
            fav = max(st, key=st.get)
            if leg.get((s.race_id, fav)) != "追":
                continue
            npos = {cc: narabi.get((s.race_id, cc), {}).get("narabi_pos") for cc in s.car_numbers}
            ln = lines.get(s.race_id)
            dist = corrected_trifecta_probs(st, npos, MEN_PARAMS, lines=ln)
            up = sum(p for p in dist.values() if p <= thr)
            if not (0.20 <= up < 0.30):
                continue
            wc = tuple(int(x) for x in payout[s.race_id][0].split("-"))
            recs.append({"fold": fi, "fav": fav, "wc": wc, "pay": payout[s.race_id][1],
                         "dist": dist, "bante": _is_bante(fav, ln)})

    def roi(rs, k, pred_name):
        pred = {"head_out": (lambda o, f: o[0] != f),
                "erase": (lambda o, f: f not in o)}[pred_name]
        ret = pts = hit = 0
        for r in rs:
            buys = _topk(r["dist"], k, lambda o: pred(o, r["fav"]))
            pts += len(buys)
            if r["wc"] in buys:
                ret += r["pay"]; hit += 1
        n = len(rs)
        return (ret / (pts / n * 100 * n) * 100 if pts else 0), hit / n * 100, n

    print(f"男子・中波乱・追込◎ {len(recs)}レース\n")
    print("【fold別 追込◎フェード(◎頭外し8点)】")
    for fi in range(len(bounds)):
        rs = [r for r in recs if r["fold"] == fi]
        if len(rs) < 20:
            continue
        r8 = roi(rs, 8, "head_out")
        print(f"   fold{fi}: n={r8[2]:>4}  回収 {r8[0]:>6.1f}%  的中 {r8[1]:.1f}%")
    allr = roi(recs, 8, "head_out")
    print(f"   全体 : n={allr[2]}  回収 {allr[0]:.1f}%  的中 {allr[1]:.1f}%")

    print("\n【点数感度（◎頭外し）】")
    for k in (4, 6, 8, 12):
        r = roi(recs, k, "head_out")
        print(f"   {k}点: 回収 {r[0]:.1f}%  的中 {r[1]:.1f}%")
    print("\n【消し方の比較(8点)】")
    for pn, lbl in (("head_out", "◎頭外し(◎≠1着)"), ("erase", "◎完全消し(◎∉3着)")):
        r = roi(recs, 8, pn)
        print(f"   {lbl:<18} 回収 {r[0]:.1f}%  的中 {r[1]:.1f}%")
    print("\n【追込◎ かつ 番手(ライン2番手)】")
    bt = [r for r in recs if r["bante"]]
    if bt:
        r = roi(bt, 8, "head_out")
        print(f"   n={r[2]}  ◎頭外し8点 回収 {r[0]:.1f}%  的中 {r[1]:.1f}%")
        # fold別
        for fi in range(len(bounds)):
            rs = [r for r in bt if r["fold"] == fi]
            if len(rs) >= 15:
                rr = roi(rs, 8, "head_out")
                print(f"     fold{fi}: n={rr[2]:>3} 回収 {rr[0]:.1f}%")


if __name__ == "__main__":
    main()
