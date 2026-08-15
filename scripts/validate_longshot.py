"""万車券率が高いレースで穴狙いの買い目が成立するかを実測する（男子7車）。

**先に分かっていること（この検証の前提）**:
`validate_upset_prob.py` で、市場が100倍以上に値付けした目にモデルは合計50.33%の確率を
置いていたが、実際にその帯で決まったのは28.92%だった。つまり**モデルは裾（人気薄）の
確率を実勢より大きく見積もっている**。EV = モデル確率 × オッズ で穴を選ぶと、この
過大評価がそのままEVの過大評価になる。「妙味あり」と出た目ほどモデル誤差を拾う。
さらに控除率25%なのでROIの上限は75%。Phase 5（`men_keirin_plan.md` 4.15）でも
男子に黒字ゾーンは見つかっていない。

**それでも測る理由**: 「裾が全体として過大」でも、万車券率が高い層に限れば
別の挙動をする可能性は事前には否定できない。主張ではなく数字で確かめる。

**事前登録した採否基準（後から緩めない）**:
  推奨として出せるのは「**レース単位ブートストラップ95%区間の下限がROI 100%を超える**」
  戦略だけ。点推定が100%を超えても区間が100%をまたぐなら「エッジあり」とは言わない。
  同一レースの210点は互いに強く従属する（的中は多くて1点）ので、リサンプルの単位は
  買い目ではなく**レース**にする。

  PYTHONIOENCODING=utf-8 python scripts/validate_longshot.py
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.model.feature_sets import men_features
from src.model.plackett_luce import all_trifecta_probs
from src.model.upset import threshold_for
from src.backtest.walkforward import fold_boundaries

STAKE = 100


def load_market(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    odds: dict[str, dict[str, float]] = {}
    for rid, combo, o in c.execute("SELECT race_id,combo,odds FROM odds_final_trifecta"):
        odds.setdefault(rid, {})[combo] = o
    pay = {r: (combo, p) for r, combo, p in
           c.execute("SELECT race_id,combo,payout FROM payouts_trifecta")}
    c.close()
    return odds, pay


# 買い目の型。いずれも「1レースあたり何点買うか」を固定して比較可能にする。
def strategies(probs, odds, n=5):
    """{戦略名: [買う目...]}。目は "1-2-3" 形式。"""
    ev = {k: p * odds[k] for k, p in probs.items() if k in odds}
    out = {}
    byp = sorted(probs.items(), key=lambda kv: -kv[1])
    out["本命5点(確率順)"] = [k for k, _ in byp[:n]]
    out["EV上位5点(全帯)"] = [k for k, _ in sorted(ev.items(), key=lambda kv: -kv[1])[:n]]
    for lo in (50, 100):
        cand = {k: v for k, v in ev.items() if odds[k] >= lo}
        out[f"EV上位5点(オッズ{lo}倍以上)"] = [
            k for k, _ in sorted(cand.items(), key=lambda kv: -kv[1])[:n]]
        cand2 = {k: v for k, v in probs.items() if k in odds and odds[k] >= lo}
        out[f"確率上位5点(オッズ{lo}倍以上)"] = [
            k for k, _ in sorted(cand2.items(), key=lambda kv: -kv[1])[:n]]
    return out


def boot(races: list[tuple], n_boot=2000, seed=0) -> tuple:
    """races=[(賭け金, 払戻)]。**レース単位**でリサンプルして95%区間を出す。"""
    if not races:
        return (None, None, None)
    rnd = random.Random(seed)
    tot_s = sum(s for s, _ in races)
    tot_r = sum(r for _, r in races)
    point = tot_r / tot_s if tot_s else 0.0
    n = len(races)
    vals = []
    for _ in range(n_boot):
        s = r = 0.0
        for _ in range(n):
            a, b = races[rnd.randrange(n)]
            s += a; r += b
        if s:
            vals.append(r / s)
    vals.sort()
    return point, vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


def main() -> None:
    ap = argparse.ArgumentParser(description="万車券率が高い層での穴狙い検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--points", type=int, default=5, help="1レースあたりの点数")
    args = ap.parse_args()

    odds_all, pay = load_market(args.db)
    thr = threshold_for(False, 7)
    print(f"万車券率のしきい値(男子7車): {thr}")

    feats = men_features()
    raw = load_samples(args.db, field_size=[7], features=PL_FEATURES_FULL)
    smp = augment_samples(raw, args.db, feats)
    print(f"サンプル {len(smp):,}（7車のみ・万車券率の検証を通した層）")

    # rows[(戦略, 層)] = [(賭け金, 払戻)]
    rows: dict[tuple, list] = {}
    ups: list[float] = []
    per_race: list[tuple] = []
    for a, b, c2 in fold_boundaries(len(smp), n_folds=args.folds,
                                    warmup_frac=0.40, window="expanding"):
        model = train_gbdt(smp[a:b])
        for s in smp[b:c2]:
            if s.race_id not in pay or s.race_id not in odds_all:
                continue
            st = model.strengths(s.X, s.car_numbers)
            if not st:
                continue
            probs = {f"{k[0]}-{k[1]}-{k[2]}": v for k, v in all_trifecta_probs(st).items()}
            up = sum(p for p in probs.values() if p <= thr)
            ups.append(up)
            per_race.append((s.race_id, up, probs))

    ups.sort()
    q = lambda f: ups[int(f * (len(ups) - 1))]
    cuts = {"下位30%(堅い)": (0.0, q(.30)), "中位40%": (q(.30), q(.70)),
            "上位30%(荒れやすい)": (q(.70), 1.01),
            "上位10%(最も荒れやすい)": (q(.90), 1.01)}
    print(f"万車券率の分布: 30%点 {q(.30)*100:.1f}% / 70%点 {q(.70)*100:.1f}% "
          f"/ 90%点 {q(.90)*100:.1f}%")

    for rid, up, probs in per_race:
        od = odds_all[rid]
        win_combo, payout = pay[rid]
        sts = strategies(probs, od, n=args.points)
        for name, combos in sts.items():
            ret = payout if win_combo in combos else 0
            stake = STAKE * len(combos)
            for lab, (lo, hi) in cuts.items():
                if lo <= up < hi:
                    rows.setdefault((name, lab), []).append((stake, ret))

    names = list(strategies(per_race[0][2], odds_all[per_race[0][0]], args.points).keys())
    for lab in cuts:
        print(f"\n=== {lab} ===")
        print(f"{'戦略':<26}{'R数':>6}{'的中':>6}{'ROI':>9}{'95%区間':>20}")
        for nm in names:
            rr = rows.get((nm, lab)) or []
            if not rr:
                continue
            hits = sum(1 for _, r in rr if r > 0)
            p, lo, hi = boot(rr)
            mark = "  ← 下限>100%" if (lo is not None and lo > 1.0) else ""
            print(f"{nm:<26}{len(rr):>6}{hits:>6}{p*100:>8.1f}%"
                  f"{f'[{lo*100:.1f}%, {hi*100:.1f}%]':>20}{mark}")

    print("\n控除率25%のためROIの上限は75%。区間の下限が100%を超えた戦略のみ推奨に足る。")


if __name__ == "__main__":
    main()
